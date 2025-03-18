import os
import pinecone
import pandas as pd
import torch
import re 

from tqdm import tqdm
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from sklearn.utils import shuffle
from .model_handler import resnet_handler, vit_handler
from .reranking_feedback import feedback_manager

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_ = load_dotenv(override=True)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
data_train_path = './DATASET_2/train'
data_dev_path = './DATASET_2/dev'
data_test_path = './DATASET_2/test'

vit_handler.init_vit_model()
resnet_handler.init_resnet_model()


    
def init_pinecone(index_name):
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    index = pinecone.Index(index_name)

    return index

def create_img_df(folder_path):
    # FOLDER_PATH = './DATASET_2/test/'
    path = []
    class_label = []
    id = []
    pattern = r"(?<=img_)\d+_\d+"

    for des in os.listdir(folder_path):
        img_lst = os.listdir(os.path.join(folder_path,des))

        for img_path in img_lst:
            path.append(os.path.join(folder_path,des,img_path))

            match = re.search(pattern, img_path)
            id.append(match.group())
        class_label.extend([des]*len(img_lst))
        
    # id = np.arange(1, 1+ total_img)


    data = {
            'id': id,
            'path': path,
            'class_label':class_label
            }
    
    df = pd.DataFrame(data)
    df['id'] = df['id'].apply(str)
    df['class_label'] = df['class_label'].apply(str)

    return df

def combine_df():
    df_train = create_img_df(data_train_path)
    df_dev = create_img_df(data_dev_path)

    df_combined = pd.concat([df_train, df_dev], ignore_index=True)
    df_combined = shuffle(df_combined)
    return df_combined



def extract_img_feature_resnet(image_paths: list):
    batch_features = []
    for image_path in image_paths:
        with torch.no_grad():
            img_data = resnet_handler.load_and_preprocess_img(image_path)
            outputs = resnet_handler.resnet_model(img_data)
            # img_feature = resnet_handler.model.predict(img_data)
            # print('feature', img_feature[0])
            # print('shape: ', img_feature[0].shape)
        batch_features.append(outputs[0].tolist())
    return batch_features

def extract_img_feature_vit(image_paths: list):
    batch_features = []
    for image_path in image_paths:
        with torch.no_grad():
            img_data = vit_handler.load_and_preprocess_img(image_path)
            outputs = vit_handler.vit_model(pixel_values=img_data)
            cls_features = outputs.last_hidden_state[:, 0, :]

        batch_features.append(cls_features[0].tolist())
    return batch_features



def upserting_pinecone(df,index,model, batch_size = 32):

    fail_ids = []
    for batch_start in tqdm(range(0, len(df), batch_size)):
        try:
            batch_end = min(len(df), batch_start + batch_size)

            # print('path batch: ', paths_batch)
            ids_batch = df['id'][batch_start : batch_end].tolist()
            paths_batch = df['path'][batch_start : batch_end].tolist()
            metadata_class_batch = df['class_label'][batch_start :batch_end].tolist()
            # metadata_path_batch = df['path'][batch_start :batch_end].tolist()

            if model == 'resnet':
                batch_features = extract_img_feature_resnet(paths_batch)
            else:
                batch_features = extract_img_feature_vit(paths_batch)

            record = [(ids, features, {'img_path': img_path, 'img_class': img_class}) for ids, features, img_path, img_class in zip(ids_batch, batch_features, paths_batch, metadata_class_batch)]
            index.upsert(vectors = record, namespace='ns1')
        except Exception as e:
            print(f'Faild upserting {e}')
            fail_ids.append(ids_batch)

    if len(fail_ids)  != 0:
        for id in fail_ids:
            print(f"{id}")

def get_top_k_img(query_img_path,index, model, top_k = 10, include_metadata = True):
    if model == 'resnet':
        query_embed = extract_img_feature_resnet([query_img_path])
    elif model == 'vit':
        query_embed = extract_img_feature_vit([query_img_path])

    else: print("Not supported model")

    res = index.query(query_embed[0], top_k=top_k, include_metadata=include_metadata, namespace='ns1')
    return res['matches']


def get_top_k_img_with_filter(query_img_path, index, model, filter_list, top_k=10, include_value=True,
                              include_metadata=True):
    filter_condition = {"img_path": {"$in": filter_list}}

    if model == 'resnet':
        query_embed = extract_img_feature_resnet([query_img_path])
    else:
        query_embed = extract_img_feature_vit([query_img_path])

    res = index.query(query_embed[0], top_k=top_k, include_values=include_value, include_metadata=include_metadata,
                      namespace='ns1', filter=filter_condition)
    return res['matches']


def update_pinecone(index):
    df = pd.read_csv('./train_dev_combined.csv')

    for idx, row in df.iterrows():
        index.update(
            id=row['id'],
            set_metadata={'img_class': row['class_label'],
                          'img_path': row['path'].split('\\')[-1]},
            namespace="ns1"
        )


# feature_extraction.py
def get_top_k_img_with_weights(query_img_path, index, model_type, top_k=10, include_metadata=True):
    try:
        # Extract features
        if model_type == 'resnet':
            query_features = torch.tensor(extract_img_feature_resnet([query_img_path])[0])
        else:  # vit
            query_features = torch.tensor(extract_img_feature_vit([query_img_path])[0])

        # Apply weights
        weighted_query = feedback_manager.apply_weights_to_query(query_features, model_type)

        # Convert to list for Pinecone
        query_vector = weighted_query.tolist()

        # Query Pinecone
        res = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace='ns1'
        )

        logger.debug(f"Successfully queried with weights for {model_type}")
        return res['matches']
    except Exception as e:
        logger.error(f"Error in get_top_k_img_with_weights: {e}")
        # Fallback to original query
        return get_top_k_img(query_img_path, index, model_type, top_k, include_metadata)
