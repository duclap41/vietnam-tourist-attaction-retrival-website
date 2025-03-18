import torch
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class ViT:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_vit_model()

    def init_vit_model(self):
        # Khởi tạo model và chuyển ngay lên device phù hợp
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        self.vit_model.eval()

        img_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_feature_extractor.image_mean, std=img_feature_extractor.image_std)
        ])

    def load_and_preprocess_img(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)  # Thêm chiều batch
        # Chuyển input lên cùng device với model
        img = img.to(self.device)
        return img
