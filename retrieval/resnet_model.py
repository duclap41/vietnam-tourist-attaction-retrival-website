import numpy as np
import torch
import warnings
from PIL import Image
from torchvision import models, transforms
# from torchvision.models import ResNet50_Weights

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ResNet:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_resnet_model()

    def init_resnet_model(self):
        try:
            self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.resnet_model.fc = torch.nn.Identity() # Loại bỏ lớp FC để lấy đặc trưng
            self.resnet_model = self.resnet_model.to(self.device) # Đưa model lên GPU
            self.resnet_model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Resize ảnh về kích thước 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error initializing ResNet model: {str(e)}")
            raise

    def load_and_preprocess_img(self, image_path):
        img = Image.open(image_path).convert('RGB')  # Ensure the image is RGB
        img = self.transform(img)
        img = img.unsqueeze(0)  # Thêm batch dimension
        img = img.to(self.device)  # Đưa input lên GPU

        return img
