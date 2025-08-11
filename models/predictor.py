import sys
import torch
from torch.nn import functional as F
import torchvision

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image

from utils.logger import Logger
from config.model_config import DataConfig
from .model import Model

LOGGER = Logger(__file__, log_file='predictor.log')
LOGGER.log.info("Starting Model Serving...")

class Predictor:
    def __init__(self, model_name: str, model_weight_path: str, device: str = 'cpu'):
        self.device = device
        self.model_name = model_name
        self.model_weight_path = model_weight_path

        self.load_model()
        self.create_transform()

    async def predict(self, img):
        pil_img = Image.open(img)
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')  # Convert RGBA to RGB if necessary

        transformed_img = self.transforms_(pil_img).unsqueeze(0)
        output = await self.inference(transformed_img)
        probs, best_prob, pred_id, pred_class = self.output_to_pred(output)

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, pred_id, pred_class)

        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

        return {
            'probs': probs,
            'best_prob': best_prob,
            'pred_id': pred_id,
            'pred_class': pred_class,
            'pred_name': self.model_name
        }

    async def inference(self, input):
        if not hasattr(self, 'loaded_model'):
            raise RuntimeError("Model not loaded, cannot do inference")
        input = input.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(input).cpu()

        return output

    def load_model(self):
        try:
            model = Model(n_classes=DataConfig.N_CLASSES)
            model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.loaded_model = model
        except Exception as e:
            LOGGER.log.error(f"Error loading model: {e}")
            return None

    def output_to_pred(self, output):
        probs = F.softmax(output, dim=1)
        best_prob = torch.max(probs, dim=1)[0].item()
        pred_id = torch.max(probs, dim=1)[1].item()
        pred_class = DataConfig.ID2LABEL[pred_id]

        return probs.squeeze().tolist(), round(best_prob, 4), pred_id, pred_class

    def create_transform(self):
        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((DataConfig.IM_SIZE, DataConfig.IM_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=DataConfig.NORMALIZE_MEAN, std=DataConfig.NORMALIZE_STD)
        ])


    