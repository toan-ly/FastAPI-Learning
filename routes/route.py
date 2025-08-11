import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import File, UploadFile, APIRouter, HTTPException

from schemas.schema import Response
from config.model_config import ModelConfig
from models.predictor import Predictor

router = APIRouter()
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight_path=str(ModelConfig.MODEL_WEIGHT),
    device=ModelConfig.DEVICE
)

@router.post('/predict')
async def predict(file_upload: UploadFile = File(...)):
    """Prediction endpoint for image classification."""
    response = await predictor.predict(file_upload.file)
    return Response(**response)

