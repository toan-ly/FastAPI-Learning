import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class DataConfig:
    N_CLASSES = 2
    IM_SIZE = 64
    ID2LABEL = {0: 'cat', 1: 'dog'}
    LABEL2ID = {'cat': 0, 'dog': 1}
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

class ModelConfig:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_NAME = 'resnet18'
    MODEL_WEIGHT = ROOT_DIR / 'models' / 'weights' / 'weights.pt'
    DEVICE = 'cpu'