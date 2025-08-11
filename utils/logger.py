import sys
import logging

from pathlib import Path
sys.path.append(str(Path(__file__).parent))


from logging.handlers import RotatingFileHandler
from config.logging_config import LoggingConfig

class Logger:
    def __init__(self, name: str = "", log_level = logging.INFO, log_file = None) -> None:
        self.log = logging.getLogger(name)
        self.get_logger(log_level, log_file)

    def get_logger(self, log_level, log_file = None): 
        self.log.setLevel(log_level)
        self._init_formatter()

        if log_file:
            self._add_file_handler(LoggingConfig.LOG_DIR / log_file)
        else:
            self._add_stream_handler()

    def _init_formatter(self):
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _add_file_handler(self, log_file):
        file_handler = RotatingFileHandler(log_file, maxBytes=10**4, backupCount=10)
        file_handler.setFormatter(self.formatter)
        self.log.addHandler(file_handler)
    
    def _add_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)

    def log_model(self, predictor_name):
        self.log.info(f'Predictor name: {predictor_name}')

    def log_response(self, pred_prob, pred_id, pred_class):
        self.log.info(f'Pred Prob: {pred_prob}, Pred ID: {pred_id}, Pred Class: {pred_class}')