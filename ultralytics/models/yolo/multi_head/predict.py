import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class MultiHeadPredictor(DetectionPredictor): ...
