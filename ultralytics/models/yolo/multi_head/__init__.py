# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .train import MultiHeadTrainer
from .val import MultiHeadValidator
from .predict import MultiHeadPredictor

__all__ = "MultiHeadTrainer", "MultiHeadValidator", "MultiHeadPredictor"
