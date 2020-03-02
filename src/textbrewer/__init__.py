__version__ = "0.1.5"

from .distillation import BasicTrainer
from .distillation import BasicDistiller
from .distillation import GeneralDistiller
from .distillation import MultiTeacherDistiller
from .distillation import MultiTaskDistiller

from .configurations import TrainingConfig, DistillationConfig

from .presets import FEATURES
from .presets import ADAPTOR_KEYS
from .presets import KD_LOSS_MAP, MATCH_LOSS_MAP, PROJ_MAP
from .presets import WEIGHT_SCHEDULER, TEMPERATURE_SCHEDULER
from .presets import register_new

Distillers = {
    'Basic': BasicDistiller,
    'General': GeneralDistiller,
    'MultiTeacher': MultiTeacherDistiller,
    'MultiTask': MultiTaskDistiller,
    'Train': BasicTrainer
}