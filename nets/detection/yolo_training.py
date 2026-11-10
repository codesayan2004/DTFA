# Added: alias so train/pretrain_*.py can import nets.detection.yolo_training.
from nets.yolo_training import *  # noqa: F401,F403
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,  # noqa: F401
                                weights_init)
