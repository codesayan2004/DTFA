# Added: the original repo imports nets.detection.yolo but does not ship it.
# SPT teacher = the same YOLOX-s architecture as the AWD student (nets/AWD/yolo.py),
# trained on clear images. Its forward returns only the detection outputs so that
# YOLOLoss can consume them directly in train/pretrain_SPT.py.
from nets.AWD.yolo import YoloBody as _AWDYoloBody


class YoloBody(_AWDYoloBody):
    def forward(self, x):
        outputs, _ = super().forward(x)
        return outputs
