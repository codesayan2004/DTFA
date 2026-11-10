# Added: the original repo imports nets.reconstruction.model but does not ship it.
# IRT teacher = clear-image auto-encoder. The encoder is the same CSPDarknet used by
# ShallowNet (nets/AWD/shallow_darknet.py) so that its weights can be loaded into the
# frozen IRT branch of train/train_AWD.py; the decoder is the U-Net style decoder from
# nets/yolo.py. Parameter names are "backbone.backbone.*" so that yolox_s.pth
# initialises the encoder.
import torch.nn as nn

from nets.AWD.shallow_darknet import CSPDarknet
from nets.yolo import Decoder


class _Encoder(nn.Module):
    def __init__(self, depth=0.33, width=0.5):
        super().__init__()
        self.backbone = CSPDarknet(depth, width)


class reconstruction(nn.Module):
    def __init__(self, depth=0.33, width=0.5):
        super().__init__()
        self.backbone = _Encoder(depth, width)
        self.decoder = Decoder(512, 3)
        # Targets are ImageNet-normalised images (not in [-1, 1]), so drop the final Tanh.
        self.decoder.conv_4[2] = nn.Identity()

    def forward(self, x):
        f = self.backbone.backbone(x)
        return self.decoder(f["dark5"], f["dark4"], f["dark3"])
