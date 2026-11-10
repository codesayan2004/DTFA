"""Visualise what Dynamic Bridging Tuning learned: severity map W, IRT-trust map alpha, teacher
disagreement D, for one or more foggy images.

    python tools/visualize_dbt.py --student logs/ep100-....pth --dbt logs/dbt_ep100.pth \
        datasets/voc_fog/test/JPEGImages/2007_000027.jpg [more images ...] --out logs/dbt_vis

The clean image (needed by the teachers) is found by replacing voc_fog with voc_clean in the path,
or given with --clean when visualising a single image. Run from the repo root.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets.AWD.shallow_darknet import ShallowNet
from nets.AWD.yolo import YoloBody
from utils.DBT.loss import DBTLoss
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image


def load_teacher(path, device):
    net = ShallowNet()
    sd = net.state_dict()
    for k, v in torch.load(path, map_location="cpu").items():
        k = k.replace("backbone.backbone.", "backbone.", 1)
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
    net.load_state_dict(sd)
    return net.eval().to(device)


def to_tensor(path, device):
    img = cvtColor(Image.open(path))
    data = resize_image(img, (640, 640), True)  # letterbox, as in evaluation
    arr = np.array(data, dtype=np.float32)
    x = torch.from_numpy(np.transpose(preprocess_input(arr.copy()), (2, 0, 1))).unsqueeze(0).to(device)
    return x, np.array(data, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="foggy (adverse-weather) images")
    ap.add_argument("--student", required=True, help="student checkpoint from the same DBT run (logs/epNNN-...pth)")
    ap.add_argument("--dbt", required=True, help="gate checkpoint from the same epoch (logs/dbt_epNNN.pth)")
    ap.add_argument("--clean", default="", help="clean image (only when a single image is given)")
    ap.add_argument("--spt", default="model_data/SPT.pth")
    ap.add_argument("--irt", default="model_data/IRT.pth")
    ap.add_argument("--classes", default="model_data/classes.txt")
    ap.add_argument("--out", default="logs/dbt_vis")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.dbt, map_location="cpu")
    dbt = DBTLoss(**ckpt.get("config", {}))
    dbt.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    dbt = dbt.eval().to(device)

    _, num_classes = get_classes(args.classes)
    student = YoloBody(num_classes, "s")
    student.load_state_dict(torch.load(args.student, map_location="cpu"))
    student = student.eval().to(device)
    irt, spt = load_teacher(args.irt, device), load_teacher(args.spt, device)

    os.makedirs(args.out, exist_ok=True)
    for path in args.images:
        clean = args.clean if (args.clean and len(args.images) == 1) else path.replace("voc_fog", "voc_clean")
        if not os.path.exists(clean):
            print("skip (no clean image found):", path)
            continue
        x_a, vis_a = to_tensor(path, device)
        x_c, vis_c = to_tensor(clean, device)
        with torch.no_grad():
            _, g = student(x_a)
            r, d = irt(x_c), spt(x_c)
            out = dbt((g[1], g[0]), (r[1], r[0]), (d[1], d[0]), x_a, None)

        def up(m):
            return F.interpolate(m, size=(640, 640), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        panels = [("adverse input", vis_a, None), ("clean (teachers' input)", vis_c, None)]
        if out["W"] is not None:
            panels.append(("severity W (1 = severe)", up(out["W"]), (0, 1)))
        panels.append(("alpha = trust in IRT\n(mean %.2f)" % out["alpha"].mean().item(), up(out["alpha"]), (0, 1)))
        if out["D"] is not None:
            panels.append(("teacher disagreement D", up(out["D"]), None))

        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
        for ax, (title, im, rng) in zip(axes, panels):
            if rng is None and im.ndim == 3:
                ax.imshow(im)
            else:
                h = ax.imshow(im, cmap="magma" if "W" in title else "coolwarm" if "IRT" in title else "viridis",
                              vmin=None if rng is None else rng[0], vmax=None if rng is None else rng[1])
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        dst = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + "_dbt.png")
        fig.savefig(dst, dpi=130)
        plt.close(fig)
        print("saved", dst)


if __name__ == "__main__":
    main()
