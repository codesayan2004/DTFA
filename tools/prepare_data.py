"""Build the VOC-Clean / VOC-Fog folders that DTFA expects, from the public VOC2012 release.

    python tools/prepare_data.py --voc VOCdevkit/VOC2012

Creates (relative to the repo root, using the split in datasets/data_info/*.txt):

    datasets/voc_clean/{train,test}/JPEGImages/*.jpg    clean images (copied)
    datasets/voc_fog/{train,test}/JPEGImages/*.jpg      synthetic fog (atmospheric scattering model,
                                                        same formula as utils/syn_fog.py)
    datasets/voc_{clean,fog}/test/Annotations/*.xml     VOC annotations used by get_map.py
    datasets/voc_{clean,fog}/test/test.txt              image ids used by get_map.py

NOTE: the paper's VOC-FOG set (from TogetherNet) is only distributed via Baidu Pan. The fog made
here follows the same physical model but is not pixel-identical, so mAP will differ slightly.
Images of the original split that are not in the public VOC2012 trainval release are dropped
from datasets/data_info/train_*.txt (about 1,065 training images; the 2,129 test images are all present).
Re-running is safe: complete output files are skipped; empty or truncated ones are rebuilt.
"""
import argparse
import os
import shutil
import zlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = {"train": "train_clear.txt", "test": "val_clear.txt"}


def add_voc_fog(img, beta, A=0.5):
    """Identical to utils.syn_fog.add_voc_fog, with beta passed in for reproducibility."""
    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_wh = np.sqrt(max(w, h))
    depth_map = np.maximum(-0.04 * distances + max_wh, 0.1)
    transmission = np.exp(-beta * depth_map)[..., None]
    hazy = img * transmission + A * (1 - transmission)
    return (np.clip(hazy, 0, 1) * 255).astype(np.uint8)


def _done(path, image=True):
    """True only for a complete output file (an interrupted run can leave empty/partial files)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if image:
        try:
            with Image.open(path) as im:
                im.load()
        except Exception:
            return False
    return True


def work(job):
    image_id, split, voc, beta_lo, beta_hi, A = job
    src = os.path.join(voc, "JPEGImages", image_id + ".jpg")
    clean = os.path.join(ROOT, "datasets/voc_clean", split, "JPEGImages", image_id + ".jpg")
    fog = os.path.join(ROOT, "datasets/voc_fog", split, "JPEGImages", image_id + ".jpg")
    if not _done(clean):
        shutil.copyfile(src, clean)
    if not _done(fog):
        rng = np.random.default_rng(zlib.crc32(image_id.encode()))  # deterministic per image
        beta = rng.uniform(beta_lo, beta_hi)
        img = np.array(Image.open(src).convert("RGB"))
        Image.fromarray(add_voc_fog(img, beta, A)).save(fog, quality=95)
    if split == "test":
        for ds in ("voc_clean", "voc_fog"):
            dst = os.path.join(ROOT, "datasets", ds, "test", "Annotations", image_id + ".xml")
            if not _done(dst, image=False):
                shutil.copyfile(os.path.join(voc, "Annotations", image_id + ".xml"), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", required=True, help="path to VOCdevkit/VOC2012")
    ap.add_argument("--beta", type=float, nargs=2, default=(0.07, 0.12), help="fog density range")
    ap.add_argument("--A", type=float, default=0.5, help="atmospheric light")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    jobs = []
    for split, listfile in SPLITS.items():
        ids = [os.path.splitext(os.path.basename(l.split()[0]))[0]
               for l in open(os.path.join(ROOT, "datasets/data_info", listfile)) if l.strip()]
        # Some ids in the authors' split come from the VOC2012 *test* images, which are not publicly
        # downloadable. Drop them from the clear AND fog lists (kept line-aligned, as the AWD loader
        # pairs fog/clear images by line index).
        missing = {i for i in ids if not os.path.exists(os.path.join(args.voc, "JPEGImages", i + ".jpg"))}
        if missing:
            print(f"[{split}] {len(missing)} of {len(ids)} images are not in the public VOC2012 release -> dropped")
            ids = [i for i in ids if i not in missing]
            for name in (listfile, listfile.replace("clear", "fog")):
                path = os.path.join(ROOT, "datasets/data_info", name)
                lines = [l for l in open(path) if l.strip()
                         and os.path.splitext(os.path.basename(l.split()[0]))[0] not in missing]
                with open(path, "w") as f:
                    f.write("".join(l if l.endswith("\n") else l + "\n" for l in lines))
        for ds in ("voc_clean", "voc_fog"):
            os.makedirs(os.path.join(ROOT, "datasets", ds, split, "JPEGImages"), exist_ok=True)
            if split == "test":
                os.makedirs(os.path.join(ROOT, "datasets", ds, split, "Annotations"), exist_ok=True)
                with open(os.path.join(ROOT, "datasets", ds, split, "test.txt"), "w") as f:
                    f.write("\n".join(ids) + "\n")
        jobs += [(i, split, args.voc, args.beta[0], args.beta[1], args.A) for i in ids]

    with Pool(args.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, jobs, chunksize=32), 1):
            if n % 1000 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} images done", flush=True)


if __name__ == "__main__":
    main()
