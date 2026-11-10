# DTFA: setup and run notes

This file records what was set up, changed, downloaded and checked in the DTFA repo during the session on **27 Oct 2026**. It's separate from the authors' `README.md`, which is unchanged.

Upstream repo: https://github.com/huruo1010/DTFA (commit `1986cb3`)
Paper: *A lightweight framework for robust object detection in adverse weather based on dual-teacher feature alignment*, Neurocomputing 2026.

---

## 1. Current state at a glance

| Item | Status |
|---|---|
| `model_data/yolox_s.pth` (COCO-pretrained YOLOX-s) | Downloaded |
| `model_data/SPT.pth`, `model_data/IRT.pth` (authors' teacher weights) | Added by you from Baidu; both verified to load fully (215/215 tensors) |
| VOC-Clean / VOC-Fog datasets under `datasets/` | Built from public VOC2012; every file checksum-verified |
| Code fixes so the repo runs on Colab (recent PyTorch / Python 3.12+) | Done and tested |
| `DTFA_colab.zip` (1.8 GB, everything needed for Colab) | Rebuilt and verified |
| `DTFA_colab.ipynb` (Colab notebook) | In this folder and in your Google Drive |
| Colab 1-epoch test run | Started at about 14:53 IST; you stopped the session before it finished, so the result wasn't checked |
| Full 100-epoch training | **Not started yet** |

---

## 2. What was downloaded

Baidu Pan, Google Drive, the VOC host server and HuggingFace were all unreachable from both machines, so everything came from GitHub.

| File | Source | Size |
|---|---|---|
| `model_data/yolox_s.pth` | `github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_s.pth` | 36 MB |
| `VOCtrainval_11-May-2012.zip` (Pascal VOC 2012) | `github.com/ultralytics/assets/releases/download/v0.0.0/` | 1.95 GB |
| `model_data/SPT.pth`, `model_data/IRT.pth` | Downloaded by you from the authors' Baidu link | 57 MB + 41 MB |

The VOC zip was only extracted into the computer's temporary workspace. Only the images DTFA needs were copied into the repo.

---

## 3. Dataset preparation

The paper's VOC-FOG set (from TogetherNet) is only on Baidu, so the fog images were **generated locally** from clean VOC2012 images by the new script `tools/prepare_data.py`. It uses the same atmospheric-scattering formula as the repo's `utils/syn_fog.py`:

```
I_fog = I · t + A · (1 − t),   t = exp(−β · d)
d = sqrt(max(w,h)) − 0.04 · (distance from image centre),   A = 0.5,   β ∈ [0.07, 0.12]
```

- β is drawn per image from a seed derived from the image ID, so the output is **deterministic**. Rebuilding on Colab or another machine gives byte-identical images.
- The train/test split is the authors' own, from `datasets/data_info/*.txt`.
- **1,065 of the 9,578 training images** in the authors' split come from VOC2012's *test* images, which aren't publicly downloadable. They were dropped from both `train_clear.txt` and `train_fog.txt`, which are kept line-aligned because the AWD loader pairs fog and clean images by line number. All **2,129 test images** are present.

Resulting layout:

```
datasets/
├── data_info/                 # annotation lists (paths now relative; 8,513 train / 2,129 val lines)
├── voc_clean/
│   ├── train/JPEGImages/      # 8,513 images
│   └── test/JPEGImages/       # 2,129 images  + Annotations/*.xml + test.txt
├── voc_fog/                   # same structure, synthetic fog
└── fog_preview.jpg            # side-by-side clean vs fog example
```

To regenerate (for example on Colab, instead of uploading the zip):

```bash
curl -sSL -o voc.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip
unzip -q voc.zip 'VOCdevkit/VOC2012/JPEGImages/*' 'VOCdevkit/VOC2012/Annotations/*'
python tools/prepare_data.py --voc VOCdevkit/VOC2012
```

---

## 4. Code changes

### Files added

| File | Why |
|---|---|
| `nets/detection/yolo.py`, `nets/detection/yolo_training.py`, `__init__.py` | `train/pretrain_SPT.py` and `pretrain_IRT.py` import `nets.detection`, which the authors never published. SPT here is the same YOLOX-s as the student (`nets/AWD/yolo.py`), returning only detection outputs. |
| `nets/reconstruction/model.py`, `__init__.py` | `pretrain_IRT.py` imports `nets.reconstruction.model`, which was also missing. IRT here is an auto-encoder: the ShallowNet CSPDarknet encoder plus the U-Net decoder from `nets/yolo.py`, with the final Tanh removed. |
| `tools/prepare_data.py` | Builds VOC-Clean / VOC-Fog (section 3). Re-runs skip finished files and rebuild empty or truncated ones. |
| `DTFA_colab.ipynb` | Colab notebook (section 6). |
| `.gitignore` | Ignores `logs/`, generated datasets, `*.pth` weights and the zip. |
| `README_SESSION.md` | This file. |

You only need the `nets/detection` and `nets/reconstruction` modules to train the teachers yourself. With the authors' `SPT.pth` / `IRT.pth` they aren't used.

### Files modified

| File | Change |
|---|---|
| `train/train_AWD.py` | `Cuda = torch.cuda.is_available()` instead of hard-coded `True`. Settings can be overridden with environment variables (table below). Teacher checkpoints whose keys start `backbone.backbone.` are remapped to ShallowNet's `backbone.`. Prints `matched N/215 teacher tensors` and warns on a partial match. |
| `yolo.py`, `RTTS_yolo.py` | `model_path` comes from `MODEL_PATH` (default `model_data/AWD.pth`) instead of the authors' server path `/hy-tmp/...`. `cuda` is auto-detected. |
| `get_map.py` | Test folder comes from `TEST_DIR` (default `datasets/voc_fog/test`); output folder from `MAP_OUT`. |
| `RTTS_get_map.py` | Test folder comes from `TEST_DIR` (default `datasets/RTTS`). |
| `utils/dataloader.py`, `utils/AWD_dataloader.py` | Removed an unused `from turtle import clear`, which crashes on machines without tkinter (Colab). |
| `datasets/data_info/*.txt` | Removed the `/hy-tmp/DTFA-main/` prefix and dropped the 1,065 unavailable training images. The originals can be restored with `git checkout datasets/data_info`. |

Untouched: the model architecture, losses, training loop, hyper-parameters and evaluation code. `requirements.txt` is unchanged, but **don't install it on Colab**: its pins (torch 1.10, Pillow 7.2) don't install on current Python. Colab's preinstalled PyTorch works.

### Environment variables for `train/train_AWD.py`

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS` | 100 | Total epochs |
| `INIT_EPOCH` | 0 | Starting epoch, for resuming |
| `MODEL_PATH` | `model_data/yolox_s.pth` | Student initial / resume weights |
| `SPT_PATH`, `IRT_PATH` | `model_data/SPT.pth`, `model_data/IRT.pth` | Teacher weights |
| `BATCH_SIZE` | 8 | Batch size |
| `SAVE_PERIOD` | 50 | Save a checkpoint every N epochs (use 5 on Colab) |
| `NUM_WORKERS` | 4 | Data-loader workers (use 2 on Colab) |
| `TRAIN_ANN`, `VAL_ANN` | `datasets/data_info/train_fog.txt`, `val_fog.txt` | Annotation lists |

---

## 5. Verification done

- **End-to-end test in a cloud sandbox** (CPU, tiny subset): `pretrain_SPT.py` → `pretrain_IRT.py` → `train_AWD.py` → `get_map.py` all ran without errors.
- **Authors' teacher weights:** `SPT.pth` matches the full student detector (542/542 tensors) and ShallowNet (215/215). `IRT.pth` matches ShallowNet (215/215). One student epoch on a subset ran correctly with them.
- **Architecture check:** a student checkpoint loaded into `nets/yolo.py` (used by `yolo.py` for testing) gives *identical* outputs to `nets/AWD/yolo.py`, so evaluation uses the trained model correctly.
- **Dataset integrity:** all 25,548 dataset files on your Mac were checksum-compared with an independent clean rebuild. After the fixes in section 7, they are identical.
- **On Colab (T4):** both teachers load (215/215). Training runs at about **1.19 s per iteration**, 1,064 iterations per epoch, so about **24 min per epoch** including validation.

---

## 6. How to run on Google Colab

The notebook is in Google Drive and has been run there before. Steps for a fresh session:

1. **Runtime → Change runtime type → T4 GPU.**
2. Run **"mount Drive"** and approve access.
3. Run **"Unpack the repo"**. It unzips `My Drive/DTFA_colab.zip` to `/content/DTFA`, copies the repaired images from `My Drive/DTFA_fix/`, points `logs/` at `My Drive/DTFA_logs` (so checkpoints survive disconnects), and prints the GPU name.
4. Skip the commented "Alternative" cell.
5. Run **"Teachers (SPT / IRT)"**. It should print `found -> skipping` for both.
6. Run **"Train the student"**: `!SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py`
   - Check that `matched 215/215 teacher tensors` appears twice.
   - Checkpoints are saved to `My Drive/DTFA_logs/epNNN-loss…-val_loss….pth`.
   - On CUDA out-of-memory, add `BATCH_SIZE=4`.
7. **Resume after a disconnect:** re-run steps 2, 3 and 5, then
   ```
   !INIT_EPOCH=35 MODEL_PATH=logs/ep035-<rest of name>.pth SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py
   ```
8. **Test:** run the last cell. It copies the newest checkpoint to `model_data/AWD.pth` and prints per-class AP and mAP@0.5 on the foggy test set, then the clean test set. Results are in `My Drive/DTFA_logs/fog_map_out/` and `clean_map_out/`.

**Time estimate:** about 24 min/epoch × 100 epochs ≈ **40 hours** on a free T4. Free Colab sessions end after about 12 hours, so expect about 4 sessions with resumes. To save Drive space, a `SAVE_PERIOD` of 5 is a reasonable balance; each checkpoint is about 36 MB.

### Extra cells added to the Drive notebook

| Cell | Purpose |
|---|---|
| "Claude: 1-epoch test run…" | Trains 1 epoch, moves the checkpoint to `logs/test_run/` and evaluates it there. It was running when the session stopped. To check it, open the cell's output or look in `My Drive/DTFA_logs/test_run/`. |
| "Claude: repair 4 images…" | One-time repair of the images in section 7. It downloads VOC2012 on Colab, fixes the files and saves them to `My Drive/DTFA_fix/`. It's already done; you don't need to run it again. |
| "Unpack the repo" (modified) | Now restores `My Drive/DTFA_fix/datasets` after unzipping. |

Both test-related cells can be deleted once you're happy with the setup.

---

## 7. Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Colab training crashed with `UnidentifiedImageError … 2008_000217.jpg` | The first `prepare_data.py` run on the Mac was interrupted, leaving 3 empty images and 1 truncated one: `voc_clean/train/2007_009331`, `2008_000217`, `2008_000027`, `voc_fog/train/2007_009327`. Re-runs skipped them because they "existed". | Re-created on the Mac and inside `DTFA_colab.zip`, repaired on Colab and saved to `My Drive/DTFA_fix/`. `prepare_data.py` now validates existing files. All 21,284 images verified on Colab. |
| Teacher-training scripts couldn't import | `nets/detection` and `nets/reconstruction` are missing upstream | Added (section 4) |
| `turtle` / `tkinter` import error on Colab | Stray unused import in the data loaders | Removed |
| Hard-coded `/hy-tmp/DTFA-main/...` paths and `Cuda = True` | Authors' server setup | Relative paths, environment variables, GPU auto-detect |

---

## 8. Differences from the paper's setup

Expect mAP close to, but not exactly matching, the paper:

1. Fog images are synthesized locally with the same physical model, not the authors' VOC-FOG files.
2. 8,513 instead of 9,578 training images; the test set is complete.
3. Recent PyTorch/Python on Colab instead of PyTorch 1.10.1 / Python 3.8.

---

## 9. Housekeeping on your Mac

- About 6 GB free on the disk at the end of the session. After the zip is safely in Drive, you can delete:
  - `DTFA_colab.zip` (1.8 GB)
  - `_to_delete/`: an emptied leftover temp file from a failed `zip` attempt, and `mac_md5.txt` (the checksum list)
  - `.git/stale-maintenance-lock-claude`: an empty git lock file moved aside after a `git fetch`
- No git commits or pushes were made. `git status` shows all the changes listed above.
- The local `DTFA_colab.ipynb` doesn't have the Drive notebook's extra cells or the `DTFA_fix` restore line. If you re-upload the (now fixed) local zip, that line isn't needed.

---

## 10. How DTFA works (short version)

- **Student (AWD):** YOLOX-s with a deformable conv and a transformer block on the deepest backbone stage, and self-calibrated conv blocks in the PAFPN neck. It has a decoupled head with box, objectness and 5 classes (bicycle, bus, car, motorbike, person).
- **Teachers (frozen, fed the paired clean image):**
  - **IRT**, a clean-image reconstruction auto-encoder, provides weather-invariant structure.
  - **SPT**, a detector trained on clean images, provides task semantics.
  - Only their shallow features (`dark2` 64×160×160, `dark3` 128×80×80) are used.
- **AFB (Adaptive Feature Bridging, `utils/AFB/loss.py`):**
  1. Fuses the two shallow scales into one 128×160×160 map, for both student and teacher.
  2. Randomly masks about 65% of the student map's positions.
  3. A small conv generator must reconstruct the teacher's map from it (masked generative distillation, MSE).
- **Loss:** `0.2 · L_det + L_rc (IRT) + L_dc (SPT)`. `L_det` is the YOLOX loss: SimOTA assignment, IoU loss ×5, focal objectness loss, BCE class loss.
- **Inference:** only the student runs. It produces 8,400 candidates per 640×640 image, keeps those with objectness × class score above the threshold, and applies per-class NMS. The teachers and AFB add no test-time cost.
