# DTFA + Dynamic Bridging Tuning: How to Run

**Team SmashMasters:** Rohit Sah, Sayan Mandal, Sachin Kumar Tanti, Gaurav Singh, Karan

This repository contains:

1. **DTFA (baseline):** the dual-teacher feature-alignment detector for adverse weather from Hu *et al.*, *Neurocomputing* 671 (2026), "A lightweight framework for robust object detection in adverse weather based on dual-teacher feature alignment". The authors' original README is in [`README_DTFA_original.md`](README_DTFA_original.md).
2. **Dynamic Bridging Tuning (DBT, our extension):** a learned, spatially varying gate that decides per image region how much the student should trust each teacher (IRT vs. SPT). Method details are in [`DBT.md`](DBT.md), and the write-up is in `report.pdf` (source: `main.tex`).

Both are trained with the same script. `DBT=0` (default) trains DTFA and `DBT=1` trains DBT. The detector that is evaluated and deployed is identical in both cases (YOLOX-s, 14.11 M parameters).

---

## 1. Requirements

- **An NVIDIA GPU.** On a T4, one epoch over the full training set takes about 24 minutes, so 100 epochs is about 40 hours. Training on CPU or a Mac is only practical for quick tests.
- Python ≥ 3.10 with a recent PyTorch. Google Colab and Kaggle already have everything except possibly `tensorboard`.
- For a local install:

  ```bash
  pip install torch torchvision opencv-python numpy pillow scipy matplotlib tqdm tensorboard
  ```

  `requirements.txt` holds the authors' original pins (PyTorch 1.10, Python 3.8). Don't use it on a current Python; it won't install.

Run every command from the repository root.

---

## 2. Data and weights

### 2.1 What has to be in place

```
model_data/
  classes.txt        # bicycle, bus, car, motorbike, person
  yolox_s.pth        # COCO-pretrained YOLOX-s (student initialisation)
  SPT.pth            # Semantic Prior Teacher  (detector trained on clear images)
  IRT.pth            # Invariant Reconstruction Teacher (clear-image auto-encoder)
datasets/
  data_info/         # train_fog/train_clear/val_fog/val_clear.txt (paths + boxes)
  voc_clean/{train,test}/JPEGImages/      + test/Annotations/*.xml + test/test.txt
  voc_fog/{train,test}/JPEGImages/        + test/Annotations/*.xml + test/test.txt
```

All of these are already present in this folder. To rebuild them elsewhere:

| Item | Where it comes from |
|---|---|
| `yolox_s.pth` | `curl -L -o model_data/yolox_s.pth https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_s.pth` |
| `SPT.pth`, `IRT.pth` | Authors' pre-trained teachers: Baidu link in `README_DTFA_original.md` (code `b614`). Or train them yourself (section 3.4). |
| Datasets | Generated from PASCAL VOC 2012 by `tools/prepare_data.py` (below) |

### 2.2 Building VOC-Clean / VOC-Fog

```bash
curl -L -o voc.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip
unzip -q voc.zip 'VOCdevkit/VOC2012/JPEGImages/*' 'VOCdevkit/VOC2012/Annotations/*'
python tools/prepare_data.py --voc VOCdevkit/VOC2012
```

- Takes about 2 minutes.
- Fog is synthesised with the atmospheric scattering model, using the same formula as `utils/syn_fog.py`. Each image's fog density comes from a seed based on its ID, so every run produces identical images.
- The result is 8,513 training and 2,129 test images. 1,065 images of the authors' split are VOC test images that aren't publicly available, so they are dropped automatically.
- The script is safe to re-run: empty or truncated files are rebuilt.

---

## 3. Training

### 3.1 Local machine or Colab

```bash
export PYTHONPATH=$(pwd)

# DTFA baseline
SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py

# DTFA + Dynamic Bridging Tuning
DBT=1 SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py
```

You should see `matched 215/215 teacher tensors` printed twice. With DBT you'll also see a `Dynamic Bridging Tuning: ...` line, and a `DBT: alpha_mean=...` summary after each epoch.

Both runs write to `logs/`, so train the baseline and DBT in **separate copies of the repo**, or move `logs/` away between runs.

**Resume** after an interruption:

```bash
DBT=1 INIT_EPOCH=35 MODEL_PATH=logs/ep035-loss....pth SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py
```

The matching `logs/dbt_ep035.pth` (gate state) is found automatically.

**Shorter runs.** Set `TRAIN_SUBSET=3000 VAL_SUBSET=600` to train on a fixed random subset, and `EPOCHS`/`STOP_EPOCH` to shorten or split training. Use the same values for the baseline and DBT when comparing.

<details>
<summary>All settings (environment variables)</summary>

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS` | 100 | Total epochs; the learning-rate schedule spans this many |
| `STOP_EPOCH` | = EPOCHS | End this run early; use a multiple of `SAVE_PERIOD`, then resume |
| `INIT_EPOCH`, `MODEL_PATH` | 0, `model_data/yolox_s.pth` | Start epoch and weights (resume) |
| `BATCH_SIZE` | 8 | Use 4 on CUDA out-of-memory |
| `SAVE_PERIOD` | 50 | Save a checkpoint every N epochs (5 recommended) |
| `NUM_WORKERS` | 4 | Data-loader workers (2 on Colab) |
| `TRAIN_SUBSET`, `VAL_SUBSET` | 0 (all) | Train/validate on a fixed random subset |
| `TRAIN_ANN`, `VAL_ANN` | `datasets/data_info/train_fog.txt`, `val_fog.txt` | Annotation lists |
| `SPT_PATH`, `IRT_PATH` | `model_data/SPT.pth`, `IRT.pth` | Teacher weights |
| `DBT` | 0 | 1 = Dynamic Bridging Tuning |
| `DBT_MODE` | spatial | `spatial`, `image` (one weight per image), `fixed` (0.5/0.5, no gate) |
| `DBT_USE_S`, `DBT_USE_W`, `DBT_USE_D` | 1 | Gate inputs: student feature, severity map, teacher disagreement |
| `DBT_LAMBDA`, `DBT_ETA`, `DBT_GAMMA`, `DBT_MU` | 2, 1, 0.01, 0.1 | DBT loss weights (see `DBT.md`) |
| `DBT_SEV_SCALE`, `DBT_LR` | 0.25, 1e-3 | Severity pseudo-label scale; gate learning rate (Adam) |
| `DBT_PATH` | auto | Gate checkpoint to load |

</details>

### 3.2 Google Colab

1. Zip the repository on your Mac: `cd ~/Desktop && zip -r -0 DTFA_colab.zip DTFA -x 'DTFA/.git/*' 'DTFA/logs/*' '*/__pycache__/*'`. Upload `DTFA_colab.zip` to the top level of *My Drive*.
2. In a Colab notebook (Runtime → Change runtime type → **T4 GPU**), run:

   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !unzip -q -o /content/drive/MyDrive/DTFA_colab.zip -d /content/
   %cd /content/DTFA
   !mkdir -p /content/drive/MyDrive/DTFA_logs && rm -rf logs && ln -s /content/drive/MyDrive/DTFA_logs logs
   !pip -q install tensorboard
   import os; os.environ['PYTHONPATH'] = '/content/DTFA'
   !DBT=1 SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py
   ```

   Checkpoints are saved to `My Drive/DTFA_logs`, so they survive disconnects. Free Colab sessions end after about 12 hours; resume as in 3.1.

### 3.3 Kaggle (runs in the background; recommended for long training)

1. Create `DTFA_colab.zip` as in 3.2.
2. On Kaggle: **Datasets → New Dataset**, upload `DTFA_colab.zip`, and name it e.g. `dtfa-data`.
3. **New Notebook → File → Import Notebook →** `DTFA_kaggle.ipynb`, then **Add Input →** `dtfa-data`.
4. Settings: **Accelerator = GPU T4 x2**, Persistence = Files only.
5. In the parameter cell, set `DBT = 1` (or `0` for the baseline) and `STOP_EPOCH = 75`. Then **Save Version → Save & Run All**. You can close the browser while it runs.
6. Second chunk: *Add Input →* this notebook's previous output, set `STOP_EPOCH = 100`, and run again. It resumes automatically, then evaluates and draws the DBT maps.

### 3.4 Optional: train the teachers yourself

Needed only if `SPT.pth` and `IRT.pth` are unavailable.

```bash
python train/pretrain_SPT.py --num_workers 2 --save_period 5 --logs_dir logs/SPT/
python train/pretrain_IRT.py --num_workers 2 --save_period 5 --logs_dir logs/IRT/
cp "$(ls -t logs/SPT/*.pth | head -1)" model_data/SPT.pth
cp "$(ls -t logs/IRT/*.pth | head -1)" model_data/IRT.pth
```

---

## 4. Evaluation (mAP@0.5)

```bash
# foggy test set
MODEL_PATH=logs/ep100-loss....pth python get_map.py
# clean test set
MODEL_PATH=logs/ep100-loss....pth TEST_DIR=datasets/voc_clean/test MAP_OUT=logs/clean_map_out python get_map.py
# real-world fog (RTTS, VOC format with JPEGImages/, Annotations/, test.txt; download separately)
MODEL_PATH=logs/ep100-loss....pth TEST_DIR=datasets/RTTS python RTTS_get_map.py
```

- Per-class AP and mAP are printed.
- Detailed outputs go to `logs/fog_map_out/`, `logs/clean_map_out/` and `logs/RTTS_map_out/`.
- Without `MODEL_PATH`, `model_data/AWD.pth` is used.
- The same commands evaluate the baseline and DBT checkpoints.

---

## 5. Visualising DBT

```bash
python tools/visualize_dbt.py --student logs/ep100-....pth --dbt logs/dbt_ep100.pth \
    datasets/voc_fog/test/JPEGImages/2007_000027.jpg datasets/voc_fog/test/JPEGImages/2007_000346.jpg
```

For each image, it saves a figure to `logs/dbt_vis/` showing:
- the foggy input and its clean pair;
- the learned severity map W;
- α, the trust in IRT;
- the teacher disagreement D.

---

## 6. Output files (`logs/`)

| File | Content |
|---|---|
| `epNNN-loss…-val_loss….pth` | Student (detector) checkpoint; this is what you evaluate |
| `dbt_epNNN.pth` | DBT gate and optimizer state for resuming (DBT runs only) |
| `dbt_stats.csv` | Per epoch: `alpha_mean`, `alpha_std` (spatial variation), `W_mean`, `corr_alpha_W`, the DBT losses, `beta`, detection and validation loss |
| `loss_<date>/` | Loss curves and TensorBoard logs |
| `fog_map_out/`, `clean_map_out/` | Evaluation details |
| `dbt_vis/*.png` | DBT visualisations |

DBT's printed "Total Loss" includes the gate's own loss. Compare runs by detection/validation loss and mAP.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `UnidentifiedImageError` / broken JPEG | Re-run `python tools/prepare_data.py --voc VOCdevkit/VOC2012`; it rebuilds incomplete files |
| `matched 0/215` or `partially matches` warning | Wrong `SPT.pth` / `IRT.pth`; use the authors' teacher weights or ones trained with section 3.4 |
| CUDA out of memory | `BATCH_SIZE=4` |
| `ModuleNotFoundError: utils` / `nets` | Run from the repo root with `export PYTHONPATH=$(pwd)` |
| Resumed DBT run warns that the gate restarts | Supply `DBT_PATH=logs/dbt_epNNN.pth` from the same epoch |
| `yolo.py` `detect_image` can't find `simhei.ttf` | Copy any `.ttf` font to `model_data/simhei.ttf` (only needed for drawing boxes) |

---

## 8. Repository map

| Path | Purpose |
|---|---|
| `train/train_AWD.py` | Trains the student (DTFA or DBT) |
| `train/pretrain_SPT.py`, `train/pretrain_IRT.py` | Train the two teachers |
| `nets/AWD/` | Student detector (YOLOX-s with deformable conv, transformer block, SC blocks) and `ShallowNet` teacher backbone |
| `nets/detection/`, `nets/reconstruction/` | Teacher networks for pre-training (added; missing upstream) |
| `utils/AFB/loss.py` | DTFA's Adaptive Feature Bridging |
| `utils/DBT/loss.py` | Dynamic Bridging Tuning (ours) |
| `utils/fit.py` | One training epoch (baseline and DBT) |
| `get_map.py`, `RTTS_get_map.py`, `yolo.py` | Evaluation and inference |
| `tools/prepare_data.py`, `tools/visualize_dbt.py` | Dataset generation; DBT visualisation |
| `DTFA_colab.ipynb`, `DTFA_kaggle.ipynb` | Notebooks for Colab and Kaggle |
| `DBT.md`, `main.tex` / `report.pdf`, `README_SESSION.md` | Method notes, report, setup log |

## Citation (baseline)

```
@article{HU2026132726,
  title   = {A lightweight framework for robust object detection in adverse weather based on dual-teacher feature alignment},
  author  = {Rui Hu and Hanjun Zheng and Shengjie Ye and Linbo Qing and Honggang Chen},
  journal = {Neurocomputing}, volume = {671}, pages = {132726}, year = {2026},
  doi     = {10.1016/j.neucom.2026.132726}
}
```
