# Dynamic Bridging Tuning (DBT)

DBT is an extension of DTFA, implementing the plan in `Enhancement-In-Object-Detection.pdf`. DTFA aligns the student with the IRT teacher and the SPT teacher using a **fixed 1 : 1 weighting** everywhere in the image. DBT learns **where to trust which teacher**. A gate looks at both teachers, the student and an estimated weather-severity map, and outputs a per-location weight α(x, y) for IRT; SPT gets 1 − α.

DBT is used during training only. The detector you deploy and evaluate is the same YOLOX-s student, with no extra test-time cost.

## Method

At the shallow alignment level, all three networks are bridged into one shared 128 × 160 × 160 space (F̃_IRT, F̃_SPT, F_S):

| Quantity | Definition |
|---|---|
| Severity map | W = σ(f_w(I_adverse)) ∈ [0,1]^{H×W}: a small CNN on the foggy input |
| Disagreement | D = ‖F̃_IRT − F̃_SPT‖₂ per location, normalised by its image mean |
| Gate | [z_I, z_S] = Conv1×1(SiLU(BN(Conv3×3([F̃_IRT; F̃_SPT; F_S; W; D])))) |
| Reliabilities | r_I = softplus(z_I + β·W), r_S = softplus(z_S). β is learnable (initial value 1), so the prior is "more severe weather ⇒ more IRT" |
| Trust in IRT | α = r_I / (r_I + r_S + ε) (reliability-aware fusion) |
| Fused target | F_B = α ⊙ F̃_IRT + (1 − α) ⊙ F̃_SPT (mixture of teacher features) |

Training objective (the baseline is `0.2·L_det + L_rc + L_dc`):

```
L = 0.2·L_det + λ·L_align + η·L_gate + γ·L_reg + μ·L_sev
```

| Term | Trains | Definition | Default |
|---|---|---|---|
| L_align | student | Masked generative distillation toward F_B, the same operator as the baseline AFB, with α detached | λ = 2 (the baseline has two such terms with weight 1 each) |
| L_gate | gate | ‖sg(F_S) − F_B‖² / ‖F̃_IRT − F̃_SPT‖²: the mix of teachers that best fits the student (plan, Option 3) | η = 1 |
| L_reg | gate | 1 − H(α)/log 2: entropy regulariser against collapsing onto one teacher (plan, Option 1) | γ = 0.01 |
| L_sev | severity CNN | BCE(W, W*), with W* = clip(mean\|I_adverse − I_clean\| / 0.25, 0, 1) from the paired clean image | μ = 0.1 |

### Design decisions (where the plan left options open)

1. **Spatial gate, A ∈ [0,1]^{H×W},** as the plan recommends. The image-level version (Option A) and a fixed α = 0.5 are available for ablations.
2. **Learned severity (f_w), weakly supervised.** The training data is paired (foggy and clean), so |I_adverse − I_clean| gives a free per-pixel severity pseudo-label. It's only used in training, and `DBT_MU=0` turns it off (W is then learned only through the gate).
3. **The gate has its own loss and its own optimizer.**
   - DTFA's alignment operator passes the student's feature through a fixed, randomly initialised generator. The original `train_AWD.py` never puts the AFB modules in the optimizer, so they keep their random weights.
   - Trained only through that operator, the gate got no usable signal. In a synthetic test where the student copied IRT on the left half and SPT on the right, α stayed at about 0.50 / 0.50.
   - With the direct L_gate it learns the split: α = 0.84 on the IRT half and 0.16 on the SPT half.
   - It only learns with Adam (`DBT_LR`, default 1e-3). Under the student's SGD learning rate (~1e-3) α didn't move.
4. **The bridge projection and the generator keep their random initialisation,** exactly as in the baseline, so the comparison with DTFA isolates the effect of the gate.

## Files

| File | Change |
|---|---|
| `utils/DBT/loss.py` | New: `DBTLoss` (bridge, severity estimator, gate, losses, statistics) |
| `utils/fit.py` | Uses `DBTLoss` when given, steps the gate optimizer, writes `logs/dbt_stats.csv`, saves `logs/dbt_epNNN.pth` |
| `train/train_AWD.py` | `DBT=1` switches it on (default `0` = original DTFA, unchanged). Reads the settings below and restores the gate when resuming |
| `tools/visualize_dbt.py` | New: saves a figure per image with the input, clean image, W, α and D |
| `DTFA_kaggle.ipynb` | DBT parameters, gate checkpoint resume, visualisations after the final epoch |
| `DTFA_code_update.zip` | The changed code files, for adding to the Kaggle dataset next to `DTFA_colab.zip` |

## Running

**Kaggle:** add `DTFA_code_update.zip` to your `dtfa-data` dataset (*New Version*), set `DBT = 1` (or `0` for the baseline) in the notebook's parameter cell, and *Save & Run All*. Resuming works the same as before; the gate checkpoint is found automatically.

**Anywhere else** (Colab / local GPU), from the repo root:

```bash
export PYTHONPATH=$(pwd)
DBT=1 SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py                      # DBT
DBT=0 SAVE_PERIOD=5 NUM_WORKERS=2 python train/train_AWD.py                      # baseline
DBT=1 INIT_EPOCH=50 MODEL_PATH=logs/ep050-....pth python train/train_AWD.py       # resume (finds logs/dbt_ep050.pth)
```

Baseline and DBT both write to `logs/`, so run them in separate copies of the repo, or move `logs/` between runs.

| Variable | Default | Meaning |
|---|---|---|
| `DBT` | 0 | 1 = Dynamic Bridging Tuning |
| `DBT_MODE` | spatial | `spatial`, `image` (one α per image) or `fixed` (α = 0.5, no gate) |
| `DBT_USE_S`, `DBT_USE_W`, `DBT_USE_D` | 1 | Gate inputs: student feature, severity map, teacher disagreement |
| `DBT_LAMBDA`, `DBT_ETA`, `DBT_GAMMA`, `DBT_MU` | 2.0, 1.0, 0.01, 0.1 | Loss weights λ, η, γ, μ |
| `DBT_SEV_SCALE` | 0.25 | Pixel difference that counts as maximum severity |
| `DBT_LR` | 1e-3 | Adam learning rate of the gate and severity estimator |
| `DBT_PATH` | (auto) | Gate checkpoint to load |

The existing `EPOCHS`, `STOP_EPOCH`, `INIT_EPOCH`, `MODEL_PATH`, `TRAIN_SUBSET`, `VAL_SUBSET`, `BATCH_SIZE`, `SAVE_PERIOD` and `NUM_WORKERS` work as before.

## What to report

- **Main result:** mAP@0.5 on VOC-Fog test and VOC-Clean test (`get_map.py`) for DTFA (`DBT=0`) and DBT (`DBT=1`), trained with identical settings.
- **Suggested ablations** (one setting changed per row):

| Row | Setting |
|---|---|
| Fixed fusion | `DBT_MODE=fixed` (single fused target, no attention) |
| Image-level gate | `DBT_MODE=image` |
| Spatial gate, teachers only | `DBT_USE_S=0 DBT_USE_W=0 DBT_USE_D=0` |
| + student | `DBT_USE_W=0 DBT_USE_D=0` |
| + severity | `DBT_USE_D=0` |
| Full DBT | defaults |
| No severity supervision | `DBT_MU=0` |

- **Behaviour of the gate,** from `logs/dbt_stats.csv`, one row per epoch:
  - `alpha_mean`: average trust in IRT
  - `alpha_std`: how spatially varying the trust is (≈ 0 means no regional adaptation)
  - `corr_alpha_W`: whether trust in IRT rises with weather severity, the plan's hypothesis W↑ ⇒ α↑
  - `beta`: the learned severity prior
- **Qualitative:** the `tools/visualize_dbt.py` figures, showing α and W side by side.

Note that DBT's printed "Total Loss" also includes the gate's own loss. Compare runs by detection/validation loss and mAP, not total loss.

## Verification so far (CPU sandbox, tiny subsets; no full training yet)

- Module unit test: gradients reach the gate, the severity estimator and the student; bridge and generator stay fixed; all three modes and the input switches run.
- Synthetic gate test (above): with the final design, the gate learns the IRT-left / SPT-right split (α = 0.84 / 0.16).
- Training loop: the original DTFA (`DBT=0`) and each DBT configuration train, log and save. Resuming restores the gate and its Adam state, and the spread of α grows over the first steps (std 0.008 → 0.086).
- `tools/visualize_dbt.py` produces the figures, and the Kaggle notebook's unzip, overlay and resume logic was dry-run on a mock directory layout.

## Limitations and caveats

- No full training run has been done yet, so whether DBT improves mAP is still open.
- The gate learns which teacher mix the student is already closest to. It is kept from collapsing by the entropy term and nudged toward IRT in severe weather by the β·W prior, but in principle it can reinforce the student's current preference. Watch `alpha_mean` and `alpha_std` in `dbt_stats.csv`.
- The severity pseudo-label assumes paired clean images (true for the synthetic VOC-Fog data). `DBT_SEV_SCALE` is a heuristic.
