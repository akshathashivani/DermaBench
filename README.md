# DermaBench: An Evaluation-First Study of CNN and Vision-Transformer Skin-Lesion Classifiers on DermaMNIST

### Calibration, image-quality proxy-subgroup and corruption-robustness analysis against pre-declared acceptance criteria

![Python](https://img.shields.io/badge/python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-timm-red) ![Status](https://img.shields.io/badge/status-research%20benchmark-lightgrey) ![Not a medical device](https://img.shields.io/badge/not%20a%20medical%20device-%E2%9A%A0-orange)

> **Research and educational project. Not a medical device. Not validated for clinical use. Not for diagnosis.**

---

## TL;DR

Most image-classification repos stop at "accuracy = X". This project asks the question a regulated medical-AI team would ask: **given what the algorithm is *intended* to do, does the evidence show it is good enough, and where does it break?**

I fine-tuned a **ResNet-50** and a **ViT-S/16** on DermaMNIST (7-class dermoscopy, 224 px), **declared acceptance criteria before looking at test results**, then evaluated discrimination, calibration, image-quality proxy subgroups, robustness to corruptions and failure modes.

| Question | Finding (primary model: ViT-S/16, test set, n = 2,005) |
|---|---|
| Does it discriminate well? | **Yes.** Macro AUROC **0.988** (95% CI 0.985-0.990); balanced accuracy **0.854**; melanoma AUROC **0.961**. ViT-S clearly outperformed ResNet-50 (macro AUROC 0.949). |
| Is the melanoma miss-rate acceptable? | Melanoma sensitivity **0.744** (CI 0.686-0.804) at specificity **0.967**. This meets my pre-declared 0.60 floor, but **~1 in 4 melanomas is still missed**, mostly called benign nevus (45 of 57 misses). |
| Are the probabilities trustworthy? | **Yes, after temperature scaling.** ECE fell from 0.037 to **0.016** (T = 1.44). |
| Is performance uniform across image-quality subgroups? | **Marginal fail.** Largest balanced-accuracy gap **0.102** vs. a 0.10 limit (high-contrast images). Small subgroups, no demographic data, so this is inconclusive on its own. |
| Is it robust to realistic image degradation? | **No. This is the main finding.** Moderate contrast reduction cut melanoma sensitivity from 0.744 to **0.045**; blur, JPEG compression and colour-channel loss also caused large drops. |
| **Overall verdict** | **3 of 5 pre-declared criteria met; the 2 robustness/fairness-style criteria failed.** Strong lab-condition discrimination does not imply robustness. |

---

## Why this project exists

Siemens-style product teams, regulators and clinical partners do not ask "what is your accuracy?" They ask for **objective, reproducible evidence of stand-alone algorithm performance**: defined intended use, measurable targets, representative evaluation data, subgroup analysis, robustness, calibration, limitations, and traceability.

DermaBench is a small, fully reproducible exercise in producing that kind of evidence package on public data:

1. **Pre-declared acceptance criteria**, timestamped in `criteria_frozen.json` before test evaluation.
2. **Uncertainty everywhere:** 500-resample bootstrap 95% confidence intervals on test metrics.
3. **Calibration, not just ranking:** reliability diagrams, ECE, NLL, Brier, temperature scaling fitted on validation only.
4. **Honest subgroup analysis** within the limits of the dataset (no demographics, so image-quality proxy subgroups, labelled as such).
5. **Stress testing:** five corruption types x three severities plus a colour-channel-removal test.
6. **Failure-mode analysis** with confusion structure, confident-error inspection and EigenCAM / Grad-CAM.
7. **Provenance and reproducibility:** data fingerprints (SHA-256), seeds, environment capture, MLflow experiment tracking, auto-generated model card.

---

## Intended use (framing only)

A *hypothetical* decision-support **triage** component that prioritises dermoscopy images for clinician review. The evaluation is framed around that use, where a missed melanoma is the costly error. **No clinical claims are made.**

---

## Data

**DermaMNIST** (MedMNIST+; derived from HAM10000), 224 x 224 RGB dermoscopy images, 7 classes, fixed official train/val/test splits.

| Class | Description | Train | Val | Test |
|---|---|---:|---:|---:|
| akiec | Actinic keratoses / intraepithelial carcinoma | 228 | 33 | 66 |
| bcc | Basal cell carcinoma | 359 | 52 | 103 |
| bkl | Benign keratosis-like lesions | 769 | 110 | 220 |
| df | Dermatofibroma | 80 | 12 | 23 |
| **mel** | **Melanoma** | 779 | 111 | 223 |
| nv | Melanocytic nevi | 4,693 | 671 | 1,341 |
| vasc | Vascular lesions | 99 | 14 | 29 |
| **Total** | | **7,007** | **1,003** | **2,005** |

Class imbalance is **58.7x** (nv vs. df in train), which is why per-class metrics and balanced accuracy are reported alongside overall accuracy.

**Data fingerprints** (first 16 hex chars of SHA-256, so anyone can verify they are evaluating on identical data):

| Split | Images | Labels |
|---|---|---|
| train | `2391cc9d9413883c` | `49eaf45cb52872a8` |
| val | `2932a4571ac92866` | `1b173d696f44698c` |
| test | `967eb046ee08c218` | `b42481cd36b554c0` |

> **Important data limitation:** MedMNIST does not expose patient/lesion IDs or demographics (age, sex, lesion site, skin tone). I therefore **cannot verify that the splits are lesion-disjoint** (HAM10000 contains multiple images of some lesions) and **cannot perform demographic subgroup analysis**. Reported test performance may be optimistic.

---

## Methods

### Models and training
| | |
|---|---|
| Models | `resnet50` (CNN) and `vit_small_patch16_224` (Vision Transformer), ImageNet-pretrained via [`timm`](https://github.com/huggingface/pytorch-image-models) |
| Input | 224 x 224, ImageNet normalisation |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4, cosine schedule, 10 epochs, batch 64, mixed precision |
| Imbalance handling | Class-weighted cross-entropy (square-root inverse frequency) |
| Augmentation | Random flips, 90-degree rotations, mild brightness jitter (x0.9-1.1) |
| Model selection | Best epoch by **validation** macro-AUROC; primary model chosen on validation (**ViT-S**: 0.984 vs. ResNet-50: 0.943). The test set was never used for selection |
| Seed | 42 (single run) |
| Hardware | Kaggle, Tesla T4 |
| Tracking | MLflow (SQLite backend) |

### Evaluation protocol
- **Metrics:** accuracy, balanced accuracy, macro one-vs-rest AUROC, melanoma AUROC / sensitivity / specificity, per-class sensitivity, specificity, precision, F1, AUROC.
- **Uncertainty:** non-parametric bootstrap, 500 resamples, 95% percentile CIs.
- **Calibration:** ECE (15 bins), NLL, Brier; single-parameter **temperature scaling fitted on the validation set only**.
- **Proxy subgroups:** brightness, contrast, sharpness (variance of Laplacian) and colourfulness, each split into low/mid/high **using training-set tertiles**. These are image-quality bands, **not patient demographics**.
- **Robustness:** Gaussian noise, Gaussian blur, brightness reduction, contrast reduction, JPEG compression, each at mild/moderate/severe, plus removal of a single RGB channel (a crude stand-in for a "missing input modality").
- **Failure analysis:** confusion matrix, error rate by proxy subgroup, most-confident errors, EigenCAM and Grad-CAM overlays.

### Pre-declared acceptance criteria
Frozen before test evaluation (`results/tables/criteria_frozen.json`, timestamp 2026-11-07T08:46:10 UTC). These thresholds are illustrative defaults I chose as a plausible bar for a triage-support research prototype, **not** clinically derived targets.

| # | Criterion | Threshold |
|---|---|---|
| 1 | Macro AUROC | >= 0.90 |
| 2 | Melanoma sensitivity | >= 0.60 |
| 3 | ECE after temperature scaling | <= 0.05 |
| 4 | Max balanced-accuracy gap between overall and any proxy subgroup (n >= 50) | <= 0.10 |
| 5 | Max drop in melanoma sensitivity under *moderate* corruptions | <= 0.15 |

---

## Results

### 1. Discrimination: model comparison (test set, temperature-scaled probabilities, 95% bootstrap CI)

| Metric | ResNet-50 | **ViT-S/16 (primary)** |
|---|---|---|
| Macro AUROC | 0.949 [0.939-0.957] | **0.988 [0.985-0.990]** |
| Balanced accuracy | 0.635 [0.596-0.673] | **0.854 [0.821-0.885]** |
| Accuracy | 0.801 [0.785-0.819] | **0.904 [0.892-0.916]** |
| Melanoma AUROC | 0.895 [0.873-0.913] | **0.961 [0.950-0.972]** |
| Melanoma sensitivity | 0.574 [0.507-0.636] | **0.744 [0.686-0.804]** |
| Melanoma specificity | 0.932 [0.919-0.943] | **0.967 [0.959-0.975]** |
| ECE | 0.022 [0.019-0.042] | **0.016 [0.014-0.030]** |

The CIs do not overlap on any discrimination metric, so the ViT advantage is not a sampling artefact of this test set. It is, however, a **single-seed** comparison.

![Training curves](results/figures/02_training_curves.png)

### 2. Per-class performance (ViT-S/16)

| Class | n | Sensitivity | Specificity | Precision | F1 | AUROC (OvR) |
|---|---:|---:|---:|---:|---:|---:|
| akiec | 66 | 0.742 | 0.998 | 0.925 | 0.824 | 0.994 |
| bcc | 103 | 0.903 | 0.992 | 0.853 | 0.877 | 0.995 |
| bkl | 220 | 0.836 | 0.979 | 0.829 | 0.833 | 0.985 |
| df | 23 | 0.870 | 1.000 | 1.000 | 0.930 | 0.999 |
| **mel** | 223 | **0.744** | 0.967 | 0.741 | 0.743 | 0.961 |
| nv | 1,341 | 0.950 | 0.889 | 0.945 | 0.948 | 0.980 |
| vasc | 29 | 0.931 | 0.999 | 0.931 | 0.931 | 1.000 |

Melanoma and actinic keratoses are the weakest classes by sensitivity. Rare classes (df n = 23, vasc n = 29) have wide uncertainty, so their near-perfect scores should not be over-interpreted.

### 3. Calibration

| Model | Temperature T | ECE raw -> TS | NLL raw -> TS | Brier raw -> TS |
|---|---:|---|---|---|
| ResNet-50 | 0.784 | 0.054 -> 0.022 | 0.574 -> 0.557 | 0.287 -> 0.283 |
| **ViT-S/16** | **1.445** | **0.037 -> 0.016** | **0.300 -> 0.280** | **0.149 -> 0.145** |

ViT-S was **over-confident** before scaling (T > 1); ResNet-50 was **under-confident** (T < 1). One scalar fitted on validation data brought both models below the 0.05 ECE target.

![Calibration and per-class](results/figures/03_calibration_and_per_class.png)

### 4. Acceptance-criteria verdict (ViT-S/16)

| Criterion | Threshold | Measured | Verdict |
|---|---|---:|---|
| Macro AUROC | >= 0.90 | 0.9876 | PASS |
| Melanoma sensitivity | >= 0.60 | 0.7444 | PASS |
| ECE after temperature scaling | <= 0.05 | 0.0158 | PASS |
| Max proxy-subgroup balanced-accuracy gap | <= 0.10 | 0.1021 | **FAIL (marginal)** |
| Max melanoma-sensitivity drop, moderate corruptions | <= 0.15 | 0.6996 | **FAIL** |

**Overall: not all criteria met.** The model is accurate and well calibrated on clean test data but does not meet the robustness criterion, and narrowly misses the subgroup criterion.

### 5. Image-quality proxy-subgroup analysis (ViT-S/16)

Overall balanced accuracy: 0.854. Gap = overall - subgroup. Bands use training-set tertiles.

| Feature | Band | n | mel n | Bal. acc | Mel. sens. | ECE | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| brightness | low | 659 | 93 | 0.870 | 0.871 | 0.026 | -0.016 |
| brightness | mid | 672 | 87 | 0.852 | 0.690 | 0.029 | 0.002 |
| brightness | high | 674 | 43 | 0.823 | 0.581 | 0.030 | 0.031 |
| contrast | low | 675 | 38 | 0.859 | 0.605 | 0.031 | -0.005 |
| contrast | mid | 646 | 63 | 0.872 | 0.730 | 0.019 | -0.018 |
| contrast | **high** | 684 | 122 | **0.752** | 0.795 | 0.021 | **0.102** |
| sharpness | low | 654 | 29 | 0.819 | **0.448** | 0.031 | 0.035 |
| sharpness | mid | 672 | 87 | 0.865 | 0.770 | 0.019 | -0.011 |
| sharpness | high | 679 | 107 | 0.848 | 0.804 | 0.025 | 0.006 |
| colorfulness | low | 699 | 91 | 0.803 | 0.681 | 0.022 | 0.051 |
| colorfulness | mid | 626 | 92 | 0.872 | 0.772 | 0.029 | -0.018 |
| colorfulness | high | 680 | 40 | 0.924 | 0.825 | 0.009 | -0.070 |

**Reading this carefully:**
- The single criterion violation (high-contrast, gap 0.102) is **0.002 above the limit** and has **no confidence interval**. Subgroup balanced accuracy depends on rare-class recall in a few hundred images, so it is noisy. Plain accuracy in that subgroup (0.899) is essentially unchanged from overall (0.904). I report it as a FAIL because the criterion was declared in advance, but I would not call it evidence of a real disparity without more data.
- **Exploratory observation (not pre-declared, small n, no CIs):** melanoma sensitivity appears lower in high-brightness (0.581, n_mel = 43) and low-sharpness (0.448, n_mel = 29) images. This is worth a follow-up with more data, not a conclusion.
- Calibration (ECE) stays low (0.009-0.031) in every band.

![Proxy subgroups](results/figures/04_proxy_subgroups.png)

### 6. Robustness to corruption (ViT-S/16, temperature fixed from clean validation data)

Clean baseline: balanced accuracy 0.854, melanoma sensitivity 0.744, macro AUROC 0.988, ECE 0.016.

| Corruption | Severity (parameter) | Bal. acc | Mel. sens. | Macro AUROC | ECE | Drop in mel. sens. |
|---|---|---:|---:|---:|---:|---:|
| Gaussian noise | mild (sigma 0.02) | 0.807 | 0.803 | 0.984 | 0.011 | -0.058 |
| | moderate (0.05) | 0.625 | 0.677 | 0.968 | 0.048 | 0.067 |
| | severe (0.10) | 0.456 | 0.471 | 0.935 | 0.075 | 0.274 |
| Gaussian blur | mild (1 px) | 0.781 | 0.592 | 0.981 | 0.023 | 0.152 |
| | moderate (2 px) | 0.676 | 0.386 | 0.963 | 0.064 | 0.359 |
| | severe (4 px) | 0.490 | 0.233 | 0.919 | 0.125 | 0.511 |
| Brightness reduction | mild (x0.8) | 0.802 | 0.709 | 0.986 | 0.017 | 0.036 |
| | moderate (x0.6) | 0.700 | 0.547 | 0.978 | 0.043 | 0.197 |
| | severe (x0.4) | 0.461 | 0.341 | 0.947 | 0.129 | 0.404 |
| Contrast reduction | mild (x0.7) | 0.751 | 0.247 | 0.973 | 0.050 | 0.498 |
| | **moderate (x0.5)** | 0.671 | **0.045** | 0.943 | 0.096 | **0.700** |
| | severe (x0.3) | 0.579 | 0.013 | 0.903 | 0.067 | 0.731 |
| JPEG compression | mild (q 50) | 0.801 | 0.691 | 0.977 | 0.020 | 0.054 |
| | moderate (q 20) | 0.693 | 0.605 | 0.956 | 0.022 | 0.139 |
| | severe (q 5) | 0.325 | 0.045 | 0.801 | 0.083 | 0.700 |
| Remove R channel | exploratory | 0.289 | 0.067 | 0.854 | 0.213 | 0.677 |
| Remove G channel | exploratory | 0.155 | 0.018 | 0.646 | 0.697 | 0.726 |
| Remove B channel | exploratory | 0.228 | 0.072 | 0.837 | 0.150 | 0.673 |

For comparison, ResNet-50's largest melanoma-sensitivity drop at moderate severity was 0.574, so it also failed this criterion.

**What stands out:**
1. **Contrast reduction is the most damaging corruption.** At a moderate level (x0.5) melanoma sensitivity falls from 0.744 to 0.045, and the model essentially stops predicting melanoma, while balanced accuracy only falls to 0.671.
2. **Blur is the second largest threat** (sensitivity 0.386 at 2 px). Real-world analogues are focus errors and low-quality or smartphone-captured images.
3. **Ranking survives better than decisions.** Macro AUROC degrades far less than balanced accuracy or melanoma sensitivity (e.g. 0.943 vs. 0.671 under moderate contrast loss). This suggests the model still carries useful ranking information while its decision boundary and calibration shift. That is a *hypothesis* pointing toward corruption-aware training, recalibration or an input-quality gate as mitigations. I have not tested these.
4. **Sensitivity alone can mislead.** Mild noise *raised* melanoma sensitivity (0.744 -> 0.803) while balanced accuracy fell (0.854 -> 0.807), which suggests predictions shifted rather than improved. Always read sensitivity together with other metrics.
5. **Strong colour dependence.** Removing a colour channel is catastrophic (balanced accuracy 0.16-0.29), and removing green produces *confidently wrong* predictions (ECE 0.697). This is a crude proxy for the missing-modality problem, not a real multi-modal test.

![Robustness](results/figures/05_robustness.png)

### 7. Failure-mode analysis (ViT-S/16)

Of 2,005 test images, **1,813 (90.4%) were classified correctly**. The 192 errors break down as:

| Error type | Count |
|---|---:|
| Other confusion | 77 |
| False melanoma alarm (predicted mel, truly not) | 58 |
| **Missed melanoma (truly mel, predicted other)** | **57** |

**Where the 57 missed melanomas went:** nevus 45 (79%), benign keratosis 8, vascular 2, BCC 1, actinic keratosis 1. The dominant failure is **melanoma-vs-nevus confusion**, which is clinically unsurprising and is also the costliest direction for a triage tool.

![Confusion matrix and error rates](results/figures/06_confusion_and_error_rates.png)

**Most-confident errors (6 inspected, non-expert visual review, not clinical judgement):**
- **EigenCAM** concentrated on the lesion itself in all six examples, so these confident errors do not look like the model attending to background artefacts.
- **Grad-CAM** maps for the ViT were diffuse, often highlighting peri-lesional skin, so I treat them as low-reliability. Saliency maps are qualitative and do not establish which features the model actually uses.
- The six examples include confident melanoma-as-nevus errors (including a reddish, non-pigmented-looking lesion) and confident nevus-as-melanoma alarms. Six examples are far too few to generalise from. `results/tables/failure_review_sheet.csv` contains the top-50 confident errors for systematic manual annotation (**TODO: complete this review and summarise the categories here**).

![Confident errors with CAMs](results/figures/07_cam_confident_errors.png)

---

## Interpretation

- **Architecture matters here.** The pretrained ViT-S/16 beat ResNet-50 on every discrimination metric with non-overlapping CIs under an identical training recipe (single seed).
- **Calibration is cheap to fix.** A single temperature parameter, fitted on validation data only, brought ECE from 0.037 to 0.016.
- **Good test accuracy is not deployment evidence.** The same model that reaches macro AUROC 0.988 on clean images loses most of its melanoma sensitivity under moderate contrast loss. If this were a real product, **robustness requirements would be the gating problem**, not headline accuracy.
- **Fail honestly.** Two of five criteria failed. I kept the verdicts as measured and did not adjust thresholds after seeing results.

---

## Limitations

- **No demographics or patient/lesion identifiers.** Fairness across age, sex, skin tone or anatomical site could not be assessed. Proxy subgroups are image-quality bands only and must not be described as demographic fairness.
- **Split integrity unverified.** HAM10000 contains multiple images of some lesions; if splits are not lesion-disjoint, test performance may be optimistic. Do not compare these numbers directly with published MedMNIST leaderboards (different protocols, seeds and training recipes).
- **Downsampled, single-source dataset.** Images are resized from the originals, are dermoscopy-only, and come from one collection. There is no external validation, reader study or prospective evaluation.
- **Single training seed;** seed variance was not quantified, and no hyperparameter search was performed. ViT-S training loss reached 0.036, so some overfitting to the training set is likely.
- **Acceptance thresholds are illustrative,** not clinically derived.
- **Corruptions are synthetic.** The missing-colour-channel test is only a crude analogue of missing input modalities. True missing-modality robustness requires multi-sequence data (e.g. brain MRI), which is outside this study.
- **Subgroup metrics have no confidence intervals** and some subgroups have small melanoma counts (n_mel as low as 29).
- **Saliency maps** (EigenCAM / Grad-CAM) are qualitative and were reviewed on only six examples.

---

## Reproducibility

### Run it
1. Open `notebooks/DermaBench_Kaggle.ipynb` on Kaggle (**Accelerator: GPU, Internet: On**), or use the Colab version.
2. *Run All.* A full run at 224 px takes roughly 15-25 minutes on a T4 (about 30 s/epoch for ResNet-50, 25 s/epoch for ViT-S).
3. Outputs are written to `dermabench_outputs/`: `figures/`, `tables/`, `MODEL_CARD.md`, and an MLflow database.

For a quick smoke run, set `img_size=64` and `epochs=3` in the configuration cell.

### What is captured for traceability
| Artefact | File |
|---|---|
| Run configuration and package versions | `results/tables/run_config_environment.json` |
| Frozen acceptance criteria with timestamp | `results/tables/criteria_frozen.json` |
| Data fingerprints (SHA-256) | `results/tables/data_fingerprint.json` |
| Proxy-subgroup thresholds (from train) | `results/tables/proxy_subgroup_thresholds_from_train.json` |
| Per-epoch training history | `results/tables/train_history_*.json` |
| Test metrics with CIs | `results/tables/test_summary_*.csv` |
| Robustness results | `results/tables/robustness.csv` |
| Acceptance verdict | `results/tables/acceptance_verdict.csv` |
| Model card | `results/MODEL_CARD.md` |

Helper functions (ECE, bootstrap CIs, temperature scaling, perturbations, proxy features) are unit-tested inside the notebook (cell 5b) before the evaluation runs.

---

## Repository structure

```
dermabench/
├── README.md
├── notebooks/
│   ├── DermaBench_Kaggle.ipynb      # executed notebook with outputs
│   └── DermaBench_Colab.ipynb
├── results/
│   ├── MODEL_CARD.md
│   ├── figures/                     # 01-07 PNGs referenced above
│   └── tables/                      # CSV / JSON evidence files
└── LICENSE
```

---

## Roadmap

- [ ] Refactor the helper library into an installable `src/dermabench/` package with `pytest` tests, DVC for data/checkpoint versioning and a GitHub Actions CI workflow
- [ ] Complete the manual failure-mode review of the top-50 confident errors
- [ ] Multi-seed runs to quantify training variance
- [ ] Corruption-aware augmentation and recalibration; re-evaluate against the **same frozen criteria** to test the mitigation hypotheses
- [ ] Re-run on raw HAM10000 with lesion-grouped splits and metadata, to enable real age/sex/site subgroup analysis
- [ ] Operating-point analysis for melanoma (sensitivity-specificity trade-off) and abstention for low-quality inputs

---

## References

- Yang J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023.
- Doerrich S. et al. *Rethinking model prototyping through the MedMNIST+ dataset collection.* arXiv:2404.15786, 2024.
- Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 2018.
- Guo C. et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
- Dosovitskiy A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
- He K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
- `timm` (PyTorch Image Models); `pytorch-grad-cam`; `medmnist`; `mlflow`.

## License and data terms

Code: add your preferred licence (e.g. MIT) in `LICENSE`. DermaMNIST is derived from HAM10000, which is released under a **non-commercial** Creative Commons licence; check the [MedMNIST](https://medmnist.com/) and HAM10000 terms before any reuse, and do not redistribute the data in this repository.

## Author

**C. Akshatha Shivani** ([LinkedIn](https://www.linkedin.com/in/c-akshatha-shivani)): healthcare AI, clinical decision support, medical imaging.
