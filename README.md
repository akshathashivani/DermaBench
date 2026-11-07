# DermaBench

### An Evaluation-First Study of CNN and Vision-Transformer Skin-Lesion Classifiers on DermaMNIST

**Calibration, image-quality proxy-subgroup analysis and corruption robustness against pre-declared acceptance criteria.**

> **Research and educational project. Not a medical device. Not validated for clinical use. Not for diagnosis.**

---

## 1. Overview

Most image-classification projects focus primarily on predictive accuracy. However, medical AI systems require a broader evaluation of whether their predictions are reliable, calibrated, robust and suitable for their intended application.

**DermaBench** is an evaluation-first study of deep-learning models for seven-class skin-lesion classification using the DermaMNIST dataset.

The project compares a convolutional neural network (ResNet-50) with a Vision Transformer (ViT-S/16), evaluating not only classification performance but also calibration, uncertainty, image-quality proxy subgroups, robustness to image corruption and model failure modes.

Five acceptance criteria were declared before test-set evaluation to provide a structured and transparent assessment of model performance.

### Key findings

| Evaluation area | ViT-S/16 test result |
|---|---:|
| Accuracy | 0.904 |
| Balanced accuracy | 0.854 |
| Macro AUROC | 0.988 |
| Melanoma AUROC | 0.961 |
| Melanoma sensitivity | 0.744 |
| Melanoma specificity | 0.967 |
| ECE after temperature scaling | 0.016 |
| Acceptance criteria met | 3 of 5 |

**Central finding:** Strong classification performance on clean images does not necessarily translate into robustness under image degradation. Moderate contrast reduction decreased melanoma sensitivity from 0.744 to 0.045.

---

## 2. Research Objectives

The study investigates five questions:

1. How do CNN and Vision Transformer architectures compare in multi-class skin-lesion classification?
2. How reliable and calibrated are their predicted probabilities?
3. Does performance vary across image-quality proxy subgroups?
4. How sensitive are predictions to image degradation and colour-channel removal?
5. What failure modes emerge from confusion analysis and visual explainability?

The evaluation is framed around a hypothetical decision-support triage component that prioritises dermoscopy images for clinician review. This is research framing only and does not establish clinical suitability.

---

## 3. Dataset

**DermaMNIST (MedMNIST+)**, derived from HAM10000, is used for seven-class dermoscopic image classification.

| Property | Description |
|---|---|
| Dataset | DermaMNIST |
| Image dimensions | 224 × 224 |
| Image format | RGB |
| Classification task | Seven classes |
| Training images | 7,007 |
| Validation images | 1,003 |
| Test images | 2,005 |
| Total images | 10,015 |

### Class distribution

| Class | Description | Train | Validation | Test |
|---|---|---:|---:|---:|
| akiec | Actinic keratoses / intraepithelial carcinoma | 228 | 33 | 66 |
| bcc | Basal cell carcinoma | 359 | 52 | 103 |
| bkl | Benign keratosis-like lesions | 769 | 110 | 220 |
| df | Dermatofibroma | 80 | 12 | 23 |
| mel | Melanoma | 779 | 111 | 223 |
| nv | Melanocytic nevi | 4,693 | 671 | 1,341 |
| vasc | Vascular lesions | 99 | 14 | 29 |
| **Total** | | **7,007** | **1,003** | **2,005** |

The training set has a 58.7:1 class imbalance between melanocytic nevi and dermatofibroma. Class-weighted loss and per-class evaluation metrics are therefore used.

### Dataset provenance

SHA-256 fingerprints are recorded for the data splits to support dataset identity verification:

| Split | Image fingerprint | Label fingerprint |
|---|---|---|
| Train | `2391cc9d9413883c` | `49eaf45cb52872a8` |
| Validation | `2932a4571ac92866` | `1b173d696f44698c` |
| Test | `967eb046ee08c218` | `b42481cd36b554c0` |

**Data limitation:** Patient and lesion identifiers and demographic metadata are unavailable. Consequently, lesion-level split independence cannot be verified, and demographic subgroup analysis cannot be performed. Test performance may therefore be optimistic.

---

## 4. Model Architectures and Training

Two ImageNet-pretrained architectures were evaluated using the same general training protocol.

| Component | Configuration |
|---|---|
| CNN | ResNet-50 |
| Transformer | ViT-S/16 (`vit_small_patch16_224`) |
| Initialisation | ImageNet pretrained |
| Framework | PyTorch / timm |
| Input resolution | 224 × 224 |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | Cosine learning-rate schedule |
| Epochs | 10 |
| Batch size | 64 |
| Precision | Mixed precision |
| Imbalance handling | Square-root inverse-frequency class-weighted cross-entropy |
| Augmentation | Random flips, 90-degree rotations and mild brightness jitter |
| Random seed | 42 |
| Hardware | Kaggle Tesla T4 GPU |
| Experiment tracking | MLflow with SQLite backend |

Model selection was based on validation macro-AUROC. ViT-S/16 achieved 0.984 compared with 0.943 for ResNet-50 on the validation set and was selected as the primary model.

The test set was held out for final evaluation and was not used for model selection.

---

## 5. Evaluation Framework

DermaBench uses a multi-dimensional evaluation strategy rather than relying on accuracy alone.

### 5.1 Classification performance

- Accuracy and balanced accuracy
- Macro one-vs-rest AUROC
- Per-class sensitivity and specificity
- Precision and F1-score
- Melanoma-specific AUROC, sensitivity and specificity
- Confusion matrix and error analysis

### 5.2 Statistical uncertainty

Non-parametric bootstrap estimation was performed using 500 resamples to calculate 95% percentile confidence intervals for test metrics.

### 5.3 Probability calibration

Calibration was assessed using:

- Expected Calibration Error (ECE), with 15 bins
- Negative Log-Likelihood (NLL)
- Brier score
- Reliability diagrams
- Validation-fitted temperature scaling

Temperature scaling was fitted on validation data only and applied to test predictions.

### 5.4 Image-quality proxy-subgroup analysis

Four image characteristics were evaluated:

- Brightness
- Contrast
- Sharpness, measured using variance of the Laplacian
- Colourfulness

Each characteristic was divided into low, medium and high bands using training-set tertiles.

These are image-quality proxy subgroups, not demographic or patient subgroups.

### 5.5 Corruption robustness

The primary model was evaluated under synthetic image degradation:

- Gaussian noise
- Gaussian blur
- Brightness reduction
- Contrast reduction
- JPEG compression
- Individual RGB colour-channel removal

The first five corruption types were tested at mild, moderate and severe levels. Temperature scaling remained fixed using the clean validation calibration.

### 5.6 Explainability and failure analysis

The evaluation included:

- Confusion-matrix analysis
- Most-confident prediction errors
- EigenCAM visualisations
- Grad-CAM overlays
- Inspection of melanoma false negatives and false positives

Explainability outputs are qualitative and are not treated as proof of clinical reasoning.

---

## 6. Pre-Declared Acceptance Criteria

Five illustrative acceptance criteria were frozen before test evaluation.

The thresholds represent research-prototype evaluation targets selected for this study. They are not clinically derived or regulatory-approved thresholds.

| Criterion | Acceptance threshold |
|---|---:|
| Macro AUROC | ≥ 0.90 |
| Melanoma sensitivity | ≥ 0.60 |
| ECE after temperature scaling | ≤ 0.05 |
| Maximum proxy-subgroup balanced-accuracy gap | ≤ 0.10 |
| Maximum melanoma-sensitivity drop under moderate corruption | ≤ 0.15 |

The criteria were timestamped in `criteria_frozen.json` before the test evaluation.

---

## 7. Results

### 7.1 Model comparison

Test-set performance with 95% bootstrap confidence intervals:

| Metric | ResNet-50 | ViT-S/16 |
|---|---:|---:|
| Accuracy | 0.801 | 0.904 |
| Balanced accuracy | 0.635 | 0.854 |
| Macro AUROC | 0.949 | 0.988 |
| Melanoma AUROC | 0.895 | 0.961 |
| Melanoma sensitivity | 0.574 | 0.744 |
| Melanoma specificity | 0.932 | 0.967 |
| ECE | 0.022 | 0.016 |

ViT-S/16 outperformed ResNet-50 across the reported discrimination metrics. This comparison is based on a single training seed.

### 7.2 Per-class performance

| Class | Sensitivity | Specificity | Precision | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| akiec | 0.742 | 0.998 | 0.925 | 0.824 | 0.994 |
| bcc | 0.903 | 0.992 | 0.853 | 0.877 | 0.995 |
| bkl | 0.836 | 0.979 | 0.829 | 0.833 | 0.985 |
| df | 0.870 | 1.000 | 1.000 | 0.930 | 0.999 |
| mel | 0.744 | 0.967 | 0.741 | 0.743 | 0.961 |
| nv | 0.950 | 0.889 | 0.945 | 0.948 | 0.980 |
| vasc | 0.931 | 0.999 | 0.931 | 0.931 | 1.000 |

Melanoma and actinic keratoses were among the weaker classes by sensitivity. Rare-class results should be interpreted cautiously because of their small test sample sizes.

### 7.3 Calibration

| Model | Temperature | ECE before | ECE after |
|---|---:|---:|---:|
| ResNet-50 | 0.784 | 0.054 | 0.022 |
| ViT-S/16 | 1.445 | 0.037 | 0.016 |

Temperature scaling improved calibration for both models. ViT-S/16 achieved an ECE of 0.016 after scaling, below the pre-declared 0.05 threshold.

### 7.4 Acceptance-criteria verdict

| Criterion | Measured result | Verdict |
|---|---:|---|
| Macro AUROC | 0.9876 | PASS |
| Melanoma sensitivity | 0.7444 | PASS |
| ECE after temperature scaling | 0.0158 | PASS |
| Maximum proxy-subgroup balanced-accuracy gap | 0.1021 | FAIL |
| Maximum melanoma-sensitivity drop under moderate corruption | 0.6996 | FAIL |

**Overall: 3 of 5 criteria met.**

The model performed strongly on clean test images but narrowly failed the proxy-subgroup criterion and substantially failed the robustness criterion.

### 7.5 Image-quality proxy-subgroup findings

The largest balanced-accuracy gap was observed in the high-contrast subgroup:

- Overall balanced accuracy: 0.854
- High-contrast subgroup balanced accuracy: 0.752
- Gap: 0.102

This exceeded the 0.10 threshold by 0.002. The result is marginal, lacks a confidence interval and should not be interpreted as evidence of a real demographic disparity.

Calibration remained relatively stable across the image-quality bands, with ECE ranging from 0.009 to 0.031.

### 7.6 Corruption robustness

The clean baseline for ViT-S/16 was:

- Balanced accuracy: 0.854
- Melanoma sensitivity: 0.744
- Macro AUROC: 0.988
- ECE: 0.016

The most significant degradation occurred under contrast reduction.

| Corruption | Severity | Balanced accuracy | Melanoma sensitivity |
|---|---|---:|---:|
| Contrast reduction | Mild | 0.751 | 0.247 |
| Contrast reduction | Moderate | 0.671 | 0.045 |
| Contrast reduction | Severe | 0.579 | 0.013 |
| Gaussian blur | Moderate | 0.676 | 0.386 |
| Brightness reduction | Moderate | 0.700 | 0.547 |
| JPEG compression | Moderate | 0.693 | 0.605 |
| Gaussian noise | Moderate | 0.625 | 0.677 |

The moderate contrast reduction experiment decreased melanoma sensitivity by approximately 0.700 relative to the clean baseline.

Removing individual colour channels also caused severe degradation, with green-channel removal producing an ECE of 0.697.

These experiments highlight the importance of assessing robustness beyond clean-data classification performance.

**Important:** RGB channel removal is a crude colour-dependence experiment, not a genuine missing-modality evaluation.

### 7.7 Failure-mode analysis

Of the 2,005 test images:

- Correct classifications: 1,813
- Total errors: 192
- Other confusions: 77
- False melanoma alarms: 58
- Missed melanomas: 57

Among the 57 missed melanomas, 45 were classified as melanocytic nevi.

This dominant melanoma-versus-nevus confusion is an important failure mode in the hypothetical triage setting.

EigenCAM concentrated on the lesion in the six inspected examples. Grad-CAM outputs were more diffuse. These visual observations are qualitative and based on a limited number of examples.

---

## 8. Reproducibility and Experiment Traceability

The project incorporates reproducibility and traceability practices to support repeatable experimentation and transparent evaluation.

| Component | Purpose |
|---|---|
| SHA-256 data fingerprints | Dataset identity verification |
| Fixed random seed | Repeatable experimental configuration |
| Environment capture | Package and execution environment records |
| Frozen acceptance criteria | Pre-declared evaluation targets |
| MLflow tracking | Experiment and training-run tracking |
| Bootstrap confidence intervals | Quantification of test uncertainty |
| Validation-only temperature scaling | Separation of calibration and test evaluation |
| Automated model card generation | Consolidated model documentation |

### Generated evaluation artefacts

A full notebook execution writes outputs to `dermabench_outputs/`, including:

- Figures and evaluation visualisations
- CSV and JSON result tables
- Training history
- Dataset fingerprints
- Run configuration and environment details
- Frozen acceptance criteria and verdict
- MLflow database
- Generated `MODEL_CARD.md`

The helper-function sanity checks for evaluation utilities are included within the notebook.

---

## 9. Running the Project

The primary experiment is provided in:

`notebooks/DermaBench_Kaggle.ipynb`

### Execution environment

1. Open the notebook in Kaggle.
2. Enable GPU acceleration.
3. Enable Internet access for dependency and pretrained-weight retrieval.
4. Execute the notebook cells in order.

The recorded full-run execution takes approximately 15–25 minutes on a Tesla T4 GPU at 224 × 224 resolution.

For a quick smoke run, use 64 × 64 image resolution and three epochs in the configuration cell.

---

## 10. Repository Structure

```text
DermaBench/
├── README.md
├── requirements.txt
├── configs/
├── notebooks/
│   └── DermaBench_Kaggle.ipynb
├── src/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── visualization/
├── tests/
├── results/
│   └── figures/
│       ├── class_distribution.png
│       └── sample_images.png
├── train.py
└── evaluate.py
```

---

## 11. Limitations and Responsible Use

Several limitations affect the interpretation and generalisability of the results:

- **No demographic metadata:** Age, sex, skin tone and anatomical site subgroup performance could not be assessed.
- **Unverified lesion-level split independence:** The dataset does not provide the identifiers required to verify lesion-disjoint splits.
- **Single-source dataset:** The study uses a downsampled dermoscopy dataset without external validation.
- **Single training seed:** Training variance across multiple seeds was not evaluated.
- **Illustrative acceptance thresholds:** The thresholds were selected for this research prototype, not derived from clinical requirements.
- **Synthetic corruption:** Artificial degradation does not fully represent real-world acquisition variability.
- **Limited explainability review:** CAM outputs were qualitatively inspected on only six examples.
- **Small subgroup counts:** Some image-quality subgroups contain few melanoma cases and have no confidence intervals.

This work is not clinically validated and should not be interpreted as evidence of diagnostic performance in a healthcare setting.

---

## 12. Future Work

Planned improvements include:

- Refactoring evaluation utilities into an installable Python package.
- Expanding automated testing and introducing a pytest suite.
- Implementing GitHub Actions for continuous integration.
- Introducing DVC for dataset and checkpoint versioning.
- Completing systematic annotation of the top-50 confident errors.
- Running multiple training seeds to quantify model variance.
- Evaluating corruption-aware augmentation and recalibration.
- Validating on raw HAM10000 using lesion-grouped splits.
- Performing demographic subgroup analysis when suitable metadata are available.
- Investigating operating-point selection and abstention for low-quality images.

These are planned extensions, not completed project capabilities.

---

## 13. References

1. Yang J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023.
2. Doerrich S. et al. *Rethinking model prototyping through the MedMNIST+ dataset collection.* arXiv:2404.15786, 2024.
3. Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 2018.
4. Guo C. et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
5. Dosovitskiy A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
6. He K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.

**Libraries:** PyTorch, timm, MedMNIST, MLflow and pytorch-grad-cam.

---

## 14. Author

**C. Akshatha Shivani**

Research interests: Healthcare AI, medical imaging, clinical decision support, model evaluation and trustworthy machine learning.


