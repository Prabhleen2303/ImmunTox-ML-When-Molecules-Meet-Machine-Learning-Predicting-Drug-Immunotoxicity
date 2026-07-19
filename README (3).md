# ImmunTox-ML — When Molecules Meet Machine Learning: Predicting Drug Immunotoxicity

> **Multi-model machine learning pipeline for predicting drug immunotoxicity across 12 Tox21 assay endpoints, combining classical ML, deep neural networks, and graph neural networks with full explainability and deployment.**

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Feature Engineering](#feature-engineering)
- [Training Strategy](#training-strategy)
- [Explainability](#explainability)
- [Deployment](#deployment)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Motivation

Immunotoxicity is one of the most clinically dangerous and under-screened endpoints in drug development. When drugs suppress or dysregulate the immune system, patients become vulnerable to opportunistic infections, autoimmune reactions, and organ failure — effects that are often undetected until late clinical trials or post-market surveillance.

**Why this matters:**
- ~30% of drug failures are caused by unexpected toxicity
- Immune-mediated toxicity drives a significant fraction of post-market withdrawals
- Traditional in vitro immunotoxicity assays cost $10,000–$50,000 per compound and take weeks
- The Tox21 programme identified 12 nuclear receptor and stress response assays directly linked to immune regulation

This pipeline screens thousands of compounds in minutes, flags immunotoxic candidates before wet-lab testing, and provides structural explanations (toxicophores) that guide medicinal chemistry.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Tox21 (NIH / MoleculeNet) |
| **Direct download** | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz |
| **Secondary dataset** | UCI Drug-Induced Autoimmunity — https://archive.ics.uci.edu/dataset/1104 |
| **Molecules** | 7,831 unique compounds |
| **Assay endpoints** | 12 (nuclear receptor + stress response) |
| **Immunotoxicity assays** | NR-AhR, SR-HSE, SR-MMP, NR-PPAR-gamma |
| **Class imbalance** | ~8% positive (toxic) per assay |
| **Missing labels** | ~23% per assay |
| **Format** | SMILES strings + binary activity labels |
| **License** | Open access (CC0) |

### Tox21 Assay Endpoints

| Assay | Target | Immunotoxicity Link |
|---|---|---|
| NR-AhR | Aryl Hydrocarbon Receptor | ⭐ Key immune gene regulator |
| SR-HSE | Heat Shock Factor | ⭐ Stress-induced immune dysregulation |
| SR-MMP | Mitochondrial Membrane Potential | ⭐ Immune cell cytotoxicity |
| NR-PPAR-gamma | Peroxisome Proliferator Activated Receptor | ⭐ Anti-inflammatory signalling |
| NR-AR | Androgen Receptor | Hormonal immune modulation |
| NR-AR-LBD | Androgen Receptor LBD | Hormonal immune modulation |
| NR-Aromatase | Aromatase enzyme | Oestrogen-immune axis |
| NR-ER | Estrogen Receptor | Oestrogen-immune axis |
| NR-ER-LBD | Estrogen Receptor LBD | Oestrogen-immune axis |
| SR-ARE | Antioxidant Response Element | Oxidative stress |
| SR-ATAD5 | DNA damage response | Genotoxicity |
| SR-p53 | Tumour suppressor p53 | DNA damage response |

---

## Project Structure

```
Final Immunotoxicity model/
├── Script_40_50_complete.ipynb     # Complete pipeline notebook (136 cells)
├── deployment/
│   ├── predict.py                  # ImmunotoxicityPredictor class (production API)
│   ├── __init__.py
│   ├── models/
│   │   ├── random_forest/          # RF .pkl (1 per endpoint × 12)
│   │   ├── xgboost/                # XGB .pkl
│   │   ├── catboost/               # CatBoost .pkl
│   │   ├── dnn/                    # DNN .pt (PyTorch state dicts)
│   │   └── gine/                   # GNN .pt (GINE state dicts)
│   ├── preprocessing/
│   │   ├── feature_columns.pkl     # Ordered 851 feature names
│   │   ├── preprocessing_metadata.json
│   │   ├── descriptor_medians.pkl  # NaN imputation values
│   │   └── descriptor_names.pkl
│   └── config/
│       ├── ensemble_weights.json   # Per-endpoint model weights
│       └── gnn_thresholds.pkl      # Per-endpoint classification thresholds
├── figures/                        # All saved plots
│   ├── roc_curves/
│   ├── pr_curves/
│   ├── confusion_matrices/
│   ├── calibration_curves/
│   ├── shap_importance/
│   └── toxicophore_bit_*.svg       # SHAP-decoded molecular substructures
├── deployment/results/
│   └── toxicophore_analysis.csv    # Top 15 toxicophore Morgan bits
└── README.md
```

---

## Models

Six model types were built and compared:

### 1. Random Forest
- 500 estimators, `class_weight=balanced`
- Input: Morgan FP (2048 bits) + MACCS keys (167 bits) + descriptors (10)
- **Best precision (0.7705)** — fewest false positives

### 2. XGBoost
- 300 estimators, `scale_pos_weight` for imbalance
- `subsample=0.8`, `colsample_bytree=0.8`, `learning_rate=0.05`
- Strong overall F1 performance

### 3. CatBoost
- Native categorical handling, symmetric tree structure
- SHAP TreeExplainer used for toxicophore analysis
- Competitive with XGBoost, faster training

### 4. Deep Neural Network (DNN)
- Architecture: 851 → 512 → 256 → 128 → 64 → 1
- BatchNorm1d + Dropout(0.3) after each layer, Sigmoid output
- Focal loss (γ=2.0, α=0.75) for class imbalance
- Trained on GPU (CUDA), Adam optimiser, weight decay=1e-5

### 5. GNN — GINE (Graph Isomorphism Network with Edge features)
- Molecules as graphs: atoms = nodes (40 features), bonds = edges (11 features)
- 4 GINE convolutional layers with BatchNorm + Dropout
- Global mean pooling → MLP head
- Per-endpoint threshold optimisation on validation set

### 6. Performance-Weighted Ensemble ⭐ Best
- Combines RF + XGB + CatBoost + DNN + GNN
- Weights computed per-endpoint from out-of-fold validation ROC-AUC
- Better models on a given endpoint get higher weight for that endpoint

---

## Results

### Overall Performance (Averaged Across 12 Endpoints)

| Model | ROC-AUC | PR-AUC | F1 | MCC | Precision | Recall |
|---|---|---|---|---|---|---|
| **Ensemble** | **0.8643** | **0.5442** | **0.5073** | **0.4952** | 0.6652 | 0.4273 |
| Random Forest | 0.8606 | 0.5252 | 0.4502 | 0.4700 | **0.7705** | 0.3405 |
| XGBoost | 0.8512 | 0.5268 | 0.4982 | 0.4839 | 0.6466 | 0.4182 |
| CatBoost | 0.8455 | 0.5250 | 0.4869 | 0.4760 | 0.6517 | 0.4065 |
| DNN | 0.8246 | 0.4875 | 0.4817 | 0.4405 | 0.4705 | 0.5026 |
| GNN (GINE) | 0.8219 | 0.4055 | 0.4214 | 0.3858 | 0.4577 | 0.4229 |

### Per-Endpoint Performance

| Endpoint | Mean ROC-AUC | Mean PR-AUC | Difficulty |
|---|---|---|---|
| **SR-MMP** | **0.9203** | 0.7450 | Easy |
| NR-AhR | 0.9059 | 0.6380 | Easy |
| NR-AR-LBD | 0.8912 | 0.6592 | Easy |
| SR-ATAD5 | 0.8924 | 0.4593 | Easy |
| SR-p53 | 0.8871 | 0.4255 | Easy |
| SR-ARE | 0.8562 | 0.5863 | Medium |
| NR-Aromatase | 0.8521 | 0.4977 | Medium |
| NR-ER-LBD | 0.8376 | 0.4879 | Medium |
| NR-PPAR-gamma | 0.8021 | 0.2563 | **Hard** |
| SR-HSE | 0.8193 | 0.3811 | Hard |
| NR-ER | 0.7365 | 0.4435 | Hard |
| NR-AR | 0.7357 | 0.4483 | Hard |

### Bootstrap Confidence Intervals (ROC-AUC)

| Model | Mean | 95% CI Lower | 95% CI Upper |
|---|---|---|---|
| Ensemble | 0.8643 | 0.8298 | 0.8941 |
| Random Forest | 0.8606 | 0.8248 | 0.8899 |
| XGBoost | 0.8512 | 0.8142 | 0.8856 |
| CatBoost | 0.8455 | 0.8076 | 0.8788 |
| DNN | 0.8246 | 0.7885 | 0.8581 |
| GNN (GINE) | 0.8219 | 0.7865 | 0.8561 |

---

## Feature Engineering

Three complementary molecular representations computed from SMILES:

### 1. Morgan Fingerprints (ECFP4)
- Circular fingerprint, radius=2, 2048 bits
- Encodes local chemical environment around each atom up to 2 bonds away
- Industry standard for structure-activity relationship modelling

### 2. MACCS Keys
- 167 predefined structural keys (pharmacophoric patterns)
- Captures functional groups: aromatic rings, halogens, carbonyls, nitro groups

### 3. Physicochemical Descriptors (RDKit)
- Molecular weight, LogP, H-bond donors/acceptors, TPSA
- Rotatable bonds, aromatic rings, heavy atom count, ring count
- Directly interpretable, linked to ADMET properties

### 4. Graph Representation (GNN only)
- Atoms = nodes: atomic number, degree, hybridisation, aromaticity, formal charge, H count (40 features)
- Bonds = edges: bond type, conjugation, ring membership (11 features)
- Preserves full molecular topology with no information loss from hashing

**Combined feature vector (classical ML/DNN): 851 features**

---

## Training Strategy

### Split
- **Scaffold split** (Bemis-Murcko) — molecules with same core scaffold go to same partition
- Train 80% / Validation 10% / Test 10%
- Prevents data leakage, simulates deployment on novel chemical scaffolds

### Imbalance Handling
- SMOTE oversampling on training set only (never applied to test set)
- Focal loss (γ=2.0, α=0.75) in DNN
- `scale_pos_weight` in XGBoost/CatBoost
- `class_weight=balanced` in Random Forest

### Cross-Validation
- 5-fold stratified CV for classical models
- Per-endpoint threshold optimisation for GNN (maximise F1 on validation)
- Out-of-fold ensemble weight computation

### Evaluation Metrics
- **ROC-AUC** — overall discrimination (inflated for imbalanced data)
- **PR-AUC** — precision-recall, most relevant for minority (toxic) class
- **F1 score** — harmonic mean of precision and recall
- **MCC** — Matthews Correlation Coefficient, robust to imbalance
- **Calibration** — Brier score and calibration curves
- **Bootstrap CI** — 1000 resample confidence intervals

---

## Explainability

### SHAP Analysis
- TreeExplainer applied to CatBoost models across all 12 endpoints
- 300-molecule sample per endpoint for computational efficiency
- Top 15 Morgan fingerprint bits ranked by mean |SHAP| value

### Toxicophore Decoding
- Each top Morgan bit decoded to its molecular substructure via `rdkit.Chem.Draw.DrawMorganBit`
- SVG images saved to `figures/toxicophore_bit_*.svg`
- Results table saved to `deployment/results/toxicophore_analysis.csv`

### Key Toxicophore Findings
Top SHAP-important structural features correspond to:
- Aromatic ring systems with electron-withdrawing groups
- Halogenated hydrocarbons (Cl, F substitutions)
- Nitro groups and quinone-like structures
- Ester/amide linkages adjacent to aromatic systems

These align with well-known immunotoxic pharmacophores in the literature.

---

## Deployment

### Quick Start

```python
from deployment.predict import ImmunotoxicityPredictor

# Load predictor
predictor = ImmunotoxicityPredictor(deployment_dir="deployment")
predictor.scaler = None  # scaler serialisation bug in training pipeline

# Predict single molecule
result = predictor.predict_smiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
print(result["endpoints"])

# Batch prediction
import pandas as pd
smiles_list = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
df = predictor.predict_batch(smiles_list)
print(df)
```

### Output Format (per endpoint)

```python
{
  "NR-AhR": {
    "ensemble_probability": 0.016,   # Calibrated toxicity probability
    "ensemble_prediction": 0,         # 0=Non-toxic, 1=Toxic
    "model_agreement": 1.0,           # Fraction of models agreeing
    "individual_probs": {
      "RF": 0.012, "XGB": 0.018,
      "CAT": 0.014, "DNN": 0.019, "GNN": 0.017
    },
    "threshold_used": 0.5
  }
}
```

### Aspirin Prediction (Validation Example)

| Endpoint | Prediction | Probability | Confidence | Agreement |
|---|---|---|---|---|
| NR-AR | Non-Toxic | 0.003 | 99.4% | 100% |
| NR-AhR | Non-Toxic | 0.016 | 96.9% | 100% |
| SR-MMP | Non-Toxic | 0.012 | 97.6% | 100% |
| SR-ARE | Non-Toxic | 0.044 | 91.2% | 100% |

All 12 endpoints: Non-Toxic ✅ (correct — aspirin is not immunotoxic at standard doses)

---

## Installation

### Requirements

```bash
pip install rdkit
pip install scikit-learn==1.4.2
pip install xgboost==2.0.3
pip install catboost
pip install imbalanced-learn==0.12.3
pip install torch==2.2.2
pip install torch_geometric==2.5.3
pip install shap==0.45.1
pip install matplotlib==3.8.4
pip install seaborn==0.13.2
pip install ucimlrepo
pip install pandas numpy joblib
```

### Python Version
Tested on **Python 3.11.9** (Windows 11 25H2, CUDA GPU)

### Known Issues
- If `from rdkit import Chem` fails with `DLL load failed`: disable Windows Smart App Control (Settings → Windows Security → App & browser control → Smart App Control → Off), then restart
- `predictor.scaler` must be set to `None` after loading (scaler was saved as unfitted class in training pipeline)

---

## Limitations

1. **In vitro only** — Tox21 assays are cell-free or single cell-line; in vitro activity ≠ in vivo immunotoxicity
2. **Class imbalance** — 5-20% positive labels; PR-AUC of 0.54 leaves significant room for improvement
3. **Chemical space** — Trained on drug-like small molecules; performance on peptides, macrocycles, or natural products unknown
4. **Missing labels** — ~23% per assay limits training data for some endpoints
5. **Scaler bug** — StandardScaler stored as unfitted class; workaround: `predictor.scaler = None`
6. **No external validation** — No fully held-out external test set from a different source

---

## Future Work

- **Transfer learning** — Fine-tune ChemBERTa/MolBERT pre-trained on 77M SMILES (+3-5% AUC expected)
- **Multi-task GNN** — Train single model on all 12 endpoints simultaneously
- **3D conformer features** — SchNet/DimeNet++ for geometry-aware predictions
- **Larger datasets** — Integrate ToxCast (EPA, ~10,000 compounds, 700+ assays)
- **Active learning** — Use uncertainty to select informative compounds for wet-lab testing
- **REST API** — FastAPI endpoint with Docker container
- **Fix scaler** — Refit and re-save StandardScaler correctly in training pipeline

---

## References

1. Tox21 Challenge Dataset — https://tripod.nih.gov/tox21/challenge/
2. MoleculeNet: A Benchmark for Molecular Machine Learning — Wu et al., 2018
3. GINE: Strategies for Pre-training Graph Neural Networks — Hu et al., ICLR 2020
4. SHAP: A Unified Approach to Interpreting Model Predictions — Lundberg & Lee, NeurIPS 2017
5. AttentiveFP: Pushing the Boundaries of Molecular Representation — Xiong et al., JCIM 2020
6. DeepChem: Democratizing Deep Learning for Drug Discovery — Ramsundar et al., 2019
7. RDKit: Open-Source Cheminformatics — https://www.rdkit.org
8. UCI Drug-Induced Autoimmunity Dataset — https://archive.ics.uci.edu/dataset/1104
9. Focal Loss for Dense Object Detection — Lin et al., ICCV 2017
10. SMOTE: Synthetic Minority Over-sampling Technique — Chawla et al., JAIR 2002

---

## Acknowledgements

This project was developed during an **NPTEL Summer Internship 2026** at **IIT BHU Varanasi**.

I would like to sincerely thank:

- **Dr. Rajnish Kumar** — Department of Pharmaceutical Engineering and Technology, Indian Institute of Technology (BHU) Varanasi, for his invaluable guidance, mentorship, and introducing me to the intersection of computational chemistry and drug safety
- **Indian Institute of Technology (BHU) Varanasi** — for providing the research environment, computational resources, and the opportunity to work on a real-world pharmaceutical problem
- **NPTEL** — for facilitating this internship programme connecting students across premier Indian institutions
- **IISER Mohali** — for the strong foundation in biology and data science that made this project possible
- **MoleculeNet / NIH Tox21** — for providing the open-access dataset that underpins this entire pipeline

---

## Citation

If you use this pipeline in your work, please cite:

```
@misc{immunotox2026,
  title       = {ImmunTox-ML: Multi-Model Machine Learning Pipeline for Drug Immunotoxicity Prediction},
  author      = {Saini, Prabhleen Kaur},
  year        = {2026},
  institution = {Indian Institute of Technology (BHU) Varanasi},
  note        = {NPTEL Summer Internship, 1 June 2026 -- 26 July 2026.
                 Supervised by Dr. Rajnish Kumar, Department of Pharmaceutical
                 Engineering and Technology, Indian Institute of Technology (BHU)
                 Varanasi. Author is a BS-MS Student (4th Year, Biology Major,
                 Data Science Minor), IISER Mohali.
                 Contact: ms23170@iisermohali.ac.in}
}
```

---

## Author

**Prabhleen Kaur Saini**  
BS-MS Student (4th Year) · Biology Major · Data Science Minor  
Indian Institute of Science Education and Research (IISER) Mohali  
📧 ms23170@iisermohali.ac.in

---

### About This Project

This project was carried out as part of an **NPTEL Summer Internship (1 June 2026 — 26 July 2026)** at the  
**Indian Institute of Technology (BHU) Varanasi**  
under the supervision of **Dr. Rajnish Kumar**  
Department of Pharmaceutical Engineering and Technology, IIT (BHU) Varanasi.

The internship focused on applying machine learning and cheminformatics to drug safety — specifically building a computational pipeline to predict immunotoxicity from molecular structure, bridging the gap between computational biology and pharmaceutical sciences.

| | |
|---|---|
| **Intern** | Prabhleen Kaur Saini |
| **Home Institute** | IISER Mohali (BS-MS, 4th Year — Biology Major, Data Science Minor) |
| **Supervisor** | Dr. Rajnish Kumar |
| **Department** | Department of Pharmaceutical Engineering and Technology, IIT (BHU) Varanasi |
| **Programme** | NPTEL Summer Internship 2026 |
| **Duration** | 1 June 2026 — 26 July 2026 |
| **Contact** | ms23170@iisermohali.ac.in |
