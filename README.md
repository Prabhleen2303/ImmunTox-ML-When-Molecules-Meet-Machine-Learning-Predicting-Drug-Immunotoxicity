# ImmunTox ML When Molecules Meet Machine Learning Predicting Drug Immunotoxicity
Can a machine learn what makes a drug dangerous to your immune system?  This pipeline says yes — 7,831 compounds, 12 toxicity endpoints, 6 models  (RF → XGBoost → CatBoost → DNN → GNN), ROC-AUC 0.86, full SHAP explainability  and a deployment-ready prediction API. Built with RDKit · PyTorch · PyG · Tox21.
Immunotoxicity Drug Prediction — ML Pipeline


Multi-model machine learning pipeline for predicting drug immunotoxicity across 12 Tox21 assay endpoints, combining classical ML, deep neural networks, and graph neural networks with full explainability and deployment.




Table of Contents


Motivation
Dataset
Project Structure
Models
Results
Feature Engineering
Training Strategy
Explainability
Deployment
Installation
Quick Start
Limitations
Future Work
References



Motivation

Immunotoxicity is one of the most clinically dangerous and under-screened endpoints in drug development. When drugs suppress or dysregulate the immune system, patients become vulnerable to opportunistic infections, autoimmune reactions, and organ failure — effects that are often undetected until late clinical trials or post-market surveillance.

Why this matters:


~30% of drug failures are caused by unexpected toxicity
Immune-mediated toxicity drives a significant fraction of post-market withdrawals
Traditional in vitro immunotoxicity assays cost $10,000–$50,000 per compound and take weeks
The Tox21 programme identified 12 nuclear receptor and stress response assays directly linked to immune regulation


This pipeline screens thousands of compounds in minutes, flags immunotoxic candidates before wet-lab testing, and provides structural explanations (toxicophores) that guide medicinal chemistry.


Dataset

PropertyValueNameTox21 (NIH / MoleculeNet)Direct downloadhttps://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gzSecondary datasetUCI Drug-Induced Autoimmunity — https://archive.ics.uci.edu/dataset/1104Molecules7,831 unique compoundsAssay endpoints12 (nuclear receptor + stress response)Immunotoxicity assaysNR-AhR, SR-HSE, SR-MMP, NR-PPAR-gammaClass imbalance~8% positive (toxic) per assayMissing labels~23% per assayFormatSMILES strings + binary activity labelsLicenseOpen access (CC0)

Tox21 Assay Endpoints

AssayTargetImmunotoxicity LinkNR-AhRAryl Hydrocarbon Receptor⭐ Key immune gene regulatorSR-HSEHeat Shock Factor⭐ Stress-induced immune dysregulationSR-MMPMitochondrial Membrane Potential⭐ Immune cell cytotoxicityNR-PPAR-gammaPeroxisome Proliferator Activated Receptor⭐ Anti-inflammatory signallingNR-ARAndrogen ReceptorHormonal immune modulationNR-AR-LBDAndrogen Receptor LBDHormonal immune modulationNR-AromataseAromatase enzymeOestrogen-immune axisNR-EREstrogen ReceptorOestrogen-immune axisNR-ER-LBDEstrogen Receptor LBDOestrogen-immune axisSR-AREAntioxidant Response ElementOxidative stressSR-ATAD5DNA damage responseGenotoxicitySR-p53Tumour suppressor p53DNA damage response


Project Structure

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


Models

Six model types were built and compared:

1. Random Forest


500 estimators, class_weight=balanced
Input: Morgan FP (2048 bits) + MACCS keys (167 bits) + descriptors (10)
Best precision (0.7705) — fewest false positives


2. XGBoost


300 estimators, scale_pos_weight for imbalance
subsample=0.8, colsample_bytree=0.8, learning_rate=0.05
Strong overall F1 performance


3. CatBoost


Native categorical handling, symmetric tree structure
SHAP TreeExplainer used for toxicophore analysis
Competitive with XGBoost, faster training


4. Deep Neural Network (DNN)


Architecture: 851 → 512 → 256 → 128 → 64 → 1
BatchNorm1d + Dropout(0.3) after each layer, Sigmoid output
Focal loss (γ=2.0, α=0.75) for class imbalance
Trained on GPU (CUDA), Adam optimiser, weight decay=1e-5


5. GNN — GINE (Graph Isomorphism Network with Edge features)


Molecules as graphs: atoms = nodes (40 features), bonds = edges (11 features)
4 GINE convolutional layers with BatchNorm + Dropout
Global mean pooling → MLP head
Per-endpoint threshold optimisation on validation set


6. Performance-Weighted Ensemble ⭐ Best


Combines RF + XGB + CatBoost + DNN + GNN
Weights computed per-endpoint from out-of-fold validation ROC-AUC
Better models on a given endpoint get higher weight for that endpoint



Results

Overall Performance (Averaged Across 12 Endpoints)

ModelROC-AUCPR-AUCF1MCCPrecisionRecallEnsemble0.86430.54420.50730.49520.66520.4273Random Forest0.86060.52520.45020.47000.77050.3405XGBoost0.85120.52680.49820.48390.64660.4182CatBoost0.84550.52500.48690.47600.65170.4065DNN0.82460.48750.48170.44050.47050.5026GNN (GINE)0.82190.40550.42140.38580.45770.4229

Per-Endpoint Performance

EndpointMean ROC-AUCMean PR-AUCDifficultySR-MMP0.92030.7450EasyNR-AhR0.90590.6380EasyNR-AR-LBD0.89120.6592EasySR-ATAD50.89240.4593EasySR-p530.88710.4255EasySR-ARE0.85620.5863MediumNR-Aromatase0.85210.4977MediumNR-ER-LBD0.83760.4879MediumNR-PPAR-gamma0.80210.2563HardSR-HSE0.81930.3811HardNR-ER0.73650.4435HardNR-AR0.73570.4483Hard

Bootstrap Confidence Intervals (ROC-AUC)

ModelMean95% CI Lower95% CI UpperEnsemble0.86430.82980.8941Random Forest0.86060.82480.8899XGBoost0.85120.81420.8856CatBoost0.84550.80760.8788DNN0.82460.78850.8581GNN (GINE)0.82190.78650.8561


Feature Engineering

Three complementary molecular representations computed from SMILES:

1. Morgan Fingerprints (ECFP4)


Circular fingerprint, radius=2, 2048 bits
Encodes local chemical environment around each atom up to 2 bonds away
Industry standard for structure-activity relationship modelling


2. MACCS Keys


167 predefined structural keys (pharmacophoric patterns)
Captures functional groups: aromatic rings, halogens, carbonyls, nitro groups


3. Physicochemical Descriptors (RDKit)


Molecular weight, LogP, H-bond donors/acceptors, TPSA
Rotatable bonds, aromatic rings, heavy atom count, ring count
Directly interpretable, linked to ADMET properties


4. Graph Representation (GNN only)


Atoms = nodes: atomic number, degree, hybridisation, aromaticity, formal charge, H count (40 features)
Bonds = edges: bond type, conjugation, ring membership (11 features)
Preserves full molecular topology with no information loss from hashing


Combined feature vector (classical ML/DNN): 851 features


Training Strategy

Split


Scaffold split (Bemis-Murcko) — molecules with same core scaffold go to same partition
Train 80% / Validation 10% / Test 10%
Prevents data leakage, simulates deployment on novel chemical scaffolds


Imbalance Handling


SMOTE oversampling on training set only (never applied to test set)
Focal loss (γ=2.0, α=0.75) in DNN
scale_pos_weight in XGBoost/CatBoost
class_weight=balanced in Random Forest


Cross-Validation


5-fold stratified CV for classical models
Per-endpoint threshold optimisation for GNN (maximise F1 on validation)
Out-of-fold ensemble weight computation


Evaluation Metrics


ROC-AUC — overall discrimination (inflated for imbalanced data)
PR-AUC — precision-recall, most relevant for minority (toxic) class
F1 score — harmonic mean of precision and recall
MCC — Matthews Correlation Coefficient, robust to imbalance
Calibration — Brier score and calibration curves
Bootstrap CI — 1000 resample confidence intervals



Explainability

SHAP Analysis


TreeExplainer applied to CatBoost models across all 12 endpoints
300-molecule sample per endpoint for computational efficiency
Top 15 Morgan fingerprint bits ranked by mean |SHAP| value


Toxicophore Decoding


Each top Morgan bit decoded to its molecular substructure via rdkit.Chem.Draw.DrawMorganBit
SVG images saved to figures/toxicophore_bit_*.svg
Results table saved to deployment/results/toxicophore_analysis.csv


Key Toxicophore Findings

Top SHAP-important structural features correspond to:


Aromatic ring systems with electron-withdrawing groups
Halogenated hydrocarbons (Cl, F substitutions)
Nitro groups and quinone-like structures
Ester/amide linkages adjacent to aromatic systems


These align with well-known immunotoxic pharmacophores in the literature.


Deployment

Quick Start

pythonfrom deployment.predict import ImmunotoxicityPredictor

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

Output Format (per endpoint)

python{
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

Aspirin Prediction (Validation Example)

EndpointPredictionProbabilityConfidenceAgreementNR-ARNon-Toxic0.00399.4%100%NR-AhRNon-Toxic0.01696.9%100%SR-MMPNon-Toxic0.01297.6%100%SR-ARENon-Toxic0.04491.2%100%

All 12 endpoints: Non-Toxic ✅ (correct — aspirin is not immunotoxic at standard doses)


Installation

Requirements

bashpip install rdkit
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

Python Version

Tested on Python 3.11.9 (Windows 11 25H2, CUDA GPU)

Known Issues


If from rdkit import Chem fails with DLL load failed: disable Windows Smart App Control (Settings → Windows Security → App & browser control → Smart App Control → Off), then restart
predictor.scaler must be set to None after loading (scaler was saved as unfitted class in training pipeline)



Limitations


In vitro only — Tox21 assays are cell-free or single cell-line; in vitro activity ≠ in vivo immunotoxicity
Class imbalance — 5-20% positive labels; PR-AUC of 0.54 leaves significant room for improvement
Chemical space — Trained on drug-like small molecules; performance on peptides, macrocycles, or natural products unknown
Missing labels — ~23% per assay limits training data for some endpoints
Scaler bug — StandardScaler stored as unfitted class; workaround: predictor.scaler = None
No external validation — No fully held-out external test set from a different source



Future Work


Transfer learning — Fine-tune ChemBERTa/MolBERT pre-trained on 77M SMILES (+3-5% AUC expected)
Multi-task GNN — Train single model on all 12 endpoints simultaneously
3D conformer features — SchNet/DimeNet++ for geometry-aware predictions
Larger datasets — Integrate ToxCast (EPA, ~10,000 compounds, 700+ assays)
Active learning — Use uncertainty to select informative compounds for wet-lab testing
REST API — FastAPI endpoint with Docker container
Fix scaler — Refit and re-save StandardScaler correctly in training pipeline



References


Tox21 Challenge Dataset — https://tripod.nih.gov/tox21/challenge/
MoleculeNet: A Benchmark for Molecular Machine Learning — Wu et al., 2018
GINE: Strategies for Pre-training Graph Neural Networks — Hu et al., ICLR 2020
SHAP: A Unified Approach to Interpreting Model Predictions — Lundberg & Lee, NeurIPS 2017
AttentiveFP: Pushing the Boundaries of Molecular Representation — Xiong et al., JCIM 2020
DeepChem: Democratizing Deep Learning for Drug Discovery — Ramsundar et al., 2019
RDKit: Open-Source Cheminformatics — https://www.rdkit.org
UCI Drug-Induced Autoimmunity Dataset — https://archive.ics.uci.edu/dataset/1104
Focal Loss for Dense Object Detection — Lin et al., ICCV 2017
SMOTE: Synthetic Minority Over-sampling Technique — Chawla et al., JAIR 2002



Citation

If you use this pipeline in your work, please cite:

@misc{immunotox2026,
  title  = {Immunotoxicity Drug Prediction — Multi-Model ML Pipeline},
  author = {Prabhleen, IISER Mohali},
  year   = {2026},
  note   = { NPTEL Summer Internship Project IIT-BHU (Guide-Dr.Rajnish Kumar[IIT-BHU]), IISER Mohali — Biology Major, Data Science Minor}
}


Author

Prabhleen
BS-MS Student · Fourth Year · Biology Major, Data Science Minor
Indian Institute of Science Education and Research (IISER) Mohali
📧 ms23170@iisermohali.ac.in
📅 July 2026
