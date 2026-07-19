"""
==============================================================================
utils.py

Research-Grade Immunotoxicity Prediction Pipeline

Author : Prabhleen Kaur Saini
Project: IISER Mohali BS-MS Research Project

Purpose
-------
Loading helpers for everything predict.py needs: trained models (RF/XGB/
CatBoost/DNN/GINE), preprocessing artifacts, GNN thresholds, and ensemble
weights. Also handles device selection.

Expected directory layout (created by the training notebook, Cells 50-51):

    deployment/
        models/
            random_forest/{endpoint}.pkl
            xgboost/{endpoint}.pkl
            catboost/{endpoint}.pkl
            dnn/{endpoint}.pt              (state_dict)
            gine/{endpoint}.pt             (state_dict)
        preprocessing/
            feature_columns.pkl
            descriptor_names.pkl
            descriptor_medians.pkl
        config/
            gnn_thresholds.pkl
            ensemble_weights.json          (added by the export cell below)
        metadata/
            project_metadata.json
==============================================================================
"""

import json
import logging
import os
from typing import Dict, List

import joblib
import torch

from .models import DeepToxNet, GINEModel

logger = logging.getLogger("deployment.utils")

TARGET_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

NODE_FEATURE_DIM = 40
EDGE_FEATURE_DIM = 11

DEFAULT_ENSEMBLE_WEIGHTS = {
    "RF": 0.22, "XGB": 0.22, "CAT": 0.22, "DNN": 0.22, "GNN": 0.12
}


# =============================================================================
# Device
# =============================================================================

def get_device() -> torch.device:
    """Select CUDA if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


# =============================================================================
# Preprocessing Artifacts
# =============================================================================

def load_preprocessing_artifacts(deployment_dir: str = "deployment") -> Dict:
    """
    Load everything feature_engineering.py needs to reconstruct the 851-
    column tabular feature vector.
    """

    prep_dir = os.path.join(deployment_dir, "preprocessing")

    feature_columns = joblib.load(os.path.join(prep_dir, "feature_columns.pkl"))
    descriptor_names_final = joblib.load(os.path.join(prep_dir, "descriptor_names.pkl"))
    descriptor_medians = joblib.load(os.path.join(prep_dir, "descriptor_medians.pkl"))

    return {
        "feature_columns": feature_columns,
        "descriptor_names_final": descriptor_names_final,
        "descriptor_medians": descriptor_medians,
        "n_features": len(feature_columns)
    }


# =============================================================================
# Thresholds & Ensemble Weights
# =============================================================================

def load_gnn_thresholds(deployment_dir: str = "deployment") -> Dict[str, float]:
    """Load per-endpoint GNN classification thresholds (Cell 41)."""

    path = os.path.join(deployment_dir, "config", "gnn_thresholds.pkl")

    if not os.path.exists(path):
        logger.warning("gnn_thresholds.pkl not found - defaulting every "
                        "endpoint's GNN threshold to 0.50.")
        return {target: 0.50 for target in TARGET_COLUMNS}

    return joblib.load(path)


def load_ensemble_weights(deployment_dir: str = "deployment") -> Dict[str, Dict[str, float]]:
    """
    Load per-endpoint ensemble weights (Cell 42 / model_weights). Falls back
    to a flat default split if the export cell hasn't been run yet, so
    predict.py still works (with a logged warning) rather than crashing.
    """

    path = os.path.join(deployment_dir, "config", "ensemble_weights.json")

    if not os.path.exists(path):
        logger.warning(
            "ensemble_weights.json not found - falling back to equal-ish "
            "default weights for every endpoint. Run the "
            "'Export Ensemble Weights' notebook cell to fix this."
        )
        return {target: dict(DEFAULT_ENSEMBLE_WEIGHTS) for target in TARGET_COLUMNS}

    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# Model Loading
# =============================================================================

def load_all_models(
    deployment_dir: str = "deployment",
    n_features: int = None,
    device: torch.device = None
) -> Dict[str, Dict]:
    """
    Load all 5 model families for all 12 endpoints.

    Parameters
    ----------
    deployment_dir : str
    n_features : int
        Required to reconstruct DeepToxNet's input layer. If None, it is
        read from preprocessing/feature_columns.pkl.
    device : torch.device

    Returns
    -------
    dict with keys "RF", "XGB", "CAT", "DNN", "GNN", each mapping
    endpoint -> loaded model (sklearn/xgboost/catboost object, or a
    DeepToxNet/GINEModel in eval() mode on `device`).
    """

    if device is None:
        device = get_device()

    if n_features is None:
        n_features = load_preprocessing_artifacts(deployment_dir)["n_features"]

    models_dir = os.path.join(deployment_dir, "models")

    models = {"RF": {}, "XGB": {}, "CAT": {}, "DNN": {}, "GNN": {}}

    for endpoint in TARGET_COLUMNS:

        models["RF"][endpoint] = joblib.load(
            os.path.join(models_dir, "random_forest", f"{endpoint}.pkl"))

        models["XGB"][endpoint] = joblib.load(
            os.path.join(models_dir, "xgboost", f"{endpoint}.pkl"))

        models["CAT"][endpoint] = joblib.load(
            os.path.join(models_dir, "catboost", f"{endpoint}.pkl"))

        dnn = DeepToxNet(input_dim=n_features)
        dnn.load_state_dict(torch.load(
            os.path.join(models_dir, "dnn", f"{endpoint}.pt"),
            map_location=device
        ))
        dnn.to(device).eval()
        models["DNN"][endpoint] = dnn

        gine = GINEModel(node_dim=NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM)
        gine.load_state_dict(torch.load(
            os.path.join(models_dir, "gine", f"{endpoint}.pt"),
            map_location=device
        ))
        gine.to(device).eval()
        models["GNN"][endpoint] = gine

    logger.info(f"Loaded all 5 model families for {len(TARGET_COLUMNS)} endpoints.")

    return models


def load_project_metadata(deployment_dir: str = "deployment") -> Dict:
    """Load Cell 51's project_metadata.json, if present."""

    path = os.path.join(deployment_dir, "metadata", "project_metadata.json")

    if not os.path.exists(path):
        logger.warning("project_metadata.json not found.")
        return {}

    with open(path, "r") as f:
        return json.load(f)
