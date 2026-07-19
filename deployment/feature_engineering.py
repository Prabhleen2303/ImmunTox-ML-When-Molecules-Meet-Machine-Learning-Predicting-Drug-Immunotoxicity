"""
==============================================================================
feature_engineering.py

Research-Grade Immunotoxicity Prediction Pipeline

Author : Prabhleen Kaur Saini
Project: IISER Mohali BS-MS Research Project

Purpose
-------
Reconstructs the exact tabular feature vector used to train the Random
Forest, XGBoost, CatBoost, and DNN models, for a new SMILES string at
inference time.

Design note
-----------
Training built the tabular matrix as:

    RDKit descriptors (216) + Morgan/ECFP4 fingerprint (2048) + MACCS (167)
    -> VarianceThreshold(0.01) -> correlation filter (>0.95) -> 851 features

Rather than re-implementing the VarianceThreshold / correlation-drop logic
here (which would require access to the full training matrix and would risk
silently drifting from what was actually fit), this module regenerates the
FULL raw feature superset (216 + 2048 + 167 columns) and then reindexes it
to the exact `feature_columns.pkl` list saved during training. This
guarantees column-for-column parity with what the models were trained on,
using the artifact that is the actual source of truth.
==============================================================================
"""

import logging
from typing import List, Union

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, DataStructs

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger("deployment.feature_engineering")

# -----------------------------------------------------------------------------
# Constants (must match training - Cell 9 / Cell 10)
# -----------------------------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
MACCS_BITS = 167


# =============================================================================
# SMILES Validation (matches training's inline validators)
# =============================================================================

def validate_smiles(smiles: str) -> str:
    """
    Validate a SMILES string.

    Parameters
    ----------
    smiles : str

    Returns
    -------
    str
        The cleaned (stripped) SMILES string.

    Raises
    ------
    TypeError
        If input is not a string.
    ValueError
        If the SMILES is empty or does not parse to a valid molecule.
    """

    if not isinstance(smiles, str):
        raise TypeError(f"SMILES must be a string, got {type(smiles)}")

    smiles = smiles.strip()

    if len(smiles) == 0:
        raise ValueError("SMILES string is empty.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES (failed to parse): {smiles}")

    return smiles


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Convert a validated SMILES string into a sanitized RDKit molecule.
    """

    smiles = validate_smiles(smiles)

    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)

    return mol


# =============================================================================
# RDKit Descriptors
# =============================================================================

def compute_descriptors(
    mol: Chem.Mol,
    descriptor_names_final: List[str],
    descriptor_medians: pd.Series
) -> pd.Series:
    """
    Compute only the descriptors that survived training's stability filter
    (Cell 8 dropped any descriptor whose training-set max abs value exceeded
    1e6, e.g. Ipc). Any descriptor that fails to compute for this molecule
    (RDKit exception, NaN, or inf) is imputed with its TRAINING median, not
    this molecule's own value - this matches the imputation training used
    and avoids inconsistent single-molecule statistics.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    descriptor_names_final : list of str
        Loaded from deployment/preprocessing/descriptor_names.pkl
    descriptor_medians : pd.Series
        Loaded from deployment/preprocessing/descriptor_medians.pkl

    Returns
    -------
    pd.Series indexed by descriptor name
    """

    all_descriptors = dict(Descriptors._descList)

    values = {}

    for name in descriptor_names_final:

        func = all_descriptors.get(name)

        if func is None:
            # Should not happen if descriptor_names_final came from this
            # RDKit version, but fail safe to the training median.
            logger.warning(f"Descriptor '{name}' not found in this RDKit "
                            f"version - falling back to training median.")
            values[name] = descriptor_medians.get(name, 0.0)
            continue

        try:
            value = func(mol)

            if not np.isfinite(value):
                raise ValueError("non-finite descriptor value")

        except Exception:
            value = descriptor_medians.get(name, 0.0)

        values[name] = value

    return pd.Series(values, index=descriptor_names_final)


# =============================================================================
# Morgan Fingerprint (ECFP4)
# =============================================================================

def compute_morgan_fingerprint(mol: Chem.Mol) -> pd.Series:
    """
    Generate the 2048-bit Morgan (ECFP4, radius=2) fingerprint, matching
    training's Cell 9 exactly. Column names: FP_0 ... FP_2047.
    """

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=MORGAN_RADIUS,
        nBits=MORGAN_N_BITS
    )

    arr = np.zeros((MORGAN_N_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return pd.Series(arr, index=[f"FP_{i}" for i in range(MORGAN_N_BITS)])


# =============================================================================
# MACCS Keys
# =============================================================================

def compute_maccs_keys(mol: Chem.Mol) -> pd.Series:
    """
    Generate the 167-bit MACCS fingerprint, matching training's Cell 10
    exactly. Column names: MACCS_0 ... MACCS_166.
    """

    fp = MACCSkeys.GenMACCSKeys(mol)

    arr = np.zeros((MACCS_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return pd.Series(arr, index=[f"MACCS_{i}" for i in range(MACCS_BITS)])


# =============================================================================
# Full Feature Vector (single molecule)
# =============================================================================

def build_feature_vector(
    smiles: str,
    descriptor_names_final: List[str],
    descriptor_medians: pd.Series,
    feature_columns: List[str]
) -> pd.Series:
    """
    Build the full 851-column tabular feature vector for one SMILES string,
    in the exact column order the models were trained on.

    Parameters
    ----------
    smiles : str
    descriptor_names_final : list of str
        deployment/preprocessing/descriptor_names.pkl
    descriptor_medians : pd.Series
        deployment/preprocessing/descriptor_medians.pkl
    feature_columns : list of str
        deployment/preprocessing/feature_columns.pkl
        (the authoritative post-selection 851-column order)

    Returns
    -------
    pd.Series indexed by feature_columns, ready to feed to RF/XGB/CAT/DNN.
    """

    mol = smiles_to_mol(smiles)

    descriptors = compute_descriptors(mol, descriptor_names_final, descriptor_medians)
    morgan = compute_morgan_fingerprint(mol)
    maccs = compute_maccs_keys(mol)

    raw_features = pd.concat([descriptors, morgan, maccs])
    raw_features.index = raw_features.index.astype(str)

    # Reindex to the exact trained column set/order. Any column expected by
    # the models but not produced here (should not normally happen) is
    # filled with 0.0 rather than silently dropped.
    feature_vector = raw_features.reindex(feature_columns).fillna(0.0)

    return feature_vector.astype(np.float64)


def build_feature_matrix(
    smiles_list: List[str],
    descriptor_names_final: List[str],
    descriptor_medians: pd.Series,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Batch version of build_feature_vector. Invalid SMILES are skipped with
    a logged warning rather than raising, so one bad row doesn't kill an
    entire batch job; check `.attrs['failed_indices']` on the result for
    which input rows were dropped.
    """

    rows = []
    failed_indices = []

    for i, smiles in enumerate(smiles_list):

        try:
            rows.append(build_feature_vector(
                smiles,
                descriptor_names_final,
                descriptor_medians,
                feature_columns
            ))

        except (TypeError, ValueError) as e:
            logger.warning(f"Skipping row {i} ('{smiles}'): {e}")
            failed_indices.append(i)

    matrix = pd.DataFrame(rows, columns=feature_columns)
    matrix.attrs["failed_indices"] = failed_indices

    return matrix
