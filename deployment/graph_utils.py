"""
==============================================================================
graph_utils.py

Research-Grade Immunotoxicity Prediction Pipeline

Author : Prabhleen Kaur Saini
Project: IISER Mohali BS-MS Research Project

Purpose
-------
Builds a torch_geometric.data.Data graph for a single SMILES string, using
the exact same atom/bond feature encoders as training (Cell 37A / Cell 37B).
Node feature dimension: 40. Edge feature dimension: 11.

These encoders are copied verbatim from the training notebook. Do not
change them without retraining the GINE model - the saved state_dict's
input_projection layer is shaped for exactly 40 node features / 11 edge
features.
==============================================================================
"""

from typing import Tuple

import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

from .feature_engineering import smiles_to_mol

# -----------------------------------------------------------------------------
# Atom / Bond Categories (must match Cell 37A exactly)
# -----------------------------------------------------------------------------

ATOM_SYMBOLS = [
    "B", "C", "N", "O", "F",
    "Si", "P", "S", "Cl", "Br",
    "I", "Unknown"
]

DEGREES = [0, 1, 2, 3, 4, 5]

HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2
]

CHIRALITY = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER
]

TOTAL_H = [0, 1, 2, 3, 4]

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
]

STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE
]

NODE_FEATURE_DIM = 40
EDGE_FEATURE_DIM = 11


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def one_hot(value, allowable_set):
    encoding = [0] * len(allowable_set)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


# -----------------------------------------------------------------------------
# Atom Feature Encoder (verbatim from Cell 37A)
# -----------------------------------------------------------------------------

def atom_features(atom):

    features = []

    symbol = atom.GetSymbol()
    if symbol not in ATOM_SYMBOLS[:-1]:
        symbol = "Unknown"

    features.extend(one_hot(symbol, ATOM_SYMBOLS))
    features.extend(one_hot(atom.GetDegree(), DEGREES))
    features.append(atom.GetFormalCharge())
    features.extend(one_hot(atom.GetTotalNumHs(), TOTAL_H))
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend(one_hot(atom.GetChiralTag(), CHIRALITY))
    features.append(atom.GetAtomicNum() / 100.0)
    features.append(atom.GetMass() / 250.0)
    features.append(atom.GetImplicitValence())
    features.append(atom.GetExplicitValence())
    features.append(atom.GetNumRadicalElectrons())
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))

    return features


# -----------------------------------------------------------------------------
# Bond Feature Encoder (verbatim from Cell 37A)
# -----------------------------------------------------------------------------

def bond_features(bond):

    features = []

    features.extend(one_hot(bond.GetBondType(), BOND_TYPES))
    features.extend(one_hot(bond.GetStereo(), STEREO_TYPES))
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    features.append(float(bond.GetIsAromatic()))

    return features


# -----------------------------------------------------------------------------
# SMILES -> PyG Data (verbatim logic from Cell 37B, single-molecule version)
# -----------------------------------------------------------------------------

def smiles_to_graph(smiles: str) -> Data:
    """
    Convert a SMILES string into a torch_geometric.data.Data graph, using
    identical node/edge feature construction to training.

    Raises
    ------
    ValueError
        If the SMILES does not parse (propagated from smiles_to_mol).
    """

    mol = smiles_to_mol(smiles)

    # ---- Node Features ----
    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    # ---- Edge Features (bidirectional, matches training) ----
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index.append([start, end])
        edge_attr.append(bf)

        edge_index.append([end, start])
        edge_attr.append(bf)

    if len(edge_index) == 0:
        # Single-atom molecule edge case, matches training
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.smiles = smiles
    graph.num_nodes = x.size(0)

    return graph
