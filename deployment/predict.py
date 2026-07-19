import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Check PyG availability FIRST
# ---------------------------------------------------------------------------

try:
    from torch_geometric.nn import GINEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _PYG_OK = True
except ImportError:
    _PYG_OK = False
    Data  = None
    Batch = None


# ---------------------------------------------------------------------------
# DNN Architecture — matches saved state_dict exactly
# keys: network.0, network.1 ... self.network, input=851
# 512 -> 256 -> 128 -> 64 -> 1 -> Sigmoid
# ---------------------------------------------------------------------------

class DeepToxNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


# ---------------------------------------------------------------------------
# GNN Architecture
# ---------------------------------------------------------------------------

if _PYG_OK:
    class GINEModel(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden=128, n_layers=4, dropout=0.2):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns   = nn.ModuleList()
            in_d = node_dim
            for _ in range(n_layers):
                mlp = nn.Sequential(
                    nn.Linear(in_d, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden)
                )
                self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
                self.bns.append(nn.BatchNorm1d(hidden))
                in_d = hidden
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1)
            )

        def forward(self, data):
            x, ei, ea, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            for conv, bn in zip(self.convs, self.bns):
                x = bn(torch.relu(conv(x, ei, ea)))
                x = self.dropout(x)
            x = global_mean_pool(x, batch)
            return self.head(x).squeeze(1)
else:
    GINEModel = None


# ---------------------------------------------------------------------------
# Atom / Bond features
# ---------------------------------------------------------------------------

def _one_hot(val, choices):
    enc = [0] * len(choices)
    if val in choices:
        enc[choices.index(val)] = 1
    else:
        enc[-1] = 1
    return enc

def _atom_features(atom):
    return (
        _one_hot(atom.GetAtomicNum(),
                 [1,5,6,7,8,9,14,15,16,17,35,53,0])
        + _one_hot(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10])
        + _one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])
        + _one_hot(atom.GetImplicitValence(), [0,1,2,3,4,5,6])
        + [int(atom.GetIsAromatic())]
        + _one_hot(str(atom.GetHybridization()),
                   ["SP","SP2","SP3","SP3D","SP3D2","OTHER"])
    )

def _bond_features(bond):
    from rdkit.Chem import rdchem
    bt = bond.GetBondType()
    return [
        int(bt == rdchem.BondType.SINGLE),
        int(bt == rdchem.BondType.DOUBLE),
        int(bt == rdchem.BondType.TRIPLE),
        int(bt == rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def _smiles_to_graph(smiles):
    if not _PYG_OK:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor(
        [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )
    src, dst, ea = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src += [i, j]; dst += [j, i]; ea += [bf, bf]
    if not src:
        return None
    return Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.float)
    )


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048

def _smiles_to_feature_row(smiles, feature_columns, descriptor_medians=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp     = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    morgan = list(fp)
    maccs  = list(MACCSkeys.GenMACCSKeys(mol))

    desc_names  = [name for name, _ in Descriptors.descList]
    desc_values = []
    for _, fn in Descriptors.descList:
        try:
            v = fn(mol)
            desc_values.append(float(v) if v is not None else np.nan)
        except Exception:
            desc_values.append(np.nan)
    desc_dict = dict(zip(desc_names, desc_values))

    row = {}
    for col in feature_columns:
        if col.startswith("FP_"):
            idx = int(col.replace("FP_", ""))
            row[col] = morgan[idx] if idx < len(morgan) else 0
        elif col.startswith("MACCS_"):
            idx = int(col.replace("MACCS_", ""))
            row[col] = maccs[idx] if idx < len(maccs) else 0
        elif col in desc_dict:
            row[col] = desc_dict[col]
        else:
            row[col] = 0.0

    X = pd.DataFrame([row], columns=feature_columns)

    if descriptor_medians is not None:
        for c in X.columns:
            if X[c].isna().any():
                try:
                    X[c] = descriptor_medians[c]
                except Exception:
                    X[c] = 0.0
    else:
        X = X.fillna(0.0)

    X.replace([np.inf, -np.inf], 0.0, inplace=True)
    return X


# ---------------------------------------------------------------------------
# ImmunotoxicityPredictor
# ---------------------------------------------------------------------------

class ImmunotoxicityPredictor:

    def __init__(self, deployment_dir="deployment"):
        self.deployment_dir = deployment_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_preprocessing()
        self._load_models()
        print(f"Predictor ready | device={self.device} | "
              f"endpoints={len(self.target_columns)} | GNN={_PYG_OK}")

    def _load_preprocessing(self):
        preproc = os.path.join(self.deployment_dir, "preprocessing")

        self.feature_columns = joblib.load(
            os.path.join(preproc, "feature_columns.pkl"))

        with open(os.path.join(preproc, "preprocessing_metadata.json")) as f:
            self.metadata = json.load(f)
        self.target_columns = self.metadata.get("targets", [])

        medians_path = os.path.join(preproc, "descriptor_medians.pkl")
        self.descriptor_medians = (
            joblib.load(medians_path) if os.path.exists(medians_path) else None
        )

        self.scaler = None
        if os.path.isdir(preproc):
            for fname in os.listdir(preproc):
                if "scaler" in fname.lower() and fname.endswith(".pkl"):
                    obj = joblib.load(os.path.join(preproc, fname))
                    if hasattr(obj, "transform"):
                        self.scaler = obj
                        break

        thresh_pkl  = os.path.join(self.deployment_dir, "config", "gnn_thresholds.pkl")
        thresh_json = os.path.join(self.deployment_dir, "config", "gnn_thresholds.json")
        if os.path.exists(thresh_pkl):
            self.gnn_thresholds = joblib.load(thresh_pkl)
        elif os.path.exists(thresh_json):
            with open(thresh_json) as f:
                self.gnn_thresholds = json.load(f)
        else:
            self.gnn_thresholds = {t: 0.5 for t in self.target_columns}

        weights_path = os.path.join(
            self.deployment_dir, "config", "ensemble_weights.json")
        self.ensemble_weights = None
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                self.ensemble_weights = json.load(f)

    def _load_models(self):
        mdir = os.path.join(self.deployment_dir, "models")
        self.rf_models  = self._load_sklearn(os.path.join(mdir, "random_forest"))
        self.xgb_models = self._load_sklearn(os.path.join(mdir, "xgboost"))
        self.cat_models = self._load_sklearn(os.path.join(mdir, "catboost"))
        self.dnn_models = self._load_dnn(os.path.join(mdir, "dnn"))
        gine_dir = os.path.join(mdir, "gine")
        if _PYG_OK and os.path.isdir(gine_dir) and GINEModel is not None:
            self.gnn_models = self._load_gnn(gine_dir)
        else:
            self.gnn_models = {}
            if not _PYG_OK:
                print("GNN skipped — torch_geometric not available")

    def _load_sklearn(self, folder):
        models = {}
        if not os.path.isdir(folder):
            return models
        for fname in os.listdir(folder):
            if fname.endswith(".pkl"):
                try:
                    models[fname.replace(".pkl", "")] = joblib.load(
                        os.path.join(folder, fname))
                except Exception as e:
                    print(f"  Warning: could not load {fname}: {e}")
        return models

    def _load_dnn(self, folder):
        models = {}
        if not os.path.isdir(folder):
            return models
        # input dim read from first layer of first .pt file
        pt_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        if not pt_files:
            return models
        # infer input_dim from saved weights
        sample_sd = torch.load(
            os.path.join(folder, pt_files[0]), map_location="cpu")
        input_dim = sample_sd["network.0.weight"].shape[1]
        for fname in pt_files:
            try:
                m = DeepToxNet(input_dim).to(self.device)
                m.load_state_dict(torch.load(
                    os.path.join(folder, fname),
                    map_location=self.device))
                m.eval()
                models[fname.replace(".pt", "")] = m
            except Exception as e:
                print(f"  Warning: could not load DNN {fname}: {e}")
        return models

    def _load_gnn(self, folder):
        models = {}
        if not _PYG_OK or GINEModel is None:
            return models
        sample = _smiles_to_graph("CCO")
        if sample is None:
            print("  Warning: could not build test graph for GNN")
            return models
        nd = sample.x.shape[1]
        ed = sample.edge_attr.shape[1]
        for fname in os.listdir(folder):
            if fname.endswith(".pt"):
                try:
                    m = GINEModel(node_dim=nd, edge_dim=ed).to(self.device)
                    m.load_state_dict(torch.load(
                        os.path.join(folder, fname),
                        map_location=self.device))
                    m.eval()
                    models[fname.replace(".pt", "")] = m
                except Exception as e:
                    print(f"  Warning: could not load GNN {fname}: {e}")
        return models

    def _featurize(self, smiles):
        X_df = _smiles_to_feature_row(
            smiles, self.feature_columns, self.descriptor_medians)
        if X_df is None:
            return None, None
        if self.scaler is not None:
            X_arr = self.scaler.transform(X_df.values).astype(np.float32)
        else:
            X_arr = X_df.values.astype(np.float32)
        return X_df, X_arr

    def predict_smiles(self, smiles):
        if not isinstance(smiles, str) or Chem.MolFromSmiles(smiles) is None:
            return {"smiles": smiles, "error": "Invalid SMILES", "endpoints": {}}

        X_df, X_arr = self._featurize(smiles)
        if X_arr is None:
            return {"smiles": smiles, "error": "Feature extraction failed",
                    "endpoints": {}}

        graph = _smiles_to_graph(smiles) if _PYG_OK else None

        endpoint_results = {}

        for target in self.target_columns:
            probs = {}

            if target in self.rf_models:
                try:
                    probs["RF"] = float(
                        self.rf_models[target].predict_proba(X_arr)[0, 1])
                except Exception:
                    pass

            if target in self.xgb_models:
                try:
                    probs["XGB"] = float(
                        self.xgb_models[target].predict_proba(X_arr)[0, 1])
                except Exception:
                    pass

            if target in self.cat_models:
                try:
                    probs["CAT"] = float(
                        self.cat_models[target].predict_proba(X_arr)[0, 1])
                except Exception:
                    pass

            if target in self.dnn_models:
                try:
                    t_in = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        probs["DNN"] = float(
                            self.dnn_models[target](t_in).cpu().item())
                except Exception:
                    pass

            if graph is not None and target in self.gnn_models:
                try:
                    batch = Batch.from_data_list([graph]).to(self.device)
                    with torch.no_grad():
                        probs["GNN"] = float(torch.sigmoid(
                            self.gnn_models[target](batch)).cpu().item())
                except Exception:
                    pass

            if not probs:
                continue

            if self.ensemble_weights and target in self.ensemble_weights:
                w = self.ensemble_weights[target]
                total_w = sum(w.get(k, 0) for k in probs)
                ens_prob = (
                    sum(probs[k] * w.get(k, 0) for k in probs) / total_w
                    if total_w > 0
                    else float(np.mean(list(probs.values())))
                )
            else:
                ens_prob = float(np.mean(list(probs.values())))

            threshold = float(self.gnn_thresholds.get(target, 0.5))
            ens_pred  = int(ens_prob >= threshold)
            agreement = sum(
                1 for p in probs.values()
                if (p >= threshold) == bool(ens_pred)
            ) / len(probs)

            endpoint_results[target] = {
                "ensemble_probability" : round(ens_prob, 4),
                "ensemble_prediction"  : ens_pred,
                "model_agreement"      : round(agreement, 4),
                "individual_probs"     : {k: round(v, 4) for k, v in probs.items()},
                "threshold_used"       : round(threshold, 4),
            }

        return {"smiles": smiles, "error": None, "endpoints": endpoint_results}

    def predict_batch(self, smiles_list):
        rows = []
        for smi in smiles_list:
            res = self.predict_smiles(smi)
            if res["error"] is not None:
                rows.append({"smiles": smi, "error": res["error"]})
                continue
            row = {"smiles": smi, "error": None}
            for target, vals in res["endpoints"].items():
                row[f"{target}_prob"] = vals["ensemble_probability"]
                row[f"{target}_pred"] = vals["ensemble_prediction"]
            rows.append(row)
        return pd.DataFrame(rows)