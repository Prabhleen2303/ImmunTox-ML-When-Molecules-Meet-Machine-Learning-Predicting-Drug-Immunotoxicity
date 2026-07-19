"""
==============================================================================
models.py

Research-Grade Immunotoxicity Prediction Pipeline

Author : Prabhleen Kaur Saini
Project: IISER Mohali BS-MS Research Project

Purpose
-------
Cell 50 of the training notebook exports the DNN and GINE models using
`torch.save(model.state_dict(), ...)`, not the full model object. That means
deployment needs the exact architecture classes to reconstruct the models
before loading the weights. This module defines those two classes verbatim
against the training notebook (Cell 29 : Deep Neural Network Architecture,
Cell 39A : Research-Grade GINE Model) so the state_dicts load without a
shape/key mismatch.

Do not change these class definitions without retraining - the saved
state_dicts are tied to this exact architecture.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool


# =============================================================================
# Deep Neural Network (Cell 29)
# =============================================================================

class DeepToxNet(nn.Module):
    """
    Feedforward network used for the tabular (descriptor + fingerprint)
    representation. Architecture must match training exactly:
    input_dim -> 512 -> 256 -> 128 -> 1
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.network = nn.Sequential(

            # Block 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.30),

            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),

            # Block 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),

            # Output
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# GINE Graph Neural Network (Cell 39A)
# =============================================================================

def _build_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Helper MLP used inside each GINEConv layer, as in training."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim)
    )


class GINEModel(nn.Module):
    """
    3-layer GINE model with edge features, BatchNorm, GELU, and a
    2-layer MLP classification head. Must match training exactly:
    node_dim=40, edge_dim=11, hidden_dim=128, dropout=0.30.
    """

    HIDDEN_DIM = 128
    DROPOUT = 0.30

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT
    ):
        super().__init__()

        self.dropout = dropout

        self.input_projection = nn.Linear(node_dim, hidden_dim)

        self.conv1 = GINEConv(_build_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = GINEConv(_build_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = GINEConv(_build_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
        self.bn3 = BatchNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        data : torch_geometric.data.Data or Batch
            Must expose .x, .edge_index, .edge_attr, .batch
        """

        x = self.input_projection(data.x)

        # Block 1
        residual = x
        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = self.bn1(x)
        x = F.gelu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Block 2
        residual = x
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = self.bn2(x)
        x = F.gelu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Block 3 (no dropout, matches training)
        residual = x
        x = self.conv3(x, data.edge_index, data.edge_attr)
        x = self.bn3(x)
        x = F.gelu(x)
        x = x + residual

        # Graph-level pooling + prediction
        x = global_mean_pool(x, data.batch)
        logits = self.classifier(x)

        return logits.view(-1)
