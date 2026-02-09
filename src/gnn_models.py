"""
Graph Neural Network Models for Link Prediction
================================================

This module implements GNN-based models for knowledge graph link prediction:

1. **RGCN**: Relational Graph Convolutional Network (encoder)
2. **RGCN + DistMult**: RGCN encoder with DistMult decoder
3. **RGCN + RotatE**: RGCN encoder with RotatE decoder (complex-valued)

Theory:
-------
Unlike KGE models that learn static embeddings, GNN-based approaches:
1. Learn embeddings conditioned on the local graph structure
2. Aggregate information from neighbors
3. Can generalize better to unseen entities (inductive setting)

Architecture:
- Encoder (RGCN): Aggregates neighbor information with relation-specific weights
- Decoder (DistMult/RotatE): Scores triples using learned node representations

Author: MetaFam Analysis Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Layer.
    
    Reference: Schlichtkrull et al., 2018
    "Modeling Relational Data with Graph Convolutional Networks"
    
    For each node v, aggregates messages from neighbors:
    
    h_v^{l+1} = σ(Σ_r Σ_{u∈N_r(v)} (1/c_{v,r}) W_r^l h_u^l + W_0^l h_v^l)
    
    where:
    - N_r(v): neighbors of v under relation r
    - c_{v,r}: normalization constant
    - W_r: relation-specific weight matrix
    - W_0: self-loop weight matrix
    
    Basis Decomposition:
    --------------------
    To handle many relations efficiently, we decompose W_r as:
    W_r = Σ_b a_{rb} V_b
    
    This reduces parameters from O(R × d × d) to O(B × d × d + R × B).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        num_bases: int = None,
        bias: bool = True,
        activation: nn.Module = None,
        dropout: float = 0.0,
        self_loop: bool = True
    ):
        """
        Initialize RGCN layer.
        
        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension.
        num_relations : int
            Number of relation types.
        num_bases : int, optional
            Number of basis matrices for decomposition.
            If None, use full weight matrices per relation.
        bias : bool
            Whether to use bias.
        activation : nn.Module, optional
            Activation function.
        dropout : float
            Dropout rate.
        self_loop : bool
            Whether to add self-loops.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases else num_relations
        self.self_loop = self_loop
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Basis matrices
        if self.num_bases < num_relations:
            # Use basis decomposition
            self.basis = nn.Parameter(
                torch.Tensor(self.num_bases, in_features, out_features)
            )
            self.att = nn.Parameter(
                torch.Tensor(num_relations, self.num_bases)
            )
        else:
            # Full weight matrices per relation
            self.weight = nn.Parameter(
                torch.Tensor(num_relations, in_features, out_features)
            )
        
        # Self-loop weight
        if self_loop:
            self.self_weight = nn.Parameter(
                torch.Tensor(in_features, out_features)
            )
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.num_bases < self.num_relations:
            nn.init.xavier_uniform_(self.basis)
            nn.init.xavier_uniform_(self.att)
        else:
            nn.init.xavier_uniform_(self.weight)
        
        if self.self_loop:
            nn.init.xavier_uniform_(self.self_weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int = None
    ) -> torch.Tensor:
        """
        Forward pass of RGCN layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, in_features].
        edge_index : torch.Tensor
            Edge indices [2, num_edges].
        edge_type : torch.Tensor
            Edge types [num_edges].
        num_nodes : int, optional
            Number of nodes (for handling isolated nodes).
            
        Returns
        -------
        torch.Tensor
            Updated node features [num_nodes, out_features].
        """
        if num_nodes is None:
            num_nodes = x.size(0)
        
        # Get relation-specific weights
        if self.num_bases < self.num_relations:
            # Basis decomposition: W_r = Σ_b a_{rb} V_b
            weight = torch.einsum('rb,bij->rij', self.att, self.basis)
        else:
            weight = self.weight
        
        # Initialize output
        out = torch.zeros(num_nodes, self.out_features, device=x.device)
        
        # Aggregate messages for each relation type
        src, dst = edge_index
        
        for rel in range(self.num_relations):
            # Get edges of this relation type
            mask = edge_type == rel
            if not mask.any():
                continue
            
            rel_src = src[mask]
            rel_dst = dst[mask]
            
            # Compute normalization
            # c_{v,r} = degree of v under relation r
            deg = torch.bincount(rel_dst, minlength=num_nodes).float()
            deg = deg.clamp(min=1)  # Avoid division by zero
            norm = 1.0 / deg[rel_dst]
            
            # Message: h_u * W_r
            msg = torch.mm(x[rel_src], weight[rel])
            
            # Aggregate with normalization
            msg = msg * norm.unsqueeze(-1)
            out.index_add_(0, rel_dst, msg)
        
        # Self-loop
        if self.self_loop:
            out = out + torch.mm(x, self.self_weight)
        
        # Bias
        if self.bias is not None:
            out = out + self.bias
        
        # Activation
        if self.activation is not None:
            out = self.activation(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network.
    
    Multi-layer RGCN for learning entity representations that are
    conditioned on the local graph structure.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        num_bases: int = None,
        dropout: float = 0.2
    ):
        """
        Initialize RGCN.
        
        Parameters
        ----------
        num_entities : int
            Number of entities.
        num_relations : int
            Number of relations.
        hidden_dim : int
            Hidden dimension.
        num_layers : int
            Number of RGCN layers.
        num_bases : int, optional
            Number of basis matrices for weight decomposition.
        dropout : float
            Dropout rate.
        """
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # Initial entity embeddings (learnable)
        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        # RGCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                RGCNLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    activation=nn.ReLU() if i < num_layers - 1 else None,
                    dropout=dropout if i < num_layers - 1 else 0.0
                )
            )
    
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entity representations.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices [2, num_edges].
        edge_type : torch.Tensor
            Edge types [num_edges].
            
        Returns
        -------
        torch.Tensor
            Entity representations [num_entities, hidden_dim].
        """
        x = self.entity_embeddings.weight
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, self.num_entities)
        
        return x


class DistMultDecoder(nn.Module):
    """
    DistMult scoring function for decoded entity embeddings.
    
    f(h, r, t) = <h, r, t> = Σ_i h_i * r_i * t_i
    """
    
    def __init__(self, num_relations: int, embedding_dim: int):
        super().__init__()
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(
        self,
        entity_emb: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Score triples using DistMult.
        
        Parameters
        ----------
        entity_emb : torch.Tensor
            Entity embeddings from encoder [num_entities, dim].
        heads : torch.Tensor
            Head indices [batch_size].
        relations : torch.Tensor
            Relation indices [batch_size].
        tails : torch.Tensor
            Tail indices [batch_size].
            
        Returns
        -------
        torch.Tensor
            Scores [batch_size].
        """
        h = entity_emb[heads]
        r = self.relation_embeddings(relations)
        t = entity_emb[tails]
        
        return torch.sum(h * r * t, dim=-1)


class RotatEDecoder(nn.Module):
    """
    RotatE scoring function for decoded entity embeddings.
    
    For complex embeddings: f(h, r, t) = -||h ∘ r - t||
    
    Requires entity embeddings to be interpreted as complex numbers.
    """
    
    def __init__(self, num_relations: int, embedding_dim: int):
        super().__init__()
        # embedding_dim should be even (real + imaginary)
        assert embedding_dim % 2 == 0, "embedding_dim must be even for RotatE"
        self.half_dim = embedding_dim // 2
        
        # Relation phases
        self.relation_phase = nn.Embedding(num_relations, self.half_dim)
        nn.init.uniform_(self.relation_phase.weight, a=-np.pi, b=np.pi)
    
    def forward(
        self,
        entity_emb: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Score triples using RotatE.
        
        Entity embeddings are split: first half = real, second half = imaginary.
        """
        # Split entity embeddings into real and imaginary
        h_re = entity_emb[heads, :self.half_dim]
        h_im = entity_emb[heads, self.half_dim:]
        t_re = entity_emb[tails, :self.half_dim]
        t_im = entity_emb[tails, self.half_dim:]
        
        # Relation as unit complex number
        phase = self.relation_phase(relations)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)
        
        # Complex multiplication: h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        
        # Distance to t
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        
        # Negative L2 norm
        score = -torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-12).sum(dim=-1)
        return score


class RGCNLinkPredictor(nn.Module):
    """
    RGCN-based Link Prediction Model.
    
    Combines:
    - RGCN encoder for learning structure-aware entity representations
    - Decoder (DistMult or RotatE) for scoring triples
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        num_bases: int = None,
        dropout: float = 0.2,
        decoder_type: str = 'DistMult'
    ):
        """
        Initialize RGCN Link Predictor.
        
        Parameters
        ----------
        num_entities : int
            Number of entities.
        num_relations : int
            Number of relations.
        hidden_dim : int
            Hidden dimension.
        num_layers : int
            Number of RGCN layers.
        num_bases : int, optional
            Number of basis matrices.
        dropout : float
            Dropout rate.
        decoder_type : str
            Decoder type: 'DistMult' or 'RotatE'.
        """
        super().__init__()
        
        self.encoder = RGCN(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout
        )
        
        if decoder_type == 'DistMult':
            self.decoder = DistMultDecoder(num_relations, hidden_dim)
        elif decoder_type == 'RotatE':
            self.decoder = RotatEDecoder(num_relations, hidden_dim)
        else:
            raise ValueError(f"Unknown decoder: {decoder_type}")
        
        self.decoder_type = decoder_type
    
    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode entities using graph structure.
        
        Returns
        -------
        torch.Tensor
            Entity embeddings [num_entities, hidden_dim].
        """
        return self.encoder(edge_index, edge_type)
    
    def decode(
        self,
        entity_emb: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Score triples using decoder.
        
        Returns
        -------
        torch.Tensor
            Scores [batch_size].
        """
        return self.decoder(entity_emb, heads, relations, tails)
    
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass: encode then decode.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Graph edge indices [2, num_edges].
        edge_type : torch.Tensor
            Graph edge types [num_edges].
        heads : torch.Tensor
            Query head indices [batch_size].
        relations : torch.Tensor
            Query relation indices [batch_size].
        tails : torch.Tensor
            Query tail indices [batch_size].
            
        Returns
        -------
        torch.Tensor
            Triple scores [batch_size].
        """
        entity_emb = self.encode(edge_index, edge_type)
        return self.decode(entity_emb, heads, relations, tails)


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def build_graph_tensors(
    triples: List[Tuple[str, str, str]],
    entity_vocab: Dict[str, int],
    relation_vocab: Dict[str, int],
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index and edge_type tensors from triples.
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        List of (head, relation, tail) triples.
    entity_vocab : Dict[str, int]
        Entity to index mapping.
    relation_vocab : Dict[str, int]
        Relation to index mapping.
    device : torch.device, optional
        Target device.
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (edge_index [2, num_edges], edge_type [num_edges])
    """
    src_indices = []
    dst_indices = []
    edge_types = []
    
    for h, r, t in triples:
        if h in entity_vocab and t in entity_vocab and r in relation_vocab:
            src_indices.append(entity_vocab[h])
            dst_indices.append(entity_vocab[t])
            edge_types.append(relation_vocab[r])
    
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    if device is not None:
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
    
    return edge_index, edge_type


def create_gnn_model(
    model_name: str,
    num_entities: int,
    num_relations: int,
    hidden_dim: int = 100,
    **kwargs
) -> RGCNLinkPredictor:
    """
    Factory function to create GNN models.
    
    Parameters
    ----------
    model_name : str
        Model name: 'RGCN_DistMult' or 'RGCN_RotatE'.
    num_entities : int
        Number of entities.
    num_relations : int
        Number of relations.
    hidden_dim : int
        Hidden dimension.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    RGCNLinkPredictor
        Initialized model.
    """
    if model_name == 'RGCN_DistMult':
        decoder_type = 'DistMult'
    elif model_name == 'RGCN_RotatE':
        decoder_type = 'RotatE'
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: RGCN_DistMult, RGCN_RotatE")
    
    return RGCNLinkPredictor(
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        decoder_type=decoder_type,
        **kwargs
    )
