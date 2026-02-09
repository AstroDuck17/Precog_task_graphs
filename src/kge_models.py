"""
Knowledge Graph Embedding Models
================================

This module implements four popular KGE models for link prediction:

1. **TransE**: Translation-based model (h + r ≈ t)
2. **DistMult**: Bilinear diagonal model
3. **ComplEx**: Complex-valued embeddings for asymmetric relations
4. **RotatE**: Rotation in complex space

Theory:
-------
Knowledge Graph Embeddings learn low-dimensional representations of entities
and relations such that valid triples (h, r, t) have high scores.

Scoring Functions:
- TransE: -||h + r - t||
- DistMult: <h, r, t> = Σ h_i * r_i * t_i
- ComplEx: Re(<h, r, conj(t)>) where embeddings are complex
- RotatE: -||h ∘ r - t|| where r is unit complex (rotation)

Author: MetaFam Analysis Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class BaseKGEModel(nn.Module):
    """
    Base class for Knowledge Graph Embedding models.
    
    All KGE models share:
    - Entity embeddings
    - Relation embeddings
    - Score function (to be implemented by subclass)
    - Negative sampling support
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 1.0,
        regularization: float = 0.0
    ):
        """
        Initialize base KGE model.
        
        Parameters
        ----------
        num_entities : int
            Number of unique entities.
        num_relations : int
            Number of unique relations.
        embedding_dim : int
            Dimension of embeddings.
        margin : float
            Margin for margin-based loss.
        regularization : float
            L2 regularization coefficient.
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.regularization = regularization
        
    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute score for triples.
        
        Parameters
        ----------
        h : torch.Tensor
            Head entity embeddings [batch_size, dim].
        r : torch.Tensor
            Relation embeddings [batch_size, dim].
        t : torch.Tensor
            Tail entity embeddings [batch_size, dim].
            
        Returns
        -------
        torch.Tensor
            Scores [batch_size].
        """
        raise NotImplementedError
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass computing scores for triples.
        
        Parameters
        ----------
        heads : torch.Tensor
            Head entity indices [batch_size].
        relations : torch.Tensor
            Relation indices [batch_size].
        tails : torch.Tensor
            Tail entity indices [batch_size].
            
        Returns
        -------
        torch.Tensor
            Scores [batch_size].
        """
        raise NotImplementedError
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss."""
        return torch.tensor(0.0)


class TransE(BaseKGEModel):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data.
    
    Reference: Bordes et al., 2013
    
    Scoring Function:
    -----------------
    f(h, r, t) = -||h + r - t||_p
    
    The model interprets relations as translations in the embedding space.
    For a valid triple (h, r, t), we want h + r ≈ t.
    
    Theory:
    -------
    TransE works well for 1-to-1 relations but struggles with 1-to-N, N-to-1,
    and N-to-N relations because multiple entities would need to occupy the
    same point after translation.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        p_norm: int = 2,
        regularization: float = 0.0
    ):
        super().__init__(num_entities, num_relations, embedding_dim, margin, regularization)
        self.p_norm = p_norm
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=1
            )
    
    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """TransE score: -||h + r - t||_p"""
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        return self.score(h, r, t)
    
    def get_regularization_loss(self) -> torch.Tensor:
        if self.regularization == 0:
            return torch.tensor(0.0, device=self.entity_embeddings.weight.device)
        return self.regularization * (
            torch.norm(self.entity_embeddings.weight, p=2) ** 2 +
            torch.norm(self.relation_embeddings.weight, p=2) ** 2
        )


class DistMult(BaseKGEModel):
    """
    DistMult: Embedding Entities and Relations for Learning and Inference.
    
    Reference: Yang et al., 2014
    
    Scoring Function:
    -----------------
    f(h, r, t) = <h, r, t> = Σ_i h_i * r_i * t_i
    
    A bilinear model with diagonal relation matrices, making it efficient
    while still capturing interactions between dimensions.
    
    Theory:
    -------
    DistMult is symmetric: f(h, r, t) = f(t, r, h). This makes it unsuitable
    for asymmetric relations unless combined with other techniques.
    Good for symmetric relations like "similar_to" or "sibling_of".
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        regularization: float = 0.001
    ):
        super().__init__(num_entities, num_relations, embedding_dim, margin, regularization)
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """DistMult score: <h, r, t>"""
        return torch.sum(h * r * t, dim=-1)
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        return self.score(h, r, t)
    
    def get_regularization_loss(self) -> torch.Tensor:
        if self.regularization == 0:
            return torch.tensor(0.0, device=self.entity_embeddings.weight.device)
        return self.regularization * (
            torch.norm(self.entity_embeddings.weight, p=2) ** 2 +
            torch.norm(self.relation_embeddings.weight, p=2) ** 2
        )


class ComplEx(BaseKGEModel):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction.
    
    Reference: Trouillon et al., 2016
    
    Scoring Function:
    -----------------
    f(h, r, t) = Re(<h, r, conj(t)>)
               = Re(Σ_i h_i * r_i * conj(t_i))
    
    Uses complex-valued embeddings to model asymmetric relations.
    Each embedding has real and imaginary parts.
    
    Theory:
    -------
    ComplEx extends DistMult to handle asymmetric relations by using
    complex numbers. The conjugate operation on t breaks the symmetry:
    f(h, r, t) ≠ f(t, r, h) in general.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        regularization: float = 0.001
    ):
        super().__init__(num_entities, num_relations, embedding_dim, margin, regularization)
        
        # Complex embeddings: store real and imaginary parts separately
        # Each tensor has shape [num_items, embedding_dim]
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_im = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.xavier_uniform_(self.relation_re.weight)
        nn.init.xavier_uniform_(self.relation_im.weight)
    
    def score(
        self,
        h_re: torch.Tensor, h_im: torch.Tensor,
        r_re: torch.Tensor, r_im: torch.Tensor,
        t_re: torch.Tensor, t_im: torch.Tensor
    ) -> torch.Tensor:
        """
        ComplEx score: Re(<h, r, conj(t)>).
        
        Using complex multiplication:
        (a + bi)(c + di)(e - fi) = ...
        
        Simplified formula:
        Re(<h, r, conj(t)>) = <h_re, r_re, t_re> + <h_re, r_im, t_im>
                            + <h_im, r_re, t_im> - <h_im, r_im, t_re>
        """
        score = (
            torch.sum(h_re * r_re * t_re, dim=-1) +
            torch.sum(h_re * r_im * t_im, dim=-1) +
            torch.sum(h_im * r_re * t_im, dim=-1) -
            torch.sum(h_im * r_im * t_re, dim=-1)
        )
        return score
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        r_re = self.relation_re(relations)
        r_im = self.relation_im(relations)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)
        
        return self.score(h_re, h_im, r_re, r_im, t_re, t_im)
    
    def get_regularization_loss(self) -> torch.Tensor:
        if self.regularization == 0:
            return torch.tensor(0.0, device=self.entity_re.weight.device)
        return self.regularization * (
            torch.norm(self.entity_re.weight, p=2) ** 2 +
            torch.norm(self.entity_im.weight, p=2) ** 2 +
            torch.norm(self.relation_re.weight, p=2) ** 2 +
            torch.norm(self.relation_im.weight, p=2) ** 2
        )


class RotatE(BaseKGEModel):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.
    
    Reference: Sun et al., 2019
    
    Scoring Function:
    -----------------
    f(h, r, t) = -||h ∘ r - t||
    
    where ∘ is element-wise complex multiplication (Hadamard product),
    and r is constrained to be a unit complex number (|r| = 1).
    
    Theory:
    -------
    RotatE interprets relations as rotations in complex space.
    For r = e^{iθ}, the multiplication h * r rotates h by angle θ.
    
    This allows modeling:
    - Symmetric relations: θ = 0 or π
    - Asymmetric relations: θ ≠ 0, π
    - Inversion: r1 * r2 = 1 (rotations cancel)
    - Composition: r3 = r1 * r2 (rotations add)
    
    Particularly good for hierarchical relations and inverse patterns.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 6.0,
        regularization: float = 0.0
    ):
        super().__init__(num_entities, num_relations, embedding_dim, margin, regularization)
        
        # Entity embeddings: complex valued
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings: phases (will be converted to unit complex)
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.uniform_(self.relation_phase.weight, a=-np.pi, b=np.pi)
        
        # Embedding range for initialization
        self.embedding_range = (margin + 2.0) / embedding_dim
    
    def score(
        self,
        h_re: torch.Tensor, h_im: torch.Tensor,
        r_re: torch.Tensor, r_im: torch.Tensor,
        t_re: torch.Tensor, t_im: torch.Tensor
    ) -> torch.Tensor:
        """
        RotatE score: -||h ∘ r - t||.
        
        Complex multiplication h * r:
        (h_re + h_im*i) * (r_re + r_im*i) = (h_re*r_re - h_im*r_im) + (h_re*r_im + h_im*r_re)*i
        """
        # h * r (complex multiplication)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        
        # Distance to t
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        
        # L2 norm of complex difference
        score = -torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-12).sum(dim=-1)
        return score
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)
        
        # Convert phase to unit complex: e^{iθ} = cos(θ) + i*sin(θ)
        phase = self.relation_phase(relations)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)
        
        return self.score(h_re, h_im, r_re, r_im, t_re, t_im)
    
    def get_regularization_loss(self) -> torch.Tensor:
        if self.regularization == 0:
            return torch.tensor(0.0, device=self.entity_re.weight.device)
        return self.regularization * (
            torch.norm(self.entity_re.weight, p=2) ** 2 +
            torch.norm(self.entity_im.weight, p=2) ** 2
        )


# ===========================================================================
# MODEL FACTORY
# ===========================================================================

def create_kge_model(
    model_name: str,
    num_entities: int,
    num_relations: int,
    embedding_dim: int = 100,
    **kwargs
) -> BaseKGEModel:
    """
    Factory function to create KGE models.
    
    Parameters
    ----------
    model_name : str
        Model name: 'TransE', 'DistMult', 'ComplEx', 'RotatE'.
    num_entities : int
        Number of entities.
    num_relations : int
        Number of relations.
    embedding_dim : int
        Embedding dimension.
    **kwargs
        Additional model-specific parameters.
        
    Returns
    -------
    BaseKGEModel
        Initialized model.
    """
    models = {
        'TransE': TransE,
        'DistMult': DistMult,
        'ComplEx': ComplEx,
        'RotatE': RotatE
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        **kwargs
    )


# ===========================================================================
# LOSS FUNCTIONS
# ===========================================================================

class MarginRankingLoss(nn.Module):
    """
    Margin-based ranking loss for KGE training.
    
    L = max(0, margin - score_pos + score_neg)
    
    Encourages positive triples to have higher scores than negative triples
    by at least the margin.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.
        
        Parameters
        ----------
        pos_scores : torch.Tensor
            Scores for positive triples.
        neg_scores : torch.Tensor
            Scores for negative triples.
            
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return F.relu(self.margin - pos_scores + neg_scores).mean()


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary cross-entropy loss for KGE training.
    
    L = -y*log(σ(score)) - (1-y)*log(1-σ(score))
    
    Treats link prediction as binary classification.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BCE loss.
        
        Parameters
        ----------
        pos_scores : torch.Tensor
            Scores for positive triples (label=1).
        neg_scores : torch.Tensor
            Scores for negative triples (label=0).
            
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        pos_loss = F.logsigmoid(pos_scores).mean()
        neg_loss = F.logsigmoid(-neg_scores).mean()
        return -(pos_loss + neg_loss) / 2
