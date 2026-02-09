"""
Training and Evaluation Module for Link Prediction
===================================================

This module provides:
1. Negative sampling strategies
2. Training loops for KGE and GNN models
3. Evaluation metrics (MRR, Hits@1, Hits@10)
4. Utility functions for model management

Key Design Decisions:
--------------------
- Filtered evaluation: When ranking candidates, filter out other true triples
- Full batch evaluation: Score all entities as candidates (not sampled)
- Negative sampling: Corrupt head or tail uniformly

Author: MetaFam Analysis Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time
import json
import os


# ===========================================================================
# DATASET CLASSES
# ===========================================================================

class TripleDataset(Dataset):
    """Dataset for KGE model training."""
    
    def __init__(
        self,
        triples: List[Tuple[str, str, str]],
        entity_vocab: Dict[str, int],
        relation_vocab: Dict[str, int],
        num_neg_samples: int = 1
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        triples : List[Tuple[str, str, str]]
            List of (head, relation, tail) triples.
        entity_vocab : Dict[str, int]
            Entity to index mapping.
        relation_vocab : Dict[str, int]
            Relation to index mapping.
        num_neg_samples : int
            Number of negative samples per positive.
        """
        self.triples = []
        self.num_neg_samples = num_neg_samples
        self.num_entities = len(entity_vocab)
        
        # Convert to indices, skip triples with unknown entities/relations
        for h, r, t in triples:
            if h in entity_vocab and t in entity_vocab and r in relation_vocab:
                self.triples.append((
                    entity_vocab[h],
                    relation_vocab[r],
                    entity_vocab[t]
                ))
        
        # Build set of known triples for filtering
        self.known_triples = set(self.triples)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """Return positive triple and negative samples."""
        h, r, t = self.triples[idx]
        
        # Generate negative samples
        neg_heads = []
        neg_tails = []
        
        for _ in range(self.num_neg_samples):
            # Randomly corrupt head or tail
            if np.random.random() < 0.5:
                # Corrupt head
                neg_h = np.random.randint(0, self.num_entities)
                while (neg_h, r, t) in self.known_triples:
                    neg_h = np.random.randint(0, self.num_entities)
                neg_heads.append(neg_h)
                neg_tails.append(t)
            else:
                # Corrupt tail
                neg_t = np.random.randint(0, self.num_entities)
                while (h, r, neg_t) in self.known_triples:
                    neg_t = np.random.randint(0, self.num_entities)
                neg_heads.append(h)
                neg_tails.append(neg_t)
        
        return {
            'head': h,
            'relation': r,
            'tail': t,
            'neg_heads': neg_heads,
            'neg_tails': neg_tails
        }


def collate_triples(batch):
    """Collate function for TripleDataset."""
    heads = torch.tensor([item['head'] for item in batch], dtype=torch.long)
    relations = torch.tensor([item['relation'] for item in batch], dtype=torch.long)
    tails = torch.tensor([item['tail'] for item in batch], dtype=torch.long)
    
    neg_heads = torch.tensor([item['neg_heads'] for item in batch], dtype=torch.long)
    neg_tails = torch.tensor([item['neg_tails'] for item in batch], dtype=torch.long)
    
    return {
        'heads': heads,
        'relations': relations,
        'tails': tails,
        'neg_heads': neg_heads,
        'neg_tails': neg_tails
    }


# ===========================================================================
# EVALUATION METRICS
# ===========================================================================

def compute_metrics(
    model,
    triples: List[Tuple[int, int, int]],
    all_triples_set: Set[Tuple[int, int, int]],
    num_entities: int,
    device: torch.device,
    batch_size: int = 100,
    # For GNN models
    edge_index: torch.Tensor = None,
    edge_type: torch.Tensor = None,
    is_gnn: bool = False
) -> Dict[str, float]:
    """
    Compute ranking metrics with filtered evaluation.
    
    Filtered Evaluation:
    -------------------
    When ranking candidate entities for a query (h, r, ?), we filter out
    tails t' where (h, r, t') is a known true triple (except the query itself).
    This avoids penalizing the model for ranking other true answers highly.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    triples : List[Tuple[int, int, int]]
        Test triples as (head_idx, rel_idx, tail_idx).
    all_triples_set : Set[Tuple[int, int, int]]
        All known true triples (train + valid + test) for filtering.
    num_entities : int
        Total number of entities.
    device : torch.device
        Device for computation.
    batch_size : int
        Batch size for evaluation.
    edge_index : torch.Tensor, optional
        Graph edges for GNN models.
    edge_type : torch.Tensor, optional
        Edge types for GNN models.
    is_gnn : bool
        Whether model is GNN-based.
        
    Returns
    -------
    Dict[str, float]
        Metrics: mrr, hits_at_1, hits_at_10.
    """
    model.eval()
    
    ranks = []
    
    # Pre-build filter indices for fast lookup
    # Instead of iterating all entities per triple, only look up known triples
    hr_to_tails = defaultdict(set)  # (head, rel) -> set of known tail indices
    rt_to_heads = defaultdict(set)  # (rel, tail) -> set of known head indices
    for h_i, r_i, t_i in all_triples_set:
        hr_to_tails[(h_i, r_i)].add(t_i)
        rt_to_heads[(r_i, t_i)].add(h_i)
    
    # Pre-compute entity embeddings for GNN models
    if is_gnn:
        with torch.no_grad():
            entity_emb = model.encode(edge_index, edge_type)
    
    # Pre-create reusable tensors
    all_entities = torch.arange(num_entities, dtype=torch.long, device=device)
    
    num_triples = len(triples)
    
    with torch.no_grad():
        for idx, (h, r, t) in enumerate(triples):
            if (idx + 1) % 500 == 0 or idx == num_triples - 1:
                print(f"  Evaluating: {idx+1}/{num_triples} triples...", end='\r')
            
            # ===== Tail Prediction: (h, r, ?) =====
            heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
            relations = torch.full((num_entities,), r, dtype=torch.long, device=device)
            
            if is_gnn:
                scores = model.decode(entity_emb, heads, relations, all_entities)
            else:
                scores = model(heads, relations, all_entities)
            
            # Filtered: mask out known tails for (h, r) except the true tail t
            filter_tails = hr_to_tails.get((h, r), set())
            if filter_tails:
                filter_indices = [idx_t for idx_t in filter_tails if idx_t != t]
                if filter_indices:
                    scores[filter_indices] = float('-inf')
            
            # Compute rank of true tail
            target_score = scores[t]
            rank = (scores > target_score).sum().item() + 1
            ranks.append(rank)
            
            # ===== Head Prediction: (?, r, t) =====
            tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
            relations = torch.full((num_entities,), r, dtype=torch.long, device=device)
            
            if is_gnn:
                scores = model.decode(entity_emb, all_entities, relations, tails)
            else:
                scores = model(all_entities, relations, tails)
            
            # Filtered: mask out known heads for (r, t) except the true head h
            filter_heads = rt_to_heads.get((r, t), set())
            if filter_heads:
                filter_indices = [idx_h for idx_h in filter_heads if idx_h != h]
                if filter_indices:
                    scores[filter_indices] = float('-inf')
            
            target_score = scores[h]
            rank = (scores > target_score).sum().item() + 1
            ranks.append(rank)
    
    print()  # Clear the \r line
    
    ranks = np.array(ranks)
    
    mrr = np.mean(1.0 / ranks)
    hits_at_1 = np.mean(ranks <= 1)
    hits_at_10 = np.mean(ranks <= 10)
    
    return {
        'mrr': mrr,
        'hits_at_1': hits_at_1,
        'hits_at_10': hits_at_10
    }


# ===========================================================================
# TRAINING FUNCTIONS
# ===========================================================================

def train_kge_model(
    model: nn.Module,
    train_triples: List[Tuple[str, str, str]],
    valid_triples: List[Tuple[str, str, str]],
    entity_vocab: Dict[str, int],
    relation_vocab: Dict[str, int],
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    num_neg_samples: int = 5,
    margin: float = 1.0,
    patience: int = 5,
    valid_freq: int = 5,
    model_name: str = "model",
    verbose: bool = True
) -> Dict:
    """
    Train a KGE model.
    
    Parameters
    ----------
    model : nn.Module
        KGE model to train.
    train_triples : List[Tuple[str, str, str]]
        Training triples.
    valid_triples : List[Tuple[str, str, str]]
        Validation triples.
    entity_vocab : Dict[str, int]
        Entity vocabulary.
    relation_vocab : Dict[str, int]
        Relation vocabulary.
    device : torch.device
        Training device.
    epochs : int
        Maximum epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    num_neg_samples : int
        Negative samples per positive.
    margin : float
        Margin for ranking loss.
    patience : int
        Early stopping patience (in validation checks, not epochs).
    valid_freq : int
        Validation frequency (validate every N epochs).
    model_name : str
        Model name for logging.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    Dict
        Training history with metrics.
    """
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = TripleDataset(
        train_triples, entity_vocab, relation_vocab, num_neg_samples
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triples
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Build triple index sets for evaluation
    train_idx = [
        (entity_vocab[h], relation_vocab[r], entity_vocab[t])
        for h, r, t in train_triples
        if h in entity_vocab and t in entity_vocab and r in relation_vocab
    ]
    valid_idx = [
        (entity_vocab[h], relation_vocab[r], entity_vocab[t])
        for h, r, t in valid_triples
        if h in entity_vocab and t in entity_vocab and r in relation_vocab
    ]
    all_known = set(train_idx) | set(valid_idx)
    
    num_entities = len(entity_vocab)
    
    # Training history
    history = {
        'train_loss': [],
        'valid_mrr': [],
        'valid_hits1': [],
        'valid_hits10': [],
        'best_epoch': 0,
        'best_mrr': 0.0
    }
    
    best_mrr = 0.0
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
        for batch in pbar:
            heads = batch['heads'].to(device)
            relations = batch['relations'].to(device)
            tails = batch['tails'].to(device)
            neg_heads = batch['neg_heads'].to(device)
            neg_tails = batch['neg_tails'].to(device)
            
            # Positive scores
            pos_scores = model(heads, relations, tails)
            
            # Negative scores
            batch_size_curr = heads.size(0)
            loss = 0.0
            
            for k in range(num_neg_samples):
                neg_scores = model(neg_heads[:, k], relations, neg_tails[:, k])
                
                # Margin ranking loss
                loss += torch.mean(torch.relu(margin - pos_scores + neg_scores))
            
            loss = loss / num_neg_samples
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        history['train_loss'].append(avg_loss)
        
        # Validation (every valid_freq epochs) — skip if no validation set
        if valid_idx and ((epoch + 1) % valid_freq == 0 or epoch == epochs - 1):
            if verbose:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} — running validation ({len(valid_idx)} triples)...")
            metrics = compute_metrics(
                model, valid_idx, all_known, num_entities, device, batch_size=100
            )
            
            history['valid_mrr'].append(metrics['mrr'])
            history['valid_hits1'].append(metrics['hits_at_1'])
            history['valid_hits10'].append(metrics['hits_at_10'])
            
            if verbose:
                print(f"  => MRR={metrics['mrr']:.4f}, H@1={metrics['hits_at_1']:.4f}, "
                      f"H@10={metrics['hits_at_10']:.4f}")
            
            # Early stopping
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                history['best_epoch'] = epoch + 1
                history['best_mrr'] = best_mrr
                no_improve_count = 0
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} checks)")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Load best model (or keep final state if no validation was done)
    if 'best_state' in dir():
        model.load_state_dict(best_state)
    else:
        history['best_epoch'] = epochs
    
    return history


def train_gnn_model(
    model: nn.Module,
    train_triples: List[Tuple[str, str, str]],
    valid_triples: List[Tuple[str, str, str]],
    entity_vocab: Dict[str, int],
    relation_vocab: Dict[str, int],
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    num_neg_samples: int = 5,
    margin: float = 1.0,
    patience: int = 5,
    valid_freq: int = 5,
    model_name: str = "model",
    verbose: bool = True
) -> Dict:
    """
    Train a GNN-based link prediction model.
    
    Similar to KGE training but:
    - Builds graph structure (edge_index, edge_type) from training data
    - Encodes entities through GNN before scoring
    """
    from gnn_models import build_graph_tensors
    
    model = model.to(device)
    
    # Build graph tensors from training data
    edge_index, edge_type = build_graph_tensors(
        train_triples, entity_vocab, relation_vocab, device
    )
    
    # Create dataset
    dataset = TripleDataset(
        train_triples, entity_vocab, relation_vocab, num_neg_samples
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triples
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Build triple sets for evaluation
    train_idx = [
        (entity_vocab[h], relation_vocab[r], entity_vocab[t])
        for h, r, t in train_triples
        if h in entity_vocab and t in entity_vocab and r in relation_vocab
    ]
    valid_idx = [
        (entity_vocab[h], relation_vocab[r], entity_vocab[t])
        for h, r, t in valid_triples
        if h in entity_vocab and t in entity_vocab and r in relation_vocab
    ]
    all_known = set(train_idx) | set(valid_idx)
    
    num_entities = len(entity_vocab)
    
    # History
    history = {
        'train_loss': [],
        'valid_mrr': [],
        'valid_hits1': [],
        'valid_hits10': [],
        'best_epoch': 0,
        'best_mrr': 0.0
    }
    
    best_mrr = 0.0
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Compute entity embeddings once per epoch (for efficiency)
        entity_emb = model.encode(edge_index, edge_type)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
        for batch in pbar:
            heads = batch['heads'].to(device)
            relations = batch['relations'].to(device)
            tails = batch['tails'].to(device)
            neg_heads = batch['neg_heads'].to(device)
            neg_tails = batch['neg_tails'].to(device)
            
            # Re-encode (needed for gradient flow)
            entity_emb = model.encode(edge_index, edge_type)
            
            # Positive scores
            pos_scores = model.decode(entity_emb, heads, relations, tails)
            
            # Negative scores and loss
            loss = 0.0
            for k in range(num_neg_samples):
                neg_scores = model.decode(
                    entity_emb, neg_heads[:, k], relations, neg_tails[:, k]
                )
                loss += torch.mean(torch.relu(margin - pos_scores + neg_scores))
            
            loss = loss / num_neg_samples
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        history['train_loss'].append(avg_loss)
        
        # Validation (every valid_freq epochs) — skip if no validation set
        if valid_idx and ((epoch + 1) % valid_freq == 0 or epoch == epochs - 1):
            if verbose:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} — running validation ({len(valid_idx)} triples)...")
            metrics = compute_metrics(
                model, valid_idx, all_known, num_entities, device,
                batch_size=100, edge_index=edge_index, edge_type=edge_type, is_gnn=True
            )
            
            history['valid_mrr'].append(metrics['mrr'])
            history['valid_hits1'].append(metrics['hits_at_1'])
            history['valid_hits10'].append(metrics['hits_at_10'])
            
            if verbose:
                print(f"  => MRR={metrics['mrr']:.4f}, H@1={metrics['hits_at_1']:.4f}, "
                      f"H@10={metrics['hits_at_10']:.4f}")
            
            # Early stopping
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                history['best_epoch'] = epoch + 1
                history['best_mrr'] = best_mrr
                no_improve_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} checks)")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    if 'best_state' in dir():
        model.load_state_dict(best_state)
    else:
        history['best_epoch'] = epochs
    
    return history


# ===========================================================================
# EVALUATION ON TEST SET
# ===========================================================================

def evaluate_on_test(
    model: nn.Module,
    test_triples: List[Tuple[str, str, str]],
    train_triples: List[Tuple[str, str, str]],
    valid_triples: List[Tuple[str, str, str]],
    entity_vocab: Dict[str, int],
    relation_vocab: Dict[str, int],
    device: torch.device,
    is_gnn: bool = False,
    edge_index: torch.Tensor = None,
    edge_type: torch.Tensor = None
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    test_triples : List[Tuple[str, str, str]]
        Test triples.
    train_triples : List[Tuple[str, str, str]]
        Training triples (for filtering).
    valid_triples : List[Tuple[str, str, str]]
        Validation triples (for filtering).
    entity_vocab : Dict[str, int]
        Entity vocabulary.
    relation_vocab : Dict[str, int]
        Relation vocabulary.
    device : torch.device
        Evaluation device.
    is_gnn : bool
        Whether model is GNN-based.
    edge_index : torch.Tensor, optional
        Graph edges for GNN.
    edge_type : torch.Tensor, optional
        Edge types for GNN.
        
    Returns
    -------
    Dict[str, float]
        Test metrics.
    """
    # Convert to indices (handle unseen entities)
    test_idx = []
    skipped = 0
    for h, r, t in test_triples:
        if h in entity_vocab and t in entity_vocab and r in relation_vocab:
            test_idx.append((
                entity_vocab[h],
                relation_vocab[r],
                entity_vocab[t]
            ))
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} test triples with unknown entities/relations")
    
    # Build all known triples set
    all_known = set()
    for triples in [train_triples, valid_triples, test_triples]:
        for h, r, t in triples:
            if h in entity_vocab and t in entity_vocab and r in relation_vocab:
                all_known.add((entity_vocab[h], relation_vocab[r], entity_vocab[t]))
    
    num_entities = len(entity_vocab)
    
    return compute_metrics(
        model, test_idx, all_known, num_entities, device,
        batch_size=100, edge_index=edge_index, edge_type=edge_type, is_gnn=is_gnn
    )


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def save_results(results: Dict, filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def results_to_csv(all_results: List[Dict], filepath: str):
    """
    Convert results list to CSV.
    
    Parameters
    ----------
    all_results : List[Dict]
        List of result dicts with keys: split_type, model, valid_mrr, etc.
    filepath : str
        Output CSV path.
    """
    import csv
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    headers = [
        'split_type', 'model',
        'valid_mrr', 'valid_hits1', 'valid_hits10',
        'test_mrr', 'test_hits1', 'test_hits10',
        'best_epoch'
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_results:
            # Select only the columns we want
            row_filtered = {k: row.get(k, '') for k in headers}
            writer.writerow(row_filtered)
