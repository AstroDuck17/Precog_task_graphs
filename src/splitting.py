"""
Data Splitting Module for Link Prediction
==========================================

This module implements three distinct data splitting strategies for evaluating
Knowledge Graph Embedding models on the MetaFam dataset.

Splitting Strategies:
---------------------
1. **Type 1 - Naive Random (Inductive Risk)**:
   - Random 80/20 split
   - Vocabulary defined ONLY from train_subset
   - Risk: Validation may contain unseen entities

2. **Type 2 - Transductive (Shared Vocabulary)**:
   - Random 80/20 split
   - Vocabulary defined from UNION of train + valid
   - Standard KGE setup with all entities having embeddings

3. **Type 3 - Inverse-Leakage Removal (Symmetry Aware)**:
   - Treats inverse relation pairs as single "interaction units"
   - Prevents (Father(A,B), Son(B,A)) from being split across train/valid
   - Forces model to learn genuine patterns, not memorize inverses

Theory:
-------
Family knowledge graphs exhibit high symmetry - if Father(A,B) exists,
Son(B,A) usually exists. Standard random splits create easy "leakage"
where the model can exploit inverse relations. Type 3 splitting removes
this leakage by ensuring inverse pairs stay together.

Author: MetaFam Analysis Team
"""

import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np


# ===========================================================================
# INVERSE RELATION DEFINITIONS
# ===========================================================================

# Define inverse relation mappings for MetaFam
# Format: relation -> list of its inverse relations
INVERSE_RELATIONS = {
    # Parent-Child Pairs
    'fatherOf': ['sonOf', 'daughterOf'],
    'motherOf': ['sonOf', 'daughterOf'],
    'sonOf': ['fatherOf', 'motherOf'],
    'daughterOf': ['fatherOf', 'motherOf'],
    
    # Grandparent-Grandchild Pairs
    'grandfatherOf': ['grandsonOf', 'granddaughterOf'],
    'grandmotherOf': ['grandsonOf', 'granddaughterOf'],
    'grandsonOf': ['grandfatherOf', 'grandmotherOf'],
    'granddaughterOf': ['grandfatherOf', 'grandmotherOf'],
    
    # Great-Grandparent Pairs
    'greatGrandfatherOf': ['greatGrandsonOf', 'greatGranddaughterOf'],
    'greatGrandmotherOf': ['greatGrandsonOf', 'greatGranddaughterOf'],
    'greatGrandsonOf': ['greatGrandfatherOf', 'greatGrandmotherOf'],
    'greatGranddaughterOf': ['greatGrandfatherOf', 'greatGrandmotherOf'],
    
    # Sibling Pairs (symmetric relations)
    'brotherOf': ['brotherOf', 'sisterOf'],
    'sisterOf': ['brotherOf', 'sisterOf'],
    
    # Uncle/Aunt - Nephew/Niece Pairs
    'uncleOf': ['nephewOf', 'nieceOf'],
    'auntOf': ['nephewOf', 'nieceOf'],
    'nephewOf': ['uncleOf', 'auntOf'],
    'nieceOf': ['uncleOf', 'auntOf'],
    
    # Great Uncle/Aunt Pairs
    'greatUncleOf': ['nephewOf', 'nieceOf'],  # Great-nephew/niece not explicit
    'greatAuntOf': ['nephewOf', 'nieceOf'],
    
    # Second Uncle/Aunt - First Cousin Once Removed
    'secondUncleOf': ['boyFirstCousinOnceRemovedOf', 'girlFirstCousinOnceRemovedOf'],
    'secondAuntOf': ['boyFirstCousinOnceRemovedOf', 'girlFirstCousinOnceRemovedOf'],
    'boyFirstCousinOnceRemovedOf': ['secondUncleOf', 'secondAuntOf'],
    'girlFirstCousinOnceRemovedOf': ['secondUncleOf', 'secondAuntOf'],
    
    # Cousin Pairs
    'boyCousinOf': ['boyCousinOf', 'girlCousinOf'],
    'girlCousinOf': ['boyCousinOf', 'girlCousinOf'],
    'boySecondCousinOf': ['boySecondCousinOf', 'girlSecondCousinOf'],
    'girlSecondCousinOf': ['boySecondCousinOf', 'girlSecondCousinOf'],
}


@dataclass
class SplitResult:
    """Container for split results with metadata."""
    train_triples: List[Tuple[str, str, str]]
    valid_triples: List[Tuple[str, str, str]]
    split_type: str
    entity_vocab: Dict[str, int]  # entity -> index
    relation_vocab: Dict[str, int]  # relation -> index
    vocab_source: str  # 'train_only' or 'train_valid_union'
    stats: Dict[str, any]
    
    def __str__(self) -> str:
        return (f"SplitResult(type={self.split_type}, "
                f"train={len(self.train_triples)}, valid={len(self.valid_triples)}, "
                f"entities={len(self.entity_vocab)}, relations={len(self.relation_vocab)})")


def load_triples(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Load triples from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the triples file.
        
    Returns
    -------
    List[Tuple[str, str, str]]
        List of (head, relation, tail) tuples.
    """
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_vocabulary(triples: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build entity and relation vocabularies from triples.
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        List of triples.
        
    Returns
    -------
    Tuple[Dict[str, int], Dict[str, int]]
        (entity_vocab, relation_vocab) mappings.
    """
    entities = set()
    relations = set()
    
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    
    entity_vocab = {e: i for i, e in enumerate(sorted(entities))}
    relation_vocab = {r: i for i, r in enumerate(sorted(relations))}
    
    return entity_vocab, relation_vocab


# ===========================================================================
# SPLIT TYPE 1: NAIVE RANDOM (INDUCTIVE RISK)
# ===========================================================================

def split_naive_random(
    triples: List[Tuple[str, str, str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> SplitResult:
    """
    Type 1: Naive Random Split (Inductive Risk).
    
    Randomly shuffles all triples and splits 80/20. The vocabulary is built
    ONLY from the training subset, creating risk of unseen entities in validation.
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        All triples to split.
    train_ratio : float
        Fraction for training (default 0.8).
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    SplitResult
        Split result with train_only vocabulary.
    
    Theory:
    -------
    This simulates a real-world scenario where new entities may appear at
    inference time. Models must handle "cold start" entities gracefully.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle and split
    shuffled = list(triples)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_triples = shuffled[:split_idx]
    valid_triples = shuffled[split_idx:]
    
    # Build vocabulary ONLY from training data
    entity_vocab, relation_vocab = build_vocabulary(train_triples)
    
    # Calculate statistics about unseen entities
    train_entities = set(entity_vocab.keys())
    train_relations = set(relation_vocab.keys())
    
    valid_entities = set()
    valid_relations = set()
    for h, r, t in valid_triples:
        valid_entities.add(h)
        valid_entities.add(t)
        valid_relations.add(r)
    
    unseen_entities = valid_entities - train_entities
    unseen_relations = valid_relations - train_relations
    
    # Count triples with unseen entities
    triples_with_unseen = 0
    for h, r, t in valid_triples:
        if h not in train_entities or t not in train_entities or r not in train_relations:
            triples_with_unseen += 1
    
    stats = {
        'train_entities': len(train_entities),
        'valid_entities': len(valid_entities),
        'unseen_entities': len(unseen_entities),
        'unseen_entity_list': list(unseen_entities)[:10],  # Sample
        'unseen_relations': len(unseen_relations),
        'triples_with_unseen': triples_with_unseen,
        'unseen_triple_pct': triples_with_unseen / len(valid_triples) * 100 if valid_triples else 0
    }
    
    return SplitResult(
        train_triples=train_triples,
        valid_triples=valid_triples,
        split_type='naive_random',
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        vocab_source='train_only',
        stats=stats
    )


# ===========================================================================
# SPLIT TYPE 2: TRANSDUCTIVE (SHARED VOCABULARY)
# ===========================================================================

def split_transductive(
    triples: List[Tuple[str, str, str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> SplitResult:
    """
    Type 2: Transductive Split (Shared Vocabulary).
    
    Same random 80/20 split as Type 1, but vocabulary is built from the
    UNION of train and validation sets. This is the standard KGE setup.
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        All triples to split.
    train_ratio : float
        Fraction for training (default 0.8).
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    SplitResult
        Split result with union vocabulary.
    
    Theory:
    -------
    All entities have embedding slots initialized, even if they don't appear
    in training. This allows the model to leverage graph structure but may
    lead to overfitting on validation set characteristics.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle and split (same as Type 1)
    shuffled = list(triples)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_triples = shuffled[:split_idx]
    valid_triples = shuffled[split_idx:]
    
    # Build vocabulary from UNION of train + valid
    all_triples = train_triples + valid_triples
    entity_vocab, relation_vocab = build_vocabulary(all_triples)
    
    # Statistics
    train_entities = set()
    for h, r, t in train_triples:
        train_entities.add(h)
        train_entities.add(t)
    
    valid_only_entities = set()
    for h, r, t in valid_triples:
        if h not in train_entities:
            valid_only_entities.add(h)
        if t not in train_entities:
            valid_only_entities.add(t)
    
    stats = {
        'total_entities': len(entity_vocab),
        'train_entities': len(train_entities),
        'valid_only_entities': len(valid_only_entities),
        'valid_only_pct': len(valid_only_entities) / len(entity_vocab) * 100 if entity_vocab else 0
    }
    
    return SplitResult(
        train_triples=train_triples,
        valid_triples=valid_triples,
        split_type='transductive',
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        vocab_source='train_valid_union',
        stats=stats
    )


# ===========================================================================
# SPLIT TYPE 3: INVERSE-LEAKAGE REMOVAL (SYMMETRY AWARE)
# ===========================================================================

def find_inverse_pairs(
    triples: List[Tuple[str, str, str]]
) -> Tuple[List[Set[int]], List[int]]:
    """
    Identify inverse relation pairs and group them as "interaction units".
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        List of triples.
        
    Returns
    -------
    Tuple[List[Set[int]], List[int]]
        - List of interaction units (sets of triple indices)
        - List mapping each triple index to its unit index
    
    Theory:
    -------
    An interaction unit contains triples that are semantically connected
    through inverse relations. If (A, fatherOf, B) exists and (B, sonOf, A)
    exists, they form one unit and must stay together during splitting.
    """
    # Build index of (head, tail) -> list of (relation, triple_idx)
    edge_to_triples: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    
    for idx, (h, r, t) in enumerate(triples):
        edge_to_triples[(h, t)].append((r, idx))
    
    # Find inverse pairs using Union-Find
    parent = list(range(len(triples)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # For each triple, check if its inverse exists
    for idx, (h, r, t) in enumerate(triples):
        inverse_rels = INVERSE_RELATIONS.get(r, [])
        
        # Check (t, h) for inverse relations
        if (t, h) in edge_to_triples:
            for inv_rel, inv_idx in edge_to_triples[(t, h)]:
                if inv_rel in inverse_rels:
                    union(idx, inv_idx)
    
    # Group triples by their root
    unit_map: Dict[int, Set[int]] = defaultdict(set)
    for idx in range(len(triples)):
        root = find(idx)
        unit_map[root].add(idx)
    
    # Convert to list format
    units = list(unit_map.values())
    triple_to_unit = [-1] * len(triples)
    for unit_idx, unit in enumerate(units):
        for triple_idx in unit:
            triple_to_unit[triple_idx] = unit_idx
    
    return units, triple_to_unit


def split_inverse_leakage_removal(
    triples: List[Tuple[str, str, str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> SplitResult:
    """
    Type 3: Inverse-Leakage Removal Split (Symmetry Aware).
    
    Groups inverse relation pairs as "interaction units" and splits these
    units 80/20. This prevents leakage where (A, fatherOf, B) is in train
    and (B, sonOf, A) is in validation.
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        All triples to split.
    train_ratio : float
        Fraction for training (default 0.8).
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    SplitResult
        Split result with inverse-leakage-free splits.
    
    Theory:
    -------
    Standard random splits create "easy" validation samples because models
    can memorize A->B implies B->A. This split forces the model to learn
    genuine relational patterns by keeping inverse pairs together.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Find interaction units
    units, triple_to_unit = find_inverse_pairs(triples)
    
    # Shuffle units (not individual triples)
    unit_indices = list(range(len(units)))
    random.shuffle(unit_indices)
    
    # Split units 80/20
    split_idx = int(len(unit_indices) * train_ratio)
    train_unit_indices = set(unit_indices[:split_idx])
    
    # Assign triples to train/valid based on their unit
    train_triples = []
    valid_triples = []
    
    for idx, triple in enumerate(triples):
        unit_idx = triple_to_unit[idx]
        if unit_idx in train_unit_indices:
            train_triples.append(triple)
        else:
            valid_triples.append(triple)
    
    # Build vocabulary from union (standard for Type 3)
    all_triples = train_triples + valid_triples
    entity_vocab, relation_vocab = build_vocabulary(all_triples)
    
    # Statistics about inverse pair handling
    multi_triple_units = [u for u in units if len(u) > 1]
    inverse_pairs_count = sum(len(u) - 1 for u in multi_triple_units)
    
    stats = {
        'total_units': len(units),
        'single_triple_units': len(units) - len(multi_triple_units),
        'multi_triple_units': len(multi_triple_units),
        'inverse_pairs_grouped': inverse_pairs_count,
        'train_units': len(train_unit_indices),
        'valid_units': len(units) - len(train_unit_indices),
        'leakage_removed': True
    }
    
    return SplitResult(
        train_triples=train_triples,
        valid_triples=valid_triples,
        split_type='inverse_leakage_removal',
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        vocab_source='train_valid_union',
        stats=stats
    )


# ===========================================================================
# MAIN INTERFACE
# ===========================================================================

def generate_splits(
    triples: List[Tuple[str, str, str]],
    split_type: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> SplitResult:
    """
    Generate train/validation splits using the specified strategy.
    
    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        All triples to split.
    split_type : str
        One of: 'naive_random', 'transductive', 'inverse_leakage_removal'
    train_ratio : float
        Fraction for training (default 0.8).
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    SplitResult
        Split result with appropriate vocabulary and statistics.
        
    Raises
    ------
    ValueError
        If split_type is not recognized.
    """
    if split_type == 'naive_random':
        return split_naive_random(triples, train_ratio, seed)
    elif split_type == 'transductive':
        return split_transductive(triples, train_ratio, seed)
    elif split_type == 'inverse_leakage_removal':
        return split_inverse_leakage_removal(triples, train_ratio, seed)
    else:
        raise ValueError(f"Unknown split type: {split_type}. "
                        f"Expected: naive_random, transductive, inverse_leakage_removal")


def save_split(split_result: SplitResult, output_dir: str, prefix: str = '') -> None:
    """
    Save split results to files.
    
    Parameters
    ----------
    split_result : SplitResult
        The split to save.
    output_dir : str
        Directory to save files.
    prefix : str
        Optional prefix for filenames.
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{prefix}_" if prefix else ""
    
    # Save train triples
    with open(output_path / f"{prefix}train.txt", 'w') as f:
        for h, r, t in split_result.train_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    # Save valid triples
    with open(output_path / f"{prefix}valid.txt", 'w') as f:
        for h, r, t in split_result.valid_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    # Save vocabulary
    with open(output_path / f"{prefix}entity_vocab.txt", 'w') as f:
        for entity, idx in sorted(split_result.entity_vocab.items(), key=lambda x: x[1]):
            f.write(f"{entity}\t{idx}\n")
    
    with open(output_path / f"{prefix}relation_vocab.txt", 'w') as f:
        for rel, idx in sorted(split_result.relation_vocab.items(), key=lambda x: x[1]):
            f.write(f"{rel}\t{idx}\n")
    
    print(f"Saved split '{split_result.split_type}' to {output_path}")


def print_split_summary(split_result: SplitResult) -> None:
    """Print a summary of the split result."""
    print(f"\n{'='*60}")
    print(f"SPLIT TYPE: {split_result.split_type.upper()}")
    print(f"{'='*60}")
    print(f"Train triples: {len(split_result.train_triples)}")
    print(f"Valid triples: {len(split_result.valid_triples)}")
    print(f"Vocabulary source: {split_result.vocab_source}")
    print(f"Total entities: {len(split_result.entity_vocab)}")
    print(f"Total relations: {len(split_result.relation_vocab)}")
    print(f"\nStatistics:")
    for key, value in split_result.stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"  {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
        else:
            print(f"  {key}: {value}")
