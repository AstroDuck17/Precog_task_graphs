"""
Data Loader Module
==================

This module handles the loading and construction of NetworkX graph objects
from the MetaFam knowledge graph dataset.

The dataset format is tab-separated triples: (Head, Relation, Tail)
where Head and Tail are entity nodes, and Relation is the edge type.

Theory:
-------
In knowledge graphs, representing data as both directed and undirected graphs
serves different analytical purposes:

1. Directed Graph (DiGraph):
   - Preserves semantic direction of relationships (e.g., "motherOf" implies directionality)
   - Essential for genealogical analysis where parent→child direction matters
   - Enables in-degree/out-degree analysis to identify ancestors vs descendants
   - Supports topological ordering and generational depth calculation

2. Undirected Graph:
   - Treats relationships as bidirectional connections
   - Useful for community detection and clustering analysis
   - Better for measuring overall connectivity and structural properties
   - Supports standard algorithms that require undirected input

Author: MetaFam Analysis Team
"""

import networkx as nx
from typing import Tuple
from pathlib import Path


def load_graph(filepath: str) -> Tuple[nx.DiGraph, nx.Graph]:
    """
    Load the MetaFam knowledge graph from a tab-separated triples file.
    
    This function reads the knowledge graph data and constructs both directed
    and undirected NetworkX graph representations. Each edge is labeled with
    its relationship type.
    
    Parameters
    ----------
    filepath : str
        Path to the input file (train.txt) containing tab-separated triples
        in the format: Head<TAB>Relation<TAB>Tail
    
    Returns
    -------
    Tuple[nx.DiGraph, nx.Graph]
        G_directed : nx.DiGraph
            Directed multigraph where edges preserve relationship direction.
            Multiple edges between same nodes (different relations) are allowed.
        G_undirected : nx.Graph
            Undirected graph for connectivity and community analysis.
    
    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If lines in the file don't conform to the expected triple format.
    
    Example
    -------
    >>> G_directed, G_undirected = load_graph('data/train.txt')
    >>> print(f"Nodes: {G_directed.number_of_nodes()}")
    >>> print(f"Edges: {G_directed.number_of_edges()}")
    
    Notes
    -----
    - The directed graph uses nx.DiGraph (not MultiDiGraph) to avoid complexity.
      If multiple relations exist between the same pair of nodes, they are stored
      as a list in the 'relations' edge attribute.
    - All nodes are automatically added when edges are created.
    - Edge attributes include 'relation' (str or list) for analysis.
    """
    
    # Validate file exists
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Initialize graphs
    # Using DiGraph for directed (single edge between node pairs, but we track all relations)
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()
    
    # Track all relations for each edge (for multi-relation edges)
    edge_relations_directed = {}
    edge_relations_undirected = {}
    
    # Statistics for sanity check
    line_count = 0
    unique_relations = set()
    parse_errors = []
    
    print(f"Loading graph from: {filepath}")
    print("-" * 50)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Try tab-separated first, then space-separated
            parts = line.split('\t')
            if len(parts) != 3:
                # Try space-separated (exactly 3 parts)
                parts = line.split()
            
            if len(parts) != 3:
                parse_errors.append((line_num, line))
                continue
            
            head, relation, tail = parts
            line_count += 1
            unique_relations.add(relation)
            
            # Track directed edges with their relations
            edge_key_directed = (head, tail)
            if edge_key_directed not in edge_relations_directed:
                edge_relations_directed[edge_key_directed] = []
            edge_relations_directed[edge_key_directed].append(relation)
            
            # Track undirected edges (canonical ordering for consistency)
            edge_key_undirected = tuple(sorted([head, tail]))
            if edge_key_undirected not in edge_relations_undirected:
                edge_relations_undirected[edge_key_undirected] = []
            edge_relations_undirected[edge_key_undirected].append(relation)
    
    # Build directed graph with relation attributes
    for (head, tail), relations in edge_relations_directed.items():
        if len(relations) == 1:
            G_directed.add_edge(head, tail, relation=relations[0])
        else:
            # Multiple relations between same node pair - store as list
            G_directed.add_edge(head, tail, relation=relations)
    
    # Build undirected graph
    for (node1, node2), relations in edge_relations_undirected.items():
        if len(relations) == 1:
            G_undirected.add_edge(node1, node2, relation=relations[0])
        else:
            G_undirected.add_edge(node1, node2, relation=relations)
    
    # Print sanity checks
    print(f"\n{'='*50}")
    print("GRAPH CONSTRUCTION SANITY CHECK")
    print(f"{'='*50}")
    print(f"\nDATA STATISTICS:")
    print(f"   - Total triples processed: {line_count:,}")
    print(f"   - Unique relationship types: {len(unique_relations)}")
    
    if parse_errors:
        print(f"\n PARSE ERRORS: {len(parse_errors)} lines")
        for line_num, line in parse_errors[:5]:  # Show first 5 errors
            print(f"      Line {line_num}: {line[:50]}...")
    
    print(f"\nDIRECTED GRAPH (G_directed):")
    print(f"   - Nodes: {G_directed.number_of_nodes():,}")
    print(f"   - Edges: {G_directed.number_of_edges():,}")
    print(f"   - Graph type: {type(G_directed).__name__}")
    
    print(f"\nUNDIRECTED GRAPH (G_undirected):")
    print(f"   - Nodes: {G_undirected.number_of_nodes():,}")
    print(f"   - Edges: {G_undirected.number_of_edges():,}")
    print(f"   - Graph type: {type(G_undirected).__name__}")
    
    return G_directed, G_undirected


def get_relationship_counts(G_directed: nx.DiGraph) -> dict:
    """
    Calculate the frequency of each relationship type in the graph.
    
    This function iterates through all edges and counts occurrences of
    each relationship type. Handles both single relations and multi-relation
    edges.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed graph with 'relation' edge attributes.
    
    Returns
    -------
    dict
        Dictionary mapping relationship types to their counts.
    
    Example
    -------
    >>> rel_counts = get_relationship_counts(G_directed)
    >>> print(rel_counts['motherOf'])
    """
    from collections import Counter
    
    relation_counts = Counter()
    
    for u, v, data in G_directed.edges(data=True):
        relation = data.get('relation', 'unknown')
        if isinstance(relation, list):
            for r in relation:
                relation_counts[r] += 1
        else:
            relation_counts[relation] += 1
    
    return dict(relation_counts)


def get_unique_relations(G_directed: nx.DiGraph) -> set:
    """
    Extract all unique relationship types from the graph.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed graph with 'relation' edge attributes.
    
    Returns
    -------
    set
        Set of unique relationship type strings.
    """
    unique_relations = set()
    
    for u, v, data in G_directed.edges(data=True):
        relation = data.get('relation', 'unknown')
        if isinstance(relation, list):
            unique_relations.update(relation)
        else:
            unique_relations.add(relation)
    
    return unique_relations


if __name__ == "__main__":
    # Quick test
    import os
    
    # Determine path relative to this file
    current_dir = Path(__file__).parent.parent
    data_path = current_dir / "data" / "train.txt"
    
    if data_path.exists():
        G_d, G_u = load_graph(str(data_path))
        print("\nRelationship Types:")
        print(sorted(get_unique_relations(G_d)))
    else:
        print(f"Test data not found at: {data_path}")
