"""
MetaFam Graph Analysis Package

Modules:
    - data_loader: Functions for loading and constructing NetworkX graphs
    - exploration: Network analysis, metrics, and node classification functions
    - communities: Community detection algorithms (Girvan-Newman, Louvain, Leiden)
    - proximity: Proximity measures (RWR, Katz) for finding closest relatives
    - rules: Rule mining and validation for horn-clause rules
    - splitting: Data splitting strategies for link prediction
    - kge_models: Knowledge Graph Embedding models (TransE, DistMult, ComplEx, RotatE)
    - gnn_models: Graph Neural Network models (RGCN + decoders)
    - train_eval: Training and evaluation utilities for link prediction
"""

from .data_loader import load_graph, get_relationship_counts, get_unique_relations
from .exploration import (
    calculate_global_metrics,
    print_global_metrics,
    calculate_node_features,
    compute_generational_depth,
    infer_gender,
    get_gender_summary,
    export_to_gephi,
    plot_relationship_distribution,
    plot_degree_distribution,
    plot_generation_distribution
)
from .communities import (
    girvan_newman_communities,
    louvain_communities,
    leiden_communities,
    calculate_modularity,
    calculate_nmi,
    calculate_ari,
    compare_partitions,
    analyze_generational_depth_per_community,
    plot_generational_histogram,
    assign_community_attribute,
    run_all_community_detection
)
from .proximity import (
    random_walk_restart,
    rwr_personalized_pagerank,
    katz_similarity,
    find_closest_relatives,
    analyze_proximity_for_node_types,
    compare_proximity_methods
)
from .rules import (
    RuleValidator,
    RuleResult,
    print_rule_details
)
from .splitting import (
    load_triples,
    build_vocabulary,
    SplitResult,
    split_naive_random,
    split_transductive,
    split_inverse_leakage_removal,
    find_inverse_pairs,
    generate_splits
)
from .kge_models import (
    BaseKGEModel,
    TransE,
    DistMult,
    ComplEx,
    RotatE,
    create_kge_model,
    MarginRankingLoss,
    BinaryCrossEntropyLoss
)
from .gnn_models import (
    RGCNLayer,
    RGCN,
    DistMultDecoder,
    RotatEDecoder,
    RGCNLinkPredictor,
    build_graph_tensors,
    create_gnn_model
)
from .train_eval import (
    TripleDataset,
    collate_triples,
    compute_metrics,
    train_kge_model,
    train_gnn_model,
    evaluate_on_test,
    save_results,
    results_to_csv
)

__all__ = [
    # Data loading
    'load_graph',
    'get_relationship_counts',
    'get_unique_relations',
    # Exploration
    'calculate_global_metrics',
    'print_global_metrics',
    'calculate_node_features',
    'compute_generational_depth',
    'infer_gender',
    'get_gender_summary',
    'export_to_gephi',
    'plot_relationship_distribution',
    'plot_degree_distribution',
    'plot_generation_distribution',
    # Communities
    'girvan_newman_communities',
    'louvain_communities',
    'leiden_communities',
    'calculate_modularity',
    'calculate_nmi',
    'calculate_ari',
    'compare_partitions',
    'analyze_generational_depth_per_community',
    'plot_generational_histogram',
    'assign_community_attribute',
    'run_all_community_detection',
    # Proximity
    'random_walk_restart',
    'rwr_personalized_pagerank',
    'katz_similarity',
    'find_closest_relatives',
    'analyze_proximity_for_node_types',
    'compare_proximity_methods',
    # Rules
    'RuleValidator',
    'RuleResult',
    'print_rule_details',
    # Splitting (Task 4)
    'load_triples',
    'build_vocabulary',
    'SplitResult',
    'split_naive_random',
    'split_transductive',
    'split_inverse_leakage_removal',
    'find_inverse_pairs',
    'generate_splits',
    # KGE Models (Task 4)
    'BaseKGEModel',
    'TransE',
    'DistMult',
    'ComplEx',
    'RotatE',
    'create_kge_model',
    'MarginRankingLoss',
    'BinaryCrossEntropyLoss',
    # GNN Models (Task 4)
    'RGCNLayer',
    'RGCN',
    'DistMultDecoder',
    'RotatEDecoder',
    'RGCNLinkPredictor',
    'build_graph_tensors',
    'create_gnn_model',
    # Training & Evaluation (Task 4)
    'TripleDataset',
    'collate_triples',
    'compute_metrics',
    'train_kge_model',
    'train_gnn_model',
    'evaluate_on_test',
    'save_results',
    'results_to_csv'
]

__version__ = '4.0.0'
