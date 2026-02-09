"""
communities.py - Community Detection Module for MetaFam Knowledge Graph

This module implements community detection algorithms for identifying
natural groupings in the family knowledge graph.

Algorithms Implemented:
    1. Girvan-Newman: Edge betweenness-based hierarchical divisive clustering
    2. Louvain: Modularity optimization through greedy agglomeration
    3. Leiden: Improved Louvain with guaranteed connectivity

Evaluation Metrics:
    - Modularity (Q): Quality of partition based on edge density
    - NMI: Normalized Mutual Information between partitions
    - ARI: Adjusted Rand Index for clustering similarity

Author: MetaFam Analysis Pipeline
"""

import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
from itertools import islice
import warnings

# Try importing community detection libraries
try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    warnings.warn("python-louvain not installed. Louvain algorithm unavailable.")

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn("leidenalg/igraph not installed. Leiden algorithm unavailable.")

try:
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. NMI/ARI metrics unavailable.")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def partition_to_dict(partition: List[Set]) -> Dict[Any, int]:
    """
    Convert a list of sets (partition) to a dict mapping node -> community_id.
    
    Parameters
    ----------
    partition : List[Set]
        List where each set contains nodes in a community.
    
    Returns
    -------
    Dict[Any, int]
        Mapping of node to community ID (0-indexed).
    """
    result = {}
    for comm_id, community in enumerate(partition):
        for node in community:
            result[node] = comm_id
    return result


def dict_to_partition(node_to_comm: Dict[Any, int]) -> List[Set]:
    """
    Convert a dict mapping node -> community_id to list of sets.
    
    Parameters
    ----------
    node_to_comm : Dict[Any, int]
        Mapping of node to community ID.
    
    Returns
    -------
    List[Set]
        List where each set contains nodes in a community.
    """
    communities = defaultdict(set)
    for node, comm_id in node_to_comm.items():
        communities[comm_id].add(node)
    return list(communities.values())


def get_labels_from_partition(partition: Dict[Any, int], node_order: List) -> List[int]:
    """
    Convert partition dict to ordered list of labels for sklearn metrics.
    
    Parameters
    ----------
    partition : Dict[Any, int]
        Mapping of node to community ID.
    node_order : List
        Ordered list of nodes.
    
    Returns
    -------
    List[int]
        Community labels in same order as node_order.
    """
    return [partition.get(node, -1) for node in node_order]


# ============================================================================
# GIRVAN-NEWMAN ALGORITHM
# ============================================================================

def girvan_newman_communities(G: nx.Graph, 
                               num_communities: Optional[int] = None,
                               verbose: bool = True) -> Tuple[Dict[Any, int], List[Set]]:
    """
    Apply Girvan-Newman algorithm to detect communities.
    
    The Girvan-Newman algorithm iteratively removes edges with highest
    betweenness centrality, producing a dendrogram of communities.
    
    Theoretical Background:
    -----------------------
    Edge betweenness centrality measures how many shortest paths pass
    through an edge. Edges connecting communities have high betweenness.
    By removing these "bridge" edges, we reveal the community structure.
    
    For genealogical graphs, this often identifies:
    - Marriage links between families
    - Key ancestral connections
    
    Computational Complexity: O(m^2 * n) where m = edges, n = nodes
    
    Parameters
    ----------
    G : nx.Graph
        Undirected graph (will convert if directed).
    num_communities : int, optional
        Target number of communities. If None, uses modularity optimization.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    Tuple[Dict[Any, int], List[Set]]
        - node_to_community: Mapping of node to community ID
        - communities: List of sets, each containing nodes in a community
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GIRVAN-NEWMAN COMMUNITY DETECTION")
        print("=" * 60)
    
    # Ensure undirected graph
    if G.is_directed():
        G_undirected = G.to_undirected()
        if verbose:
            print("Note: Converted directed graph to undirected for analysis.")
    else:
        G_undirected = G.copy()
    
    # Get component generator
    comp = nx.community.girvan_newman(G_undirected)
    
    if num_communities is not None:
        # Get partition with specified number of communities
        if verbose:
            print(f"Targeting {num_communities} communities...")
        
        for communities in comp:
            if len(communities) >= num_communities:
                break
    else:
        # Use modularity to find best partition
        if verbose:
            print("Finding optimal partition via modularity...")
        
        best_modularity = -1
        best_communities = None
        
        # Limit iterations to prevent long computation
        max_iterations = min(G_undirected.number_of_nodes() // 2, 100)
        
        for i, communities in enumerate(islice(comp, max_iterations)):
            mod = nx.community.modularity(G_undirected, communities)
            if mod > best_modularity:
                best_modularity = mod
                best_communities = communities
            
            if verbose and (i + 1) % 10 == 0:
                print(f"   Iteration {i+1}: {len(communities)} communities, Q={mod:.4f}")
        
        communities = best_communities
    
    # Convert to dict format
    node_to_community = partition_to_dict(list(communities))
    communities_list = list(communities)
    
    if verbose:
        print(f"\nResults:")
        print(f"   - Number of communities: {len(communities_list)}")
        print(f"   - Largest community: {max(len(c) for c in communities_list)} nodes")
        print(f"   - Smallest community: {min(len(c) for c in communities_list)} nodes")
        
        # Community size distribution
        sizes = sorted([len(c) for c in communities_list], reverse=True)
        print(f"   - Community sizes (top 10): {sizes[:10]}")
        
        # Modularity
        mod = nx.community.modularity(G_undirected, communities_list)
        print(f"   - Modularity: {mod:.4f}")
        
        print("=" * 60)
    
    return node_to_community, communities_list


# ============================================================================
# LOUVAIN ALGORITHM
# ============================================================================

def louvain_communities(G: nx.Graph,
                        resolution: float = 1.0,
                        verbose: bool = True) -> Tuple[Dict[Any, int], List[Set]]:
    """
    Apply Louvain algorithm for community detection.
    
    The Louvain algorithm is a greedy modularity optimization method:
    1. Phase 1: Local modularity optimization (move nodes to neighbors' communities)
    2. Phase 2: Network aggregation (contract communities to single nodes)
    3. Repeat until no improvement
    
    Theoretical Background:
    -----------------------
    Modularity Q = (1/2m) * sum[(A_ij - k_i*k_j/(2m)) * delta(c_i, c_j)]
    
    The algorithm greedily maximizes this objective. Resolution parameter
    controls community granularity (higher = smaller communities).
    
    For genealogical graphs:
    - Default resolution finds medium-sized family clusters
    - Higher resolution finds nuclear family units
    - Lower resolution finds extended family networks
    
    Parameters
    ----------
    G : nx.Graph
        Undirected graph (will convert if directed).
    resolution : float
        Resolution parameter. Higher values create more communities.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    Tuple[Dict[Any, int], List[Set]]
        - node_to_community: Mapping of node to community ID
        - communities: List of sets, each containing nodes in a community
    """
    if verbose:
        print("\n" + "=" * 60)
        print("LOUVAIN COMMUNITY DETECTION")
        print("=" * 60)
    
    if not LOUVAIN_AVAILABLE:
        raise ImportError("python-louvain package not installed. "
                         "Install with: pip install python-louvain")
    
    # Ensure undirected graph
    if G.is_directed():
        G_undirected = G.to_undirected()
        if verbose:
            print("Note: Converted directed graph to undirected for analysis.")
    else:
        G_undirected = G.copy()
    
    if verbose:
        print(f"Resolution parameter: {resolution}")
    
    # Apply Louvain algorithm
    node_to_community = community_louvain.best_partition(
        G_undirected, 
        resolution=resolution,
        random_state=42  # For reproducibility
    )
    
    # Convert to list of sets
    communities_list = dict_to_partition(node_to_community)
    
    if verbose:
        print(f"\nResults:")
        print(f"   - Number of communities: {len(communities_list)}")
        print(f"   - Largest community: {max(len(c) for c in communities_list)} nodes")
        print(f"   - Smallest community: {min(len(c) for c in communities_list)} nodes")
        
        # Community size distribution
        sizes = sorted([len(c) for c in communities_list], reverse=True)
        print(f"   - Community sizes (top 10): {sizes[:10]}")
        
        # Modularity
        mod = community_louvain.modularity(node_to_community, G_undirected)
        print(f"   - Modularity: {mod:.4f}")
        
        print("=" * 60)
    
    return node_to_community, communities_list


# ============================================================================
# LEIDEN ALGORITHM
# ============================================================================

def leiden_communities(G: nx.Graph,
                       resolution: float = 1.0,
                       verbose: bool = True) -> Tuple[Dict[Any, int], List[Set]]:
    """
    Apply Leiden algorithm for community detection.
    
    The Leiden algorithm improves upon Louvain by:
    1. Guaranteeing well-connected communities (no disconnected communities)
    2. Using a fast local move procedure
    3. Refining partitions to ensure quality
    
    Theoretical Background:
    -----------------------
    Leiden fixes a key issue in Louvain: communities can become disconnected
    during the aggregation phase. Leiden adds a refinement step that ensures
    all communities remain internally connected.
    
    For genealogical graphs:
    - Produces more coherent family groupings than Louvain
    - Better theoretical guarantees
    - Slightly slower but more accurate
    
    Parameters
    ----------
    G : nx.Graph
        Undirected graph (will convert if directed).
    resolution : float
        Resolution parameter. Higher values create more communities.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    Tuple[Dict[Any, int], List[Set]]
        - node_to_community: Mapping of node to community ID
        - communities: List of sets, each containing nodes in a community
    """
    if verbose:
        print("\n" + "=" * 60)
        print("LEIDEN COMMUNITY DETECTION")
        print("=" * 60)
    
    if not LEIDEN_AVAILABLE:
        raise ImportError("leidenalg/igraph packages not installed. "
                         "Install with: pip install leidenalg python-igraph")
    
    # Ensure undirected graph
    if G.is_directed():
        G_undirected = G.to_undirected()
        if verbose:
            print("Note: Converted directed graph to undirected for analysis.")
    else:
        G_undirected = G.copy()
    
    if verbose:
        print(f"Resolution parameter: {resolution}")
    
    # Convert NetworkX to iGraph
    # Create mapping of nodes to indices
    node_list = list(G_undirected.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Create edge list for igraph
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_undirected.edges()]
    
    # Create igraph Graph
    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    
    # Apply Leiden algorithm
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=42  # For reproducibility
    )
    
    # Convert back to NetworkX format
    node_to_community = {}
    for comm_id, community in enumerate(partition):
        for idx in community:
            node_to_community[node_list[idx]] = comm_id
    
    communities_list = dict_to_partition(node_to_community)
    
    if verbose:
        print(f"\nResults:")
        print(f"   - Number of communities: {len(communities_list)}")
        print(f"   - Largest community: {max(len(c) for c in communities_list)} nodes")
        print(f"   - Smallest community: {min(len(c) for c in communities_list)} nodes")
        
        # Community size distribution
        sizes = sorted([len(c) for c in communities_list], reverse=True)
        print(f"   - Community sizes (top 10): {sizes[:10]}")
        
        # Modularity (from Leiden)
        mod = partition.modularity
        print(f"   - Modularity: {mod:.4f}")
        
        print("=" * 60)
    
    return node_to_community, communities_list


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_modularity(G: nx.Graph, 
                         partition: Dict[Any, int]) -> float:
    """
    Calculate modularity for a given partition.
    
    Modularity Q measures the quality of a partition:
    Q = (1/2m) * sum[(A_ij - k_i*k_j/(2m)) * delta(c_i, c_j)]
    
    Interpretation:
    - Q > 0.3: Good community structure
    - Q > 0.5: Strong community structure
    - Q close to 0: No better than random
    
    Parameters
    ----------
    G : nx.Graph
        The graph (will convert to undirected if needed).
    partition : Dict[Any, int]
        Mapping of node to community ID.
    
    Returns
    -------
    float
        Modularity score.
    """
    if G.is_directed():
        G = G.to_undirected()
    
    communities = dict_to_partition(partition)
    return nx.community.modularity(G, communities)


def calculate_nmi(partition1: Dict[Any, int], 
                  partition2: Dict[Any, int]) -> float:
    """
    Calculate Normalized Mutual Information between two partitions.
    
    NMI measures the agreement between two clusterings, normalized
    by the entropy of each clustering.
    
    NMI(X,Y) = 2 * I(X;Y) / (H(X) + H(Y))
    
    Interpretation:
    - NMI = 0: Independent partitions
    - NMI = 1: Identical partitions
    - Higher = more similar partitions
    
    Parameters
    ----------
    partition1 : Dict[Any, int]
        First partition (node -> community).
    partition2 : Dict[Any, int]
        Second partition (node -> community).
    
    Returns
    -------
    float
        NMI score between 0 and 1.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. "
                         "Install with: pip install scikit-learn")
    
    # Get common nodes
    common_nodes = sorted(set(partition1.keys()) & set(partition2.keys()))
    
    if len(common_nodes) == 0:
        return 0.0
    
    # Get labels in same order
    labels1 = get_labels_from_partition(partition1, common_nodes)
    labels2 = get_labels_from_partition(partition2, common_nodes)
    
    return normalized_mutual_info_score(labels1, labels2)


def calculate_ari(partition1: Dict[Any, int], 
                  partition2: Dict[Any, int]) -> float:
    """
    Calculate Adjusted Rand Index between two partitions.
    
    ARI measures similarity between clusterings, corrected for chance.
    
    Interpretation:
    - ARI = 1: Perfect agreement
    - ARI = 0: Random agreement (chance level)
    - ARI < 0: Less than random agreement
    
    Parameters
    ----------
    partition1 : Dict[Any, int]
        First partition (node -> community).
    partition2 : Dict[Any, int]
        Second partition (node -> community).
    
    Returns
    -------
    float
        ARI score (typically between -0.5 and 1).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. "
                         "Install with: pip install scikit-learn")
    
    # Get common nodes
    common_nodes = sorted(set(partition1.keys()) & set(partition2.keys()))
    
    if len(common_nodes) == 0:
        return 0.0
    
    # Get labels in same order
    labels1 = get_labels_from_partition(partition1, common_nodes)
    labels2 = get_labels_from_partition(partition2, common_nodes)
    
    return adjusted_rand_score(labels1, labels2)


def compare_partitions(G: nx.Graph,
                       partition1: Dict[Any, int],
                       partition2: Dict[Any, int],
                       name1: str = "Partition 1",
                       name2: str = "Partition 2",
                       verbose: bool = True) -> Dict[str, float]:
    """
    Compare two partitions using multiple metrics.
    
    Parameters
    ----------
    G : nx.Graph
        The underlying graph.
    partition1 : Dict[Any, int]
        First partition.
    partition2 : Dict[Any, int]
        Second partition.
    name1 : str
        Name for first partition.
    name2 : str
        Name for second partition.
    verbose : bool
        Whether to print results.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with NMI, ARI, and modularity scores.
    """
    results = {
        f'{name1}_modularity': calculate_modularity(G, partition1),
        f'{name2}_modularity': calculate_modularity(G, partition2),
        'nmi': calculate_nmi(partition1, partition2),
        'ari': calculate_ari(partition1, partition2)
    }
    
    if verbose:
        print(f"\n{name1} vs {name2} Comparison:")
        print(f"   {name1} Modularity: {results[f'{name1}_modularity']:.4f}")
        print(f"   {name2} Modularity: {results[f'{name2}_modularity']:.4f}")
        print(f"   NMI: {results['nmi']:.4f}")
        print(f"   ARI: {results['ari']:.4f}")
    
    return results


# ============================================================================
# GENERATIONAL ANALYSIS PER COMMUNITY
# ============================================================================

def analyze_generational_depth_per_community(
    G: nx.Graph,
    partition: Dict[Any, int],
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze generational depth distribution within each community.
    
    This analysis helps understand whether communities represent:
    - Vertical lineages (multiple generations, narrow width)
    - Horizontal groupings (single generation, wide sibling groups)
    - Mixed structures (extended families spanning generations)
    
    Parameters
    ----------
    G : nx.Graph
        Graph with 'generation' node attribute.
    partition : Dict[Any, int]
        Mapping of node to community ID.
    verbose : bool
        Whether to print analysis.
    
    Returns
    -------
    Dict[int, Dict[str, Any]]
        For each community: generation counts, min, max, span, mean.
    """
    # Get generation attribute
    generations = nx.get_node_attributes(G, 'generation')
    
    if not generations:
        raise ValueError("Graph must have 'generation' node attribute. "
                        "Run compute_generational_depth() first.")
    
    # Analyze each community
    community_analysis = {}
    
    for node, comm_id in partition.items():
        if comm_id not in community_analysis:
            community_analysis[comm_id] = {
                'nodes': [],
                'generations': []
            }
        
        community_analysis[comm_id]['nodes'].append(node)
        gen = generations.get(node, -1)
        if gen >= 0:  # Only count valid generations
            community_analysis[comm_id]['generations'].append(gen)
    
    # Compute statistics for each community
    results = {}
    
    for comm_id, data in community_analysis.items():
        gen_list = data['generations']
        
        if gen_list:
            results[comm_id] = {
                'size': len(data['nodes']),
                'gen_counts': dict(Counter(gen_list)),
                'gen_min': min(gen_list),
                'gen_max': max(gen_list),
                'gen_span': max(gen_list) - min(gen_list) + 1,
                'gen_mean': np.mean(gen_list),
                'gen_std': np.std(gen_list) if len(gen_list) > 1 else 0
            }
        else:
            results[comm_id] = {
                'size': len(data['nodes']),
                'gen_counts': {},
                'gen_min': None,
                'gen_max': None,
                'gen_span': 0,
                'gen_mean': None,
                'gen_std': None
            }
    
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATIONAL DEPTH ANALYSIS PER COMMUNITY")
        print("=" * 60)
        
        # Sort by community size
        sorted_comms = sorted(results.items(), 
                              key=lambda x: x[1]['size'], 
                              reverse=True)
        
        print(f"\nTOP 10 COMMUNITIES BY SIZE:")
        print(f"{'Comm':<6} {'Size':<8} {'Gen Span':<10} {'Gen Range':<12} {'Mean Gen':<10}")
        print("-" * 50)
        
        for comm_id, stats in sorted_comms[:10]:
            if stats['gen_min'] is not None:
                gen_range = f"{stats['gen_min']}-{stats['gen_max']}"
                mean_gen = f"{stats['gen_mean']:.1f}"
            else:
                gen_range = "N/A"
                mean_gen = "N/A"
            
            print(f"{comm_id:<6} {stats['size']:<8} {stats['gen_span']:<10} "
                  f"{gen_range:<12} {mean_gen:<10}")
        
        # Summary statistics
        spans = [s['gen_span'] for s in results.values() if s['gen_span'] > 0]
        if spans:
            print(f"\nOVERALL GENERATIONAL SPAN:")
            print(f"   - Average span: {np.mean(spans):.2f} generations")
            print(f"   - Max span: {max(spans)} generations")
            print(f"   - Single-generation communities: {sum(1 for s in spans if s == 1)}")
            print(f"   - Multi-generation communities: {sum(1 for s in spans if s > 1)}")
        
        print("=" * 60)
    
    return results


def plot_generational_histogram(
    G: nx.Graph,
    partition: Dict[Any, int],
    top_n_communities: int = 6,
    save_path: Optional[str] = None
) -> None:
    """
    Plot generational depth histograms for top N communities.
    
    Parameters
    ----------
    G : nx.Graph
        Graph with 'generation' node attribute.
    partition : Dict[Any, int]
        Mapping of node to community ID.
    top_n_communities : int
        Number of largest communities to plot.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt
    
    generations = nx.get_node_attributes(G, 'generation')
    
    # Organize by community
    comm_gens = defaultdict(list)
    for node, comm_id in partition.items():
        gen = generations.get(node, -1)
        if gen >= 0:
            comm_gens[comm_id].append(gen)
    
    # Get top N by size
    sorted_comms = sorted(comm_gens.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)[:top_n_communities]
    
    # Create subplots
    n_plots = len(sorted_comms)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each community
    for i, (comm_id, gen_list) in enumerate(sorted_comms):
        ax = axes[i]
        
        gen_counts = Counter(gen_list)
        gens = sorted(gen_counts.keys())
        counts = [gen_counts[g] for g in gens]
        
        ax.bar(gens, counts, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count')
        ax.set_title(f'Community {comm_id}\n({len(gen_list)} members)')
        ax.set_xticks(gens)
    
    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Generational Depth Distribution per Community', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# ASSIGN COMMUNITY ATTRIBUTE TO GRAPH
# ============================================================================

def assign_community_attribute(
    G: nx.Graph,
    partition: Dict[Any, int],
    attribute_name: str = 'community'
) -> nx.Graph:
    """
    Add community assignment as a node attribute.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to modify.
    partition : Dict[Any, int]
        Mapping of node to community ID.
    attribute_name : str
        Name for the attribute.
    
    Returns
    -------
    nx.Graph
        Graph with community attribute added.
    """
    nx.set_node_attributes(G, partition, attribute_name)
    return G


# ============================================================================
# RUN ALL COMMUNITY DETECTION
# ============================================================================

def run_all_community_detection(
    G: nx.Graph,
    resolution: float = 1.0,
    verbose: bool = True
) -> Dict[str, Tuple[Dict[Any, int], List[Set], float]]:
    """
    Run all available community detection algorithms and compare results.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to analyze.
    resolution : float
        Resolution parameter for Louvain/Leiden.
    verbose : bool
        Whether to print progress and results.
    
    Returns
    -------
    Dict[str, Tuple[Dict[Any, int], List[Set], float]]
        For each algorithm: (partition_dict, communities_list, modularity)
    """
    results = {}
    
    # Girvan-Newman
    try:
        partition, communities = girvan_newman_communities(G, verbose=verbose)
        mod = calculate_modularity(G, partition)
        results['girvan_newman'] = (partition, communities, mod)
    except Exception as e:
        if verbose:
            print(f"Girvan-Newman failed: {e}")
    
    # Louvain
    if LOUVAIN_AVAILABLE:
        try:
            partition, communities = louvain_communities(G, resolution=resolution, verbose=verbose)
            mod = calculate_modularity(G, partition)
            results['louvain'] = (partition, communities, mod)
        except Exception as e:
            if verbose:
                print(f"Louvain failed: {e}")
    
    # Leiden
    if LEIDEN_AVAILABLE:
        try:
            partition, communities = leiden_communities(G, resolution=resolution, verbose=verbose)
            mod = calculate_modularity(G, partition)
            results['leiden'] = (partition, communities, mod)
        except Exception as e:
            if verbose:
                print(f"Leiden failed: {e}")
    
    # Summary comparison
    if verbose and len(results) > 1:
        print("\n" + "=" * 60)
        print("COMMUNITY DETECTION COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Algorithm':<20} {'Communities':<15} {'Modularity':<12}")
        print("-" * 50)
        
        for algo, (partition, communities, mod) in results.items():
            print(f"{algo:<20} {len(communities):<15} {mod:.4f}")
        
        # Pairwise comparisons
        algos = list(results.keys())
        if len(algos) >= 2 and SKLEARN_AVAILABLE:
            print(f"\nPAIRWISE COMPARISON (NMI / ARI):")
            for i in range(len(algos)):
                for j in range(i+1, len(algos)):
                    nmi = calculate_nmi(results[algos[i]][0], results[algos[j]][0])
                    ari = calculate_ari(results[algos[i]][0], results[algos[j]][0])
                    print(f"   {algos[i]} vs {algos[j]}: NMI={nmi:.4f}, ARI={ari:.4f}")
        
        print("=" * 60)
    
    return results


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Communities module loaded successfully.")
    print(f"Louvain available: {LOUVAIN_AVAILABLE}")
    print(f"Leiden available: {LEIDEN_AVAILABLE}")
    print(f"sklearn available: {SKLEARN_AVAILABLE}")
