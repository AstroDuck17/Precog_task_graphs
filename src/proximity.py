"""
proximity.py - Proximity Measures Module for MetaFam Knowledge Graph

This module implements proximity/similarity measures for finding
closest relatives within communities in the family knowledge graph.

Algorithms Implemented:
    1. Random Walk with Restarts (RWR): Stationary distribution of random walk
    2. Katz Index: Path-counting similarity with attenuation

Author: MetaFam Analysis Pipeline
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict


# ============================================================================
# RANDOM WALK WITH RESTARTS (RWR)
# ============================================================================

def random_walk_restart(
    G: nx.Graph,
    source: Any,
    restart_prob: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[Any, float]:
    """
    Compute Random Walk with Restarts proximity scores from a source node.
    
    RWR simulates a random walker that:
    1. At each step, with probability (1-c), follows a random edge
    2. With probability c, restarts at the source node
    
    The stationary distribution gives proximity scores: higher = more reachable.
    
    Theoretical Background:
    -----------------------
    RWR solves: p = (1-c) * W * p + c * e_s
    
    Where:
    - c = restart probability (typically 0.1-0.3)
    - W = column-normalized adjacency matrix (transition matrix)
    - e_s = source node indicator vector
    - p = stationary probability vector
    
    For genealogical graphs:
    - High RWR score = node is closely connected via many paths
    - Captures both path length and path diversity
    - Good for finding "close" relatives considering multiple connections
    
    Parameters
    ----------
    G : nx.Graph
        The graph (uses undirected edges for random walk).
    source : Any
        Source node to compute proximity from.
    restart_prob : float
        Probability of restarting at source (0 < c < 1). Default 0.15.
        Higher = more weight on local structure, lower = more global.
    max_iter : int
        Maximum iterations for convergence.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print convergence info.
    
    Returns
    -------
    Dict[Any, float]
        Mapping of each node to its RWR proximity score.
        Scores sum to 1 (probability distribution).
    
    Examples
    --------
    >>> scores = random_walk_restart(G, 'Alice', restart_prob=0.15)
    >>> closest = sorted(scores.items(), key=lambda x: -x[1])[:10]
    """
    if source not in G:
        raise ValueError(f"Source node '{source}' not in graph")
    
    # Use undirected version for random walk
    if G.is_directed():
        G = G.to_undirected()
    
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    source_idx = node_to_idx[source]
    
    # Initialize probability vector
    p = np.zeros(n)
    p[source_idx] = 1.0
    
    # Restart vector
    e_s = np.zeros(n)
    e_s[source_idx] = 1.0
    
    c = restart_prob
    
    # Power iteration
    for iteration in range(max_iter):
        p_new = np.zeros(n)
        
        # Random walk step
        for u_idx, u in enumerate(nodes):
            if p[u_idx] > 0:
                neighbors = list(G.neighbors(u))
                if neighbors:
                    # Uniform transition probability
                    transition_prob = p[u_idx] * (1 - c) / len(neighbors)
                    for v in neighbors:
                        v_idx = node_to_idx[v]
                        p_new[v_idx] += transition_prob
        
        # Restart component
        p_new += c * e_s
        
        # Check convergence
        diff = np.sum(np.abs(p_new - p))
        
        if verbose and (iteration + 1) % 20 == 0:
            print(f"   Iteration {iteration + 1}: diff = {diff:.8f}")
        
        if diff < tol:
            if verbose:
                print(f"   Converged at iteration {iteration + 1}")
            break
        
        p = p_new
    
    # Convert back to dict
    return {nodes[i]: p[i] for i in range(n)}


def rwr_personalized_pagerank(
    G: nx.Graph,
    source: Any,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[Any, float]:
    """
    Compute RWR using NetworkX's personalized PageRank (more efficient).
    
    This is mathematically equivalent to RWR where:
    - alpha = (1 - restart_prob)
    - So restart_prob = (1 - alpha)
    
    Parameters
    ----------
    G : nx.Graph
        The graph.
    source : Any
        Source node.
    alpha : float
        Damping factor (1 - restart_prob). Default 0.85.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    Dict[Any, float]
        Proximity scores.
    """
    if source not in G:
        raise ValueError(f"Source node '{source}' not in graph")
    
    # Use undirected version
    if G.is_directed():
        G = G.to_undirected()
    
    # Personalization vector
    personalization = {node: 0.0 for node in G.nodes()}
    personalization[source] = 1.0
    
    return nx.pagerank(
        G, 
        alpha=alpha, 
        personalization=personalization,
        max_iter=max_iter,
        tol=tol
    )


# ============================================================================
# KATZ INDEX
# ============================================================================

def katz_similarity(
    G: nx.Graph,
    source: Any,
    beta: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[Any, float]:
    """
    Compute Katz similarity scores from a source node.
    
    Katz index counts all paths between nodes, weighted by path length:
    K(s,t) = sum_{l=1}^{inf} beta^l * |paths of length l from s to t|
    
    Theoretical Background:
    -----------------------
    Katz similarity = (I - beta*A)^(-1) - I
    
    Where:
    - beta = attenuation factor (must be < 1/lambda_max)
    - A = adjacency matrix
    - Shorter paths contribute more (beta^l decays with length)
    
    For genealogical graphs:
    - Captures path diversity (multiple paths = higher similarity)
    - Siblings share many indirect paths through parents
    - Cousins share paths through grandparents
    - Good for detecting close family even without direct edge
    
    Parameters
    ----------
    G : nx.Graph
        The graph.
    source : Any
        Source node to compute similarity from.
    beta : float
        Attenuation factor. Must be less than 1/max_eigenvalue.
        Typical values: 0.001 - 0.1. Default 0.1.
        Higher = longer paths matter more.
    max_iter : int
        Maximum iterations for power method.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print convergence info.
    
    Returns
    -------
    Dict[Any, float]
        Mapping of each node to its Katz similarity score.
        Higher = more/shorter paths from source.
    
    Notes
    -----
    If beta is too high (> 1/max_eigenvalue), the series diverges.
    The function will warn but continue with a clipped beta.
    """
    if source not in G:
        raise ValueError(f"Source node '{source}' not in graph")
    
    # Use undirected version
    if G.is_directed():
        G = G.to_undirected()
    
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    source_idx = node_to_idx[source]
    
    # Check beta validity (should be < 1/lambda_max)
    # For efficiency, use approximation: lambda_max <= max_degree
    max_degree = max(dict(G.degree()).values())
    beta_limit = 1.0 / max_degree
    
    if beta >= beta_limit:
        import warnings
        warnings.warn(f"beta={beta} may be too high (limit approx {beta_limit:.4f}). "
                     f"Results may be unstable.")
    
    # Initialize
    # Katz(s,t) = sum_{l=1}^inf beta^l * A^l[s,t]
    # We compute iteratively: score_l = beta * A * score_{l-1}, then accumulate
    
    # Current power of A applied to source indicator
    curr = np.zeros(n)
    curr[source_idx] = 1.0
    
    # Accumulated Katz scores
    katz_scores = np.zeros(n)
    
    for iteration in range(1, max_iter + 1):
        # Compute beta * A * curr
        next_iter = np.zeros(n)
        for u_idx, u in enumerate(nodes):
            if curr[u_idx] > 0:
                for v in G.neighbors(u):
                    v_idx = node_to_idx[v]
                    next_iter[v_idx] += beta * curr[u_idx]
        
        # Accumulate
        katz_scores += next_iter
        
        # Check convergence
        total_change = np.sum(next_iter)
        
        if verbose and iteration % 20 == 0:
            print(f"   Iteration {iteration}: total new contribution = {total_change:.8f}")
        
        if total_change < tol:
            if verbose:
                print(f"   Converged at iteration {iteration}")
            break
        
        curr = next_iter
    
    # Convert to dict
    return {nodes[i]: katz_scores[i] for i in range(n)}


def katz_networkx(
    G: nx.Graph,
    source: Any,
    alpha: float = 0.1,
    beta: float = 1.0
) -> Dict[Any, float]:
    """
    Compute Katz centrality using NetworkX (efficient implementation).
    
    Note: NetworkX katz_centrality computes global Katz centrality,
    not pairwise from a source. For source-specific, use katz_similarity().
    
    This function returns global Katz centrality for reference.
    
    Parameters
    ----------
    G : nx.Graph
        The graph.
    source : Any
        Not used (NetworkX computes global centrality).
    alpha : float
        Attenuation factor.
    beta : float
        Weight of exogenous factor.
    
    Returns
    -------
    Dict[Any, float]
        Global Katz centrality scores.
    """
    if G.is_directed():
        G = G.to_undirected()
    
    # NetworkX Katz centrality
    return nx.katz_centrality(G, alpha=alpha, beta=beta)


# ============================================================================
# FIND CLOSEST RELATIVES
# ============================================================================

def find_closest_relatives(
    G: nx.Graph,
    target_node: Any,
    community_nodes: Optional[Set[Any]] = None,
    method: str = 'rwr',
    top_k: int = 10,
    exclude_self: bool = True,
    verbose: bool = True,
    **kwargs
) -> List[Tuple[Any, float]]:
    """
    Find the closest relatives to a target node using proximity measures.
    
    This function computes proximity scores from the target node and
    returns the top-k closest nodes (optionally filtered to a community).
    
    Parameters
    ----------
    G : nx.Graph
        The graph with family relations.
    target_node : Any
        The node to find relatives for.
    community_nodes : Set[Any], optional
        If provided, only consider nodes in this set.
        Useful for finding closest relatives within a community.
    method : str
        Proximity method to use:
        - 'rwr': Random Walk with Restarts (default)
        - 'katz': Katz Index
    top_k : int
        Number of closest relatives to return.
    exclude_self : bool
        Whether to exclude the target node from results.
    verbose : bool
        Whether to print results.
    **kwargs
        Additional arguments passed to the proximity function.
        For RWR: restart_prob (default 0.15)
        For Katz: beta (default 0.1)
    
    Returns
    -------
    List[Tuple[Any, float]]
        List of (node, proximity_score) tuples, sorted by score descending.
    
    Examples
    --------
    >>> # Find 10 closest relatives within community 5
    >>> closest = find_closest_relatives(G, 'Alice', 
    ...                                   community_nodes=communities[5],
    ...                                   method='rwr')
    """
    if target_node not in G:
        raise ValueError(f"Target node '{target_node}' not in graph")
    
    if verbose:
        print(f"\nFinding closest relatives to '{target_node}' using {method.upper()}...")
    
    # Compute proximity scores
    if method.lower() == 'rwr':
        restart_prob = kwargs.get('restart_prob', 0.15)
        scores = rwr_personalized_pagerank(G, target_node, alpha=1-restart_prob)
    elif method.lower() == 'katz':
        beta = kwargs.get('beta', 0.1)
        scores = katz_similarity(G, target_node, beta=beta, verbose=kwargs.get('verbose', False))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rwr' or 'katz'.")
    
    # Filter to community if specified
    if community_nodes is not None:
        scores = {node: score for node, score in scores.items() 
                  if node in community_nodes}
    
    # Exclude self if requested
    if exclude_self and target_node in scores:
        del scores[target_node]
    
    # Sort by score and get top-k
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if verbose:
        print(f"\nTop {top_k} closest relatives to '{target_node}':")
        print(f"{'Rank':<6} {'Node':<30} {'Score':<15}")
        print("-" * 55)
        
        # Get generation info if available
        generations = nx.get_node_attributes(G, 'generation')
        
        for i, (node, score) in enumerate(sorted_nodes, 1):
            gen = generations.get(node, 'N/A')
            print(f"{i:<6} {str(node):<30} {score:.8f} (Gen: {gen})")
    
    return sorted_nodes


# ============================================================================
# PROXIMITY ANALYSIS FOR MULTIPLE NODE TYPES
# ============================================================================

def analyze_proximity_for_node_types(
    G: nx.Graph,
    partition: Dict[Any, int],
    n_samples: int = 3,
    top_k: int = 5,
    method: str = 'rwr',
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    """
    Analyze proximity for different node types (by generation).
    
    Selects representative nodes from different generations:
    1. High-generation ancestor (generation 0)
    2. Mid-generation node (generation 1-2)
    3. Low-generation descendant (generation 3+)
    
    For each, finds closest relatives within their community.
    
    Parameters
    ----------
    G : nx.Graph
        Graph with 'generation' attribute.
    partition : Dict[Any, int]
        Community assignments.
    n_samples : int
        Number of nodes to sample per generation type.
    top_k : int
        Number of closest relatives to find for each.
    method : str
        Proximity method ('rwr' or 'katz').
    verbose : bool
        Whether to print detailed results.
    
    Returns
    -------
    Dict[str, List[Dict]]
        Results organized by generation type:
        - 'ancestors': List of analysis results for gen 0
        - 'mid_generation': List for gen 1-2
        - 'descendants': List for gen 3+
    """
    generations = nx.get_node_attributes(G, 'generation')
    
    if not generations:
        raise ValueError("Graph must have 'generation' node attribute.")
    
    # Categorize nodes by generation
    ancestors = [n for n, g in generations.items() if g == 0]
    mid_gen = [n for n, g in generations.items() if g in [1, 2]]
    descendants = [n for n, g in generations.items() if g >= 3]
    
    results = {
        'ancestors': [],
        'mid_generation': [],
        'descendants': []
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("PROXIMITY ANALYSIS FOR DIFFERENT NODE TYPES")
        print("=" * 70)
    
    # Sample and analyze
    import random
    random.seed(42)  # Reproducibility
    
    def analyze_samples(node_list: List[Any], category: str, n: int):
        if not node_list:
            if verbose:
                print(f"\nNo nodes found for {category}")
            return []
        
        # Sample up to n nodes
        sampled = random.sample(node_list, min(n, len(node_list)))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING {category.upper()} (Generation: {category})")
            print(f"{'='*60}")
        
        category_results = []
        
        for node in sampled:
            comm_id = partition.get(node)
            if comm_id is None:
                continue
            
            # Get community members
            community_nodes = {n for n, c in partition.items() if c == comm_id}
            
            if verbose:
                print(f"\nNode: {node}")
                print(f"   Generation: {generations.get(node)}")
                print(f"   Community: {comm_id} ({len(community_nodes)} members)")
            
            # Find closest relatives within community
            closest = find_closest_relatives(
                G, node,
                community_nodes=community_nodes,
                method=method,
                top_k=top_k,
                verbose=False
            )
            
            if verbose:
                print(f"   Closest relatives (within community):")
                for i, (rel, score) in enumerate(closest, 1):
                    rel_gen = generations.get(rel, 'N/A')
                    print(f"      {i}. {rel} (Gen: {rel_gen}, Score: {score:.6f})")
            
            category_results.append({
                'node': node,
                'generation': generations.get(node),
                'community': comm_id,
                'closest_relatives': closest
            })
        
        return category_results
    
    results['ancestors'] = analyze_samples(ancestors, 'ancestors (Gen 0)', n_samples)
    results['mid_generation'] = analyze_samples(mid_gen, 'mid_generation (Gen 1-2)', n_samples)
    results['descendants'] = analyze_samples(descendants, 'descendants (Gen 3+)', n_samples)
    
    if verbose:
        print("\n" + "=" * 70)
    
    return results


# ============================================================================
# COMPARE PROXIMITY METHODS
# ============================================================================

def compare_proximity_methods(
    G: nx.Graph,
    target_node: Any,
    community_nodes: Optional[Set[Any]] = None,
    top_k: int = 10,
    verbose: bool = True
) -> Dict[str, List[Tuple[Any, float]]]:
    """
    Compare RWR and Katz proximity results for a target node.
    
    Parameters
    ----------
    G : nx.Graph
        The graph.
    target_node : Any
        Node to analyze.
    community_nodes : Set[Any], optional
        Restrict to community.
    top_k : int
        Number of results.
    verbose : bool
        Print comparison.
    
    Returns
    -------
    Dict[str, List[Tuple[Any, float]]]
        Results for each method.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROXIMITY METHOD COMPARISON FOR '{target_node}'")
        print(f"{'='*60}")
    
    # RWR
    rwr_results = find_closest_relatives(
        G, target_node, community_nodes,
        method='rwr', top_k=top_k, verbose=False
    )
    
    # Katz
    katz_results = find_closest_relatives(
        G, target_node, community_nodes,
        method='katz', top_k=top_k, verbose=False
    )
    
    if verbose:
        print(f"\n{'Rank':<6} {'RWR Result':<25} {'Katz Result':<25}")
        print("-" * 60)
        
        for i in range(top_k):
            rwr_node = rwr_results[i][0] if i < len(rwr_results) else "N/A"
            katz_node = katz_results[i][0] if i < len(katz_results) else "N/A"
            print(f"{i+1:<6} {str(rwr_node):<25} {str(katz_node):<25}")
        
        # Overlap analysis
        rwr_nodes = set(n for n, _ in rwr_results)
        katz_nodes = set(n for n, _ in katz_results)
        overlap = rwr_nodes & katz_nodes
        
        print(f"\nOverlap: {len(overlap)}/{top_k} nodes ({100*len(overlap)/top_k:.0f}%)")
    
    return {
        'rwr': rwr_results,
        'katz': katz_results
    }


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Proximity module loaded successfully.")
    print("\nAvailable functions:")
    print("  - random_walk_restart(G, source, restart_prob=0.15)")
    print("  - katz_similarity(G, source, beta=0.1)")
    print("  - find_closest_relatives(G, target, community_nodes, method='rwr')")
    print("  - analyze_proximity_for_node_types(G, partition)")
    print("  - compare_proximity_methods(G, target)")
