"""
Exploration Module
==================

This module provides comprehensive network analysis functions for the MetaFam
knowledge graph, including global metrics, node feature calculation, gender
inference via rule-based classification, and Gephi export functionality.

Theoretical Foundation:
-----------------------

1. GLOBAL METRICS (Macroscopic Graph Properties):
   
   - Density: Ratio of actual edges to possible edges. For directed graphs:
     density = |E| / (|V| * (|V| - 1))
     In family graphs, low density is expected as each person only connects
     to a limited set of relatives.
   
   - Weakly Connected Components (WCC): Number of disjoint subgraphs where
     nodes are connected ignoring edge direction. Each WCC represents a
     separate family tree in the dataset.
   
   - Clustering Coefficient: Measures transitivity (if A→B and B→C, does A→C?).
     In family graphs, this measures indirect family connections through
     shared relatives.

2. NODE FEATURES (Microscopic Node Properties):
   
   - In-Degree: Number of incoming edges. High in-degree indicates many
     relations point TO this person (e.g., "X is grandchildOf many people").
   
   - Out-Degree: Number of outgoing edges. High out-degree indicates many
     relations originate FROM this person (ancestors have higher out-degree
     as they are "parentOf", "grandparentOf" many descendants).
   
   - Betweenness Centrality: Measures how often a node lies on shortest paths
     between other nodes. Identifies "bridge" individuals connecting different
     family branches (often through marriage).
   
   - Degree Centrality: Normalized degree measure. Identifies highly connected
     individuals (patriarchs/matriarchs).
   
   - Generational Depth: Distance from family tree root. Computed separately
     for each connected component to handle multiple family trees.

3. GENDER INFERENCE:
   Deterministic rule-based classification using relationship semantics.
   Male relations (fatherOf, brotherOf, etc.) imply male HEAD node.
   Female relations (motherOf, sisterOf, etc.) imply female HEAD node.

Author: MetaFam Analysis Team
"""

import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path


# ============================================================================
# GENDER CLASSIFICATION CONSTANTS
# ============================================================================

MALE_RELATIONS = frozenset([
    'fatherOf', 'brotherOf', 'sonOf', 'uncleOf', 'husbandOf', 'grandfatherOf',
    'nephewOf', 'boyCousinOf', 'grandsonOf', 'greatUncleOf', 'greatGrandfatherOf',
    'secondUncleOf', 'greatGrandsonOf', 'secondNephewOf',
    # Additional relations from actual dataset
    'boyFirstCousinOnceRemovedOf', 'boySecondCousinOf'
])

FEMALE_RELATIONS = frozenset([
    'motherOf', 'sisterOf', 'daughterOf', 'auntOf', 'wifeOf', 'grandmotherOf',
    'nieceOf', 'girlCousinOf', 'granddaughterOf', 'greatAuntOf', 'greatGrandmotherOf',
    'secondAuntOf', 'greatGranddaughterOf', 'secondNieceOf',
    # Additional relations from actual dataset
    'girlFirstCousinOnceRemovedOf', 'girlSecondCousinOf'
])

# Parental relations for generational depth calculation
PARENTAL_RELATIONS = frozenset([
    'fatherOf', 'motherOf', 'parentOf'
])

# All ancestor-to-descendant relations for generation calculation
ANCESTOR_RELATIONS = frozenset([
    'fatherOf', 'motherOf', 'parentOf',
    'grandfatherOf', 'grandmotherOf', 'grandparentOf',
    'greatGrandfatherOf', 'greatGrandmotherOf', 'greatGrandparentOf'
])


# ============================================================================
# GLOBAL METRICS (Group A - Print/Report Only)
# ============================================================================

def calculate_global_metrics(G_directed: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate global (macroscopic) network metrics for the directed graph.
    
    These metrics provide an overview of the entire network structure and
    are used for reporting purposes, not stored as node attributes.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed family knowledge graph.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'density': Graph density (float)
        - 'num_weakly_connected_components': Number of WCCs (int)
        - 'largest_wcc_size': Size of the largest WCC (int)
        - 'avg_clustering': Average clustering coefficient (float)
        - 'num_nodes': Total number of nodes (int)
        - 'num_edges': Total number of edges (int)
    
    Theoretical Notes
    -----------------
    1. Density: For a directed graph D = |E| / (|V| * (|V| - 1))
       Family graphs typically have very low density (~0.001) because
       social networks follow a sparse connectivity pattern.
    
    2. WCCs: In genealogy, each WCC represents an isolated family tree.
       Multiple WCCs suggest either data completeness issues or genuinely
       separate family lineages.
    
    3. Clustering: Measures the probability that two neighbors of a node
       are themselves connected. In family graphs, this reflects the
       closure of familial relationships (e.g., siblings sharing parents).
    """
    
    metrics = {}
    
    # Basic counts
    metrics['num_nodes'] = G_directed.number_of_nodes()
    metrics['num_edges'] = G_directed.number_of_edges()
    
    # Density calculation
    # For directed graphs: density = m / (n * (n-1)) where m=edges, n=nodes
    metrics['density'] = nx.density(G_directed)
    
    # Weakly Connected Components analysis
    # A WCC treats the graph as undirected for connectivity
    wccs = list(nx.weakly_connected_components(G_directed))
    metrics['num_weakly_connected_components'] = len(wccs)
    metrics['largest_wcc_size'] = max(len(wcc) for wcc in wccs) if wccs else 0
    metrics['smallest_wcc_size'] = min(len(wcc) for wcc in wccs) if wccs else 0
    
    # WCC size distribution
    wcc_sizes = [len(wcc) for wcc in wccs]
    metrics['wcc_sizes'] = sorted(wcc_sizes, reverse=True)
    metrics['avg_wcc_size'] = np.mean(wcc_sizes) if wcc_sizes else 0
    
    # Clustering coefficient
    # For directed graphs, we compute the average over all nodes
    # Note: Uses the generalized definition for directed graphs
    try:
        metrics['avg_clustering'] = nx.average_clustering(G_directed)
    except:
        # Fallback for graphs where clustering can't be computed
        metrics['avg_clustering'] = 0.0
    
    # Additional useful metrics
    if metrics['num_nodes'] > 0:
        metrics['avg_degree'] = metrics['num_edges'] / metrics['num_nodes']
    else:
        metrics['avg_degree'] = 0
    
    return metrics


def print_global_metrics(metrics: Dict[str, Any]) -> None:
    """
    Pretty-print global metrics for reporting.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of metrics from calculate_global_metrics()
    """
    print("\n" + "=" * 60)
    print("GLOBAL NETWORK METRICS REPORT")
    print("=" * 60)
    
    print("\nBASIC STATISTICS:")
    print(f"   - Total Nodes (People):     {metrics['num_nodes']:,}")
    print(f"   - Total Edges (Relations):  {metrics['num_edges']:,}")
    print(f"   - Average Degree:           {metrics['avg_degree']:.2f}")
    
    print("\nSTRUCTURAL METRICS:")
    print(f"   - Graph Density:            {metrics['density']:.6f}")
    print(f"     Interpretation: Very sparse (typical for social/family networks)")
    
    print(f"\n   - Average Clustering Coef:  {metrics['avg_clustering']:.4f}")
    print(f"     Measures transitivity of family relationships")
    
    print("\nCONNECTIVITY ANALYSIS:")
    print(f"   - Weakly Connected Components: {metrics['num_weakly_connected_components']}")
    print(f"   - Largest Component Size:      {metrics['largest_wcc_size']:,} nodes")
    print(f"   - Smallest Component Size:     {metrics['smallest_wcc_size']:,} nodes")
    print(f"   - Average Component Size:      {metrics['avg_wcc_size']:.1f} nodes")
    
    # Show component size distribution if multiple components
    if metrics['num_weakly_connected_components'] > 1:
        print(f"\n   Component Size Distribution (top 10):")
        for i, size in enumerate(metrics['wcc_sizes'][:10], 1):
            print(f"      {i}. {size:,} nodes")
    
    print("\n" + "=" * 60)


# ============================================================================
# NODE FEATURES (Group B - Store as Node Attributes)
# ============================================================================

def calculate_node_features(G_directed: nx.DiGraph, verbose: bool = True) -> nx.DiGraph:
    """
    Calculate and store node-level features as graph attributes.
    
    This function computes degree measures, centrality metrics, and
    generational depth for each node, storing them as node attributes
    for later analysis and Gephi export.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed family knowledge graph.
    verbose : bool, default=True
        Whether to print progress and summary statistics.
    
    Returns
    -------
    nx.DiGraph
        The same graph with added node attributes:
        - 'in_degree': Number of incoming edges
        - 'out_degree': Number of outgoing edges
        - 'pagerank': PageRank centrality score
        - 'generation': Generational depth (0 = root ancestor)
        
        Note: Betweenness centrality is computed for analysis but
        NOT stored as an attribute to reduce export size.
    
    Theoretical Notes
    -----------------
    1. In-Degree vs Out-Degree Analysis:
       In family knowledge graphs with directional relations like "fatherOf":
       - High OUT-degree: Ancestors (have many "childOf" relations pointing out)
       - High IN-degree: Descendants (many "parentOf" relations point to them)
       
       This asymmetry is fundamental to genealogical graph structure.
    
    2. PageRank Centrality:
       PR(v) = (1-d)/N + d * Σ(PR(u)/out_degree(u)) for all u pointing to v
       where d = damping factor (0.85) and N = number of nodes.
       
       In family graphs, high PageRank indicates:
       - Important ancestors with many descendants pointing to them
       - Nodes that are "referenced" by many high-quality connections
       - Historical significance in the family tree
    
    3. Betweenness Centrality (Analysis Only):
       B(v) = Σ(s≠v≠t) σst(v) / σst
       where σst is the number of shortest paths from s to t.
       
       Computed for analysis but not stored as attribute.
       In family graphs, high betweenness often indicates:
       - Marriage links connecting two families
       - Individuals bridging generations
       - Key ancestors connecting many descendants
    
    4. Generational Depth Algorithm:
       Since the graph may contain multiple disconnected family trees,
       we cannot assume a single global root. The algorithm:
       a) Identify root nodes (in-degree = 0 for parental relations) per component
       b) Run BFS from roots to assign generation levels
       c) Handle cycles gracefully (shouldn't exist in proper family trees)
    """
    
    if verbose:
        print("\n" + "=" * 60)
        print("CALCULATING NODE FEATURES")
        print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. DEGREE METRICS
    # -------------------------------------------------------------------------
    if verbose:
        print("\nComputing degree metrics...")
    
    in_degrees = dict(G_directed.in_degree())
    out_degrees = dict(G_directed.out_degree())
    
    # Store as node attributes
    nx.set_node_attributes(G_directed, in_degrees, 'in_degree')
    nx.set_node_attributes(G_directed, out_degrees, 'out_degree')
    
    if verbose:
        in_deg_values = list(in_degrees.values())
        out_deg_values = list(out_degrees.values())
        print(f"   - In-Degree:    min={min(in_deg_values)}, max={max(in_deg_values)}, "
              f"mean={np.mean(in_deg_values):.2f}")
        print(f"   - Out-Degree:   min={min(out_deg_values)}, max={max(out_deg_values)}, "
              f"mean={np.mean(out_deg_values):.2f}")
    
    # -------------------------------------------------------------------------
    # 2. CENTRALITY METRICS
    # -------------------------------------------------------------------------
    if verbose:
        print("\nComputing centrality metrics...")
    
    # PageRank Centrality - measures node importance based on incoming link quality
    pagerank = nx.pagerank(G_directed, alpha=0.85)
    nx.set_node_attributes(G_directed, pagerank, 'pagerank')
    
    # Betweenness Centrality - computed for analysis only, NOT stored as attribute
    betweenness = nx.betweenness_centrality(G_directed)
    # Note: betweenness not stored as attribute to reduce export size
    
    if verbose:
        pr_values = list(pagerank.values())
        bc_values = list(betweenness.values())
        print(f"   - PageRank: max={max(pr_values):.6f}, mean={np.mean(pr_values):.6f}")
        print(f"   - Betweenness: max={max(bc_values):.4f}, mean={np.mean(bc_values):.4f} (analysis only)")
        
        # Find top 5 by PageRank (important ancestors)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   Top 5 nodes by PageRank (Important Ancestors):")
        for node, score in top_pagerank:
            print(f"      - {node}: {score:.6f}")
        
        # Find top 5 by betweenness (family bridges)
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   Top 5 nodes by Betweenness (Family Bridges):")
        for node, score in top_betweenness:
            print(f"      - {node}: {score:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. GENERATIONAL DEPTH (Topology-Aware)
    # -------------------------------------------------------------------------
    if verbose:
        print("\nComputing generational depth...")
    
    generation_depth = compute_generational_depth(G_directed, verbose=verbose)
    nx.set_node_attributes(G_directed, generation_depth, 'generation')
    
    if verbose:
        gen_values = [v for v in generation_depth.values() if v >= 0]
        if gen_values:
            print(f"   - Generation range: {min(gen_values)} to {max(gen_values)}")
            print(f"   - Mean generation: {np.mean(gen_values):.2f}")
            
            # Generation distribution
            gen_counts = Counter(generation_depth.values())
            print(f"\n   Generation Distribution:")
            for gen in sorted(gen_counts.keys()):
                if gen >= 0:
                    print(f"      Gen {gen}: {gen_counts[gen]:,} people")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Node features successfully added to graph.")
        print("=" * 60)
    
    return G_directed


def compute_generational_depth(G_directed: nx.DiGraph, verbose: bool = True) -> Dict[str, int]:
    """
    Compute generational depth for each node, handling multiple connected components.
    
    This algorithm identifies "root" ancestors (those with no incoming parental
    relations) and computes the shortest path distance from roots to all other
    nodes, representing generational depth.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed family graph.
    verbose : bool
        Whether to print progress information.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping node IDs to their generation level.
        - Generation 0: Root ancestors (no parents in data)
        - Generation 1: Children of roots
        - Generation N: N generations from nearest root
        - Generation -1: Could not be reached (data issue)
    
    Algorithm
    ---------
    1. Build a parental subgraph containing only parent→child edges
    2. For each weakly connected component:
       a. Find local roots (in-degree = 0)
       b. BFS from roots to assign generation levels
    3. Handle edge cases (isolated nodes, cycles, etc.)
    """
    
    # Initialize all nodes with generation -1 (unassigned)
    generation = {node: -1 for node in G_directed.nodes()}
    
    # Build parental relation subgraph for generation computation
    # We create weighted edges: direct parent=1, grandparent=2, great-grandparent=3
    parent_child_edges = []  # (parent, child, generation_diff)
    
    for u, v, data in G_directed.edges(data=True):
        relation = data.get('relation', '')
        if isinstance(relation, list):
            relations = relation
        else:
            relations = [relation]
        
        for rel in relations:
            rel_lower = rel.lower()
            
            # Great-grandparent relations (3 generations)
            if any(term in rel_lower for term in ['greatgrandfatherof', 'greatgrandmotherof', 'greatgrandparentof']):
                parent_child_edges.append((u, v, 3))
            # Grandparent relations (2 generations)
            elif any(term in rel_lower for term in ['grandfatherof', 'grandmotherof', 'grandparentof']):
                parent_child_edges.append((u, v, 2))
            # Parent-to-child direction relations (1 generation)
            elif any(term in rel_lower for term in ['fatherof', 'motherof', 'parentof']):
                parent_child_edges.append((u, v, 1))
    
    # Create parental subgraph with generation weights
    G_parental = nx.DiGraph()
    G_parental.add_nodes_from(G_directed.nodes())
    
    for parent, child, gen_diff in parent_child_edges:
        if G_parental.has_edge(parent, child):
            # Keep the smallest generation difference
            existing = G_parental[parent][child].get('weight', float('inf'))
            G_parental[parent][child]['weight'] = min(existing, gen_diff)
        else:
            G_parental.add_edge(parent, child, weight=gen_diff)
    
    if verbose:
        print(f"   - Parental subgraph: {G_parental.number_of_edges()} parent->descendant edges")
    
    # Process each weakly connected component separately
    wccs = list(nx.weakly_connected_components(G_parental))
    
    for component in wccs:
        subgraph = G_parental.subgraph(component)
        
        # Find root nodes (in-degree = 0 in parental subgraph)
        roots = [node for node in component if subgraph.in_degree(node) == 0]
        
        if not roots:
            # No clear roots - might be a cycle or isolated component
            # Find the node that has the most outgoing parental relations
            node_out_degrees = {n: subgraph.out_degree(n) for n in component}
            if node_out_degrees:
                best_root = max(node_out_degrees.items(), key=lambda x: x[1])[0]
                roots = [best_root]
        
        # Dijkstra-style BFS from roots to assign generations
        # (using edge weights for multi-generation relations)
        for root in roots:
            if generation[root] == -1:  # Not yet assigned
                generation[root] = 0
                
                # Priority queue: (generation, node)
                import heapq
                pq = [(0, root)]
                visited = set()
                
                while pq:
                    current_gen, current = heapq.heappop(pq)
                    
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Update generation (take minimum if already set)
                    if generation[current] == -1:
                        generation[current] = current_gen
                    else:
                        generation[current] = min(generation[current], current_gen)
                    
                    # Explore neighbors
                    for neighbor in subgraph.neighbors(current):
                        if neighbor not in visited:
                            edge_weight = subgraph[current][neighbor].get('weight', 1)
                            new_gen = current_gen + edge_weight
                            heapq.heappush(pq, (new_gen, neighbor))
    
    # Handle nodes not reached by parental traversal
    # These might only appear in non-parental relations (sibling, cousin, etc.)
    unassigned = [n for n, g in generation.items() if g == -1]
    if unassigned and verbose:
        print(f"   - Nodes without parental links: {len(unassigned)}")
        print(f"     These nodes only appear in non-blood relations (siblings, cousins, etc.)")
    
    # For unassigned nodes, try to infer from neighbors using all relations
    for node in unassigned:
        neighbor_gens = []
        
        # Check outgoing edges
        for neighbor in G_directed.neighbors(node):
            if generation.get(neighbor, -1) >= 0:
                # Get the relation to understand generation offset
                edge_data = G_directed.get_edge_data(node, neighbor)
                rel = edge_data.get('relation', '') if edge_data else ''
                if isinstance(rel, list):
                    rel = rel[0] if rel else ''
                
                rel_lower = rel.lower()
                
                # Sibling relations = same generation
                if any(term in rel_lower for term in ['brotherof', 'sisterof', 'siblingof']):
                    neighbor_gens.append(generation[neighbor])
                # Parent relations = one generation less
                elif any(term in rel_lower for term in ['fatherof', 'motherof', 'parentof']):
                    neighbor_gens.append(generation[neighbor] - 1)
                # Child relations = one generation more
                elif any(term in rel_lower for term in ['sonof', 'daughterof', 'childof']):
                    neighbor_gens.append(generation[neighbor] + 1)
                else:
                    # Default: assume same generation
                    neighbor_gens.append(generation[neighbor])
        
        # Check incoming edges
        for neighbor in G_directed.predecessors(node):
            if generation.get(neighbor, -1) >= 0:
                edge_data = G_directed.get_edge_data(neighbor, node)
                rel = edge_data.get('relation', '') if edge_data else ''
                if isinstance(rel, list):
                    rel = rel[0] if rel else ''
                
                rel_lower = rel.lower()
                
                if any(term in rel_lower for term in ['brotherof', 'sisterof', 'siblingof']):
                    neighbor_gens.append(generation[neighbor])
                elif any(term in rel_lower for term in ['fatherof', 'motherof', 'parentof']):
                    neighbor_gens.append(generation[neighbor] + 1)
                elif any(term in rel_lower for term in ['sonof', 'daughterof', 'childof']):
                    neighbor_gens.append(generation[neighbor] - 1)
                else:
                    neighbor_gens.append(generation[neighbor])
        
        if neighbor_gens:
            # Use median to be robust to outliers
            generation[node] = int(np.median(neighbor_gens))
    
    return generation


# ============================================================================
# GENDER CLASSIFICATION
# ============================================================================

def infer_gender(G_directed: nx.DiGraph, verbose: bool = True) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Infer gender of nodes based on relationship semantics.
    
    This function uses a rule-based approach to classify gender:
    - If a node is the HEAD (source) of a male-specific relation → Male
    - If a node is the HEAD (source) of a female-specific relation → Female
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed family graph.
    verbose : bool
        Whether to print detailed conflict information.
    
    Returns
    -------
    Tuple[nx.DiGraph, Dict[str, Any]]
        - Graph with 'gender' attribute added to nodes
        - Summary dictionary with counts and conflicts
    
    Gender Classification Rules
    ---------------------------
    Male Relations (HEAD is Male):
        fatherOf, brotherOf, sonOf, uncleOf, husbandOf, grandfatherOf,
        nephewOf, boyCousinOf, grandsonOf, greatUncleOf, etc.
    
    Female Relations (HEAD is Female):
        motherOf, sisterOf, daughterOf, auntOf, wifeOf, grandmotherOf,
        nieceOf, girlCousinOf, granddaughterOf, greatAuntOf, etc.
    
    Conflict Handling
    -----------------
    If a node appears as HEAD of both male and female relations:
    1. Log the conflict with full edge details
    2. Mark the node as 'Unmapped' (data inconsistency)
    """
    
    if verbose:
        print("\n" + "=" * 60)
        print("GENDER INFERENCE")
        print("=" * 60)
    
    # Initialize all nodes as Unknown
    gender_assignments = {node: 'Unknown' for node in G_directed.nodes()}
    
    # Track evidence for each node
    gender_evidence = defaultdict(lambda: {'male': [], 'female': []})
    
    # Scan all edges and collect gender evidence
    for head, tail, data in G_directed.edges(data=True):
        relation = data.get('relation', '')
        
        if isinstance(relation, list):
            relations = relation
        else:
            relations = [relation]
        
        for rel in relations:
            if rel in MALE_RELATIONS:
                gender_evidence[head]['male'].append((rel, tail))
            elif rel in FEMALE_RELATIONS:
                gender_evidence[head]['female'].append((rel, tail))
    
    # Assign genders and detect conflicts
    conflicts = []
    
    for node in G_directed.nodes():
        male_evidence = gender_evidence[node]['male']
        female_evidence = gender_evidence[node]['female']
        
        has_male = len(male_evidence) > 0
        has_female = len(female_evidence) > 0
        
        if has_male and has_female:
            # CONFLICT: Node has both male and female relations as HEAD
            conflicts.append({
                'node': node,
                'male_relations': male_evidence,
                'female_relations': female_evidence
            })
            gender_assignments[node] = 'Unmapped'
        elif has_male:
            gender_assignments[node] = 'Male'
        elif has_female:
            gender_assignments[node] = 'Female'
        # else: remains 'Unknown' (no gendered relations as HEAD)
    
    # Store gender as node attribute
    nx.set_node_attributes(G_directed, gender_assignments, 'gender')
    
    # Create summary
    gender_counts = Counter(gender_assignments.values())
    summary = {
        'male_count': gender_counts.get('Male', 0),
        'female_count': gender_counts.get('Female', 0),
        'unknown_count': gender_counts.get('Unknown', 0),
        'unmapped_count': gender_counts.get('Unmapped', 0),
        'conflicts': conflicts
    }
    
    if verbose:
        print(f"\nGENDER CLASSIFICATION SUMMARY:")
        print(f"   - Male:     {summary['male_count']:,} nodes")
        print(f"   - Female:   {summary['female_count']:,} nodes")
        print(f"   - Unknown:  {summary['unknown_count']:,} nodes")
        print(f"   - Unmapped: {summary['unmapped_count']:,} nodes (conflicts)")
        
        if conflicts:
            print(f"\nGENDER CONFLICTS DETECTED ({len(conflicts)}):")
            for i, conflict in enumerate(conflicts[:10], 1):  # Show first 10
                print(f"\n   Conflict {i}: Node '{conflict['node']}'")
                print(f"      Male evidence ({len(conflict['male_relations'])}):")
                for rel, target in conflict['male_relations'][:3]:
                    print(f"         - {conflict['node']} --[{rel}]--> {target}")
                print(f"      Female evidence ({len(conflict['female_relations'])}):")
                for rel, target in conflict['female_relations'][:3]:
                    print(f"         - {conflict['node']} --[{rel}]--> {target}")
            
            if len(conflicts) > 10:
                print(f"\n      ... and {len(conflicts) - 10} more conflicts")
        
        print("\n" + "=" * 60)
    
    return G_directed, summary


def get_gender_summary(G_directed: nx.DiGraph) -> Dict[str, int]:
    """
    Get a summary of gender distribution in the graph.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        Graph with 'gender' node attributes.
    
    Returns
    -------
    Dict[str, int]
        Gender counts: {'Male': n, 'Female': n, 'Unknown': n, 'Unmapped': n}
    """
    genders = nx.get_node_attributes(G_directed, 'gender')
    return dict(Counter(genders.values()))


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_relationship_distribution(relation_counts: Dict[str, int], 
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Create a bar chart showing the distribution of relationship types.
    
    Parameters
    ----------
    relation_counts : Dict[str, int]
        Dictionary mapping relationship types to counts.
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Sort by count for better visualization
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    relations = [r[0] for r in sorted_relations]
    counts = [r[1] for r in sorted_relations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by gender-indication
    colors = []
    for rel in relations:
        if rel in MALE_RELATIONS:
            colors.append('#3498db')  # Blue for male relations
        elif rel in FEMALE_RELATIONS:
            colors.append('#e74c3c')  # Red for female relations
        else:
            colors.append('#95a5a6')  # Gray for neutral
    
    # Create bar plot
    bars = ax.barh(range(len(relations)), counts, color=colors)
    
    # Customize
    ax.set_yticks(range(len(relations)))
    ax.set_yticklabels(relations)
    ax.invert_yaxis()  # Highest count at top
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Relationship Type', fontsize=12)
    ax.set_title('Distribution of Relationship Types in MetaFam Knowledge Graph', 
                 fontsize=14, fontweight='bold')
    
    # Add count labels
    for i, (count, bar) in enumerate(zip(counts, bars)):
        ax.text(count + max(counts)*0.01, i, f'{count:,}', 
                va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Male Relations'),
        Patch(facecolor='#e74c3c', label='Female Relations'),
        Patch(facecolor='#95a5a6', label='Neutral Relations')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_degree_distribution(G_directed: nx.DiGraph,
                             save_path: Optional[str] = None) -> None:
    """
    Plot in-degree, out-degree, and total degree distributions.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The directed graph.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    in_degrees = [d for _, d in G_directed.in_degree()]
    out_degrees = [d for _, d in G_directed.out_degree()]
    total_degrees = [in_degrees[i] + out_degrees[i] for i in range(len(in_degrees))]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # In-degree distribution
    sns.histplot(in_degrees, ax=axes[0], kde=True, color='#3498db')
    axes[0].set_xlabel('In-Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('In-Degree Distribution\n(High = Many relations point TO this person)')
    
    # Out-degree distribution
    sns.histplot(out_degrees, ax=axes[1], kde=True, color='#e74c3c')
    axes[1].set_xlabel('Out-Degree')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Out-Degree Distribution\n(High = Many relations originate FROM this person)')
    
    # Total degree distribution
    sns.histplot(total_degrees, ax=axes[2], kde=True, color='#2ecc71')
    axes[2].set_xlabel('Total Degree')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Total Degree Distribution\n(Overall connectivity of each person)')
    
    plt.suptitle('Degree Distributions in MetaFam Knowledge Graph', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_generation_distribution(G_directed: nx.DiGraph,
                                  save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of generational depth.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        Graph with 'generation' node attribute.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    generations = nx.get_node_attributes(G_directed, 'generation')
    gen_values = [v for v in generations.values() if v >= 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gen_counts = Counter(gen_values)
    gens = sorted(gen_counts.keys())
    counts = [gen_counts[g] for g in gens]
    
    bars = ax.bar(gens, counts, color='#2ecc71', edgecolor='black')
    
    ax.set_xlabel('Generation (0 = Root Ancestor)', fontsize=12)
    ax.set_ylabel('Number of People', fontsize=12)
    ax.set_title('Generational Depth Distribution in MetaFam', 
                 fontsize=14, fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(gens)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# GEPHI EXPORT
# ============================================================================

def export_to_gephi(G_directed: nx.DiGraph, 
                    filepath: str,
                    verify_attributes: bool = True) -> None:
    """
    Export the graph to GEXF format for Gephi visualization.
    
    The exported file will include all node attributes:
    - gender: Inferred gender (Male/Female/Unknown/Unmapped)
    - generation: Generational depth from root
    - in_degree: Number of incoming edges
    - out_degree: Number of outgoing edges
    - pagerank: PageRank centrality score
    
    Note: Betweenness centrality is computed for analysis but NOT stored
    as a node attribute to reduce export file size.
    
    Parameters
    ----------
    G_directed : nx.DiGraph
        The graph with computed node attributes.
    filepath : str
        Output path for the .gexf file.
    verify_attributes : bool
        Whether to verify all required attributes exist.
    
    Notes
    -----
    GEXF (Graph Exchange XML Format) is the preferred format for Gephi as it:
    - Preserves all node and edge attributes
    - Supports directed graphs
    - Maintains attribute types (string, float, int)
    - Is an open XML standard for graph exchange
    """
    
    print("\n" + "=" * 60)
    print("GEPHI EXPORT")
    print("=" * 60)
    
    # Verify required attributes
    required_attrs = ['gender', 'generation', 'in_degree', 'out_degree', 'pagerank']
    
    if verify_attributes:
        sample_node = list(G_directed.nodes())[0] if G_directed.nodes() else None
        if sample_node:
            node_attrs = G_directed.nodes[sample_node]
            missing = [attr for attr in required_attrs if attr not in node_attrs]
            
            if missing:
                print(f"Warning: Missing attributes: {missing}")
                print("   Run calculate_node_features() and infer_gender() first.")
            else:
                print("All required node attributes present.")
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to GEXF
    nx.write_gexf(G_directed, filepath)
    
    # Print summary
    print(f"\nFile exported: {filepath}")
    print(f"   - Nodes: {G_directed.number_of_nodes():,}")
    print(f"   - Edges: {G_directed.number_of_edges():,}")
    print(f"   - Node attributes: {required_attrs}")
    
    # File size
    file_size = Path(filepath).stat().st_size
    if file_size > 1024 * 1024:
        print(f"   - File size: {file_size / (1024*1024):.2f} MB")
    else:
        print(f"   - File size: {file_size / 1024:.2f} KB")
    
    print("\nGEPHI TIPS:")
    print("   1. Import this file in Gephi (File > Open)")
    print("   2. Use 'gender' attribute for node coloring")
    print("   3. Use 'generation' for vertical layout")
    print("   4. Use 'pagerank' for node sizing (important ancestors)")
    print("   5. Run ForceAtlas2 layout for network visualization")
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Exploration module loaded successfully.")
    print(f"Male relations defined: {len(MALE_RELATIONS)}")
    print(f"Female relations defined: {len(FEMALE_RELATIONS)}")
