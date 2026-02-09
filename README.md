# MetaFam Knowledge Graph Analysis

A comprehensive analysis of the MetaFam synthetic family knowledge graph, implementing graph exploration, community detection, rule mining, and link prediction using both custom implementations and state-of-the-art libraries.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Tasks Overview](#tasks-overview)
  - [Task 1: Dataset Exploration](#task-1-dataset-exploration)
  - [Task 2: Community Detection](#task-2-community-detection)
  - [Task 3: Rule Mining](#task-3-rule-mining)
  - [Task 4: Link Prediction](#task-4-link-prediction)
- [Key Findings](#key-findings)
- [Running the Notebooks](#running-the-notebooks)
- [Results](#results)
- [References](#references)

---

## Overview

This project analyzes the **MetaFam Knowledge Graph**, a synthetic dataset representing family relationships. The analysis spans four interconnected tasks that progressively build understanding from basic exploration to advanced machine learning:

1. **Graph Exploration** - Understand structure, compute metrics, infer node attributes
2. **Community Detection** - Identify family clusters using multiple algorithms  
3. **Rule Mining** - Discover and validate logical horn-clause rules
4. **Link Prediction** - Train KGE and GNN models with leakage-aware evaluation

---

## Dataset

| Metric | Value |
|--------|-------|
| Total Nodes (Entities) | 1,316 |
| Total Edges (Triples) | 13,821 |
| Unique Relation Types | 28 |
| Connected Components | 50 (family clusters) |
| Average Degree | 21.0 |
| Graph Density | 0.008 |
| Generations | 4 |

### Relation Types

The dataset contains 28 family relationship types including:
- **Parent-Child**: `fatherOf`, `motherOf`, `sonOf`, `daughterOf`
- **Sibling**: `brotherOf`, `sisterOf`
- **Grandparent**: `grandfatherOf`, `grandmotherOf`, `grandsonOf`, `granddaughterOf`
- **Extended**: `uncleOf`, `auntOf`, `nephewOf`, `nieceOf`
- **Cousin**: `boyCousinOf`, `girlCousinOf`
- **Great-Relations**: `greatUncleOf`, `greatAuntOf`, etc.

**Notable**: No spouse relations exist, explaining the 50 isolated family components.

---

## Project Structure

```
MetaFam-Project/
├── data/
│   ├── raw/
│   │   ├── train.txt          # Training triples (13,821 edges)
│   │   └── test.txt           # Held-out test set
│   └── processed/             # Cached graph objects
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Graph construction utilities
│   ├── exploration.py         # Task 1: Metrics & attribute inference
│   ├── communities.py         # Task 2: Community detection algorithms
│   ├── proximity.py           # Task 2: Proximity metrics (RWR, Katz)
│   ├── rules.py               # Task 3: Horn-clause rule validation
│   ├── splitting.py           # Task 4: Data splitting strategies
│   ├── kge_models.py          # Task 4: TransE, DistMult, ComplEx, RotatE
│   ├── gnn_models.py          # Task 4: RGCN implementations
│   └── train_eval.py          # Task 4: Training loops & evaluation
├── notebooks/
│   ├── 01_Exploration.ipynb   # Task 1 execution
│   ├── 02_Communities.ipynb   # Task 2 execution
│   ├── 03_Rule_Mining.ipynb   # Task 3 execution
│   └── 04_Link_Pred.ipynb     # Task 4 execution (GPU recommended)
├── outputs/
│   ├── gephi/                 # Graph exports (.gexf)
│   ├── plots/                 # Visualization outputs
│   ├── rules/                 # Rule mining results
│   ├── splits/                # Generated data splits
│   └── results/               # Model evaluation metrics
├── report/
│   └── MetaFam_Complete_Report.tex  # LaTeX report with all findings
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for Task 4)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd Precog_task_graphs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# For Task 4 (Link Prediction), install PyTorch and PyTorch Geometric
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pykeen  # Optional: for library model comparison
```

### Dependencies

**Core:**
- `networkx>=3.0` - Graph operations
- `numpy>=1.24` - Numerical computing
- `pandas>=2.0` - Data manipulation

**Visualization:**
- `matplotlib>=3.7` - Plotting
- `seaborn>=0.12` - Statistical visualization

**Task 2 (Communities):**
- `python-louvain` - Louvain algorithm
- `leidenalg` - Leiden algorithm
- `igraph` - Graph algorithms

**Task 4 (Link Prediction):**
- `torch>=2.0` - Deep learning
- `torch-geometric` - GNN layers
- `pykeen` - KGE library (optional)

---

## Tasks Overview

### Task 1: Dataset Exploration

**Objective:** Understand the graph structure and infer useful node attributes.

**Implementation:** `src/exploration.py`, `notebooks/01_Exploration.ipynb`

**Key Operations:**
- Load triples and construct directed/undirected NetworkX graphs
- Calculate global metrics (density, clustering coefficient, connected components)
- Infer node attributes:
  - **Gender**: From relation semantics (e.g., `fatherOf` → Male)
  - **Generation**: Shortest path from local roots (in-degree=0 nodes)
  - **PageRank**: Node importance score
  - **Degree**: In-degree and out-degree

**Output:** `outputs/gephi/metafam_task1_refined.gexf`

---

### Task 2: Community Detection

**Objective:** Identify family clusters using multiple algorithms and compare results.

**Implementation:** `src/communities.py`, `src/proximity.py`, `notebooks/02_Communities.ipynb`

**Algorithms Implemented:**
1. **Girvan-Newman** - Divisive hierarchical (edge betweenness)
2. **Louvain** - Greedy modularity optimization
3. **Leiden** - Improved Louvain with guaranteed connectivity

**Evaluation Metrics:**
- Modularity (Q ≈ 0.98 for all algorithms)
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)

**Proximity Metrics:**
- Random Walk with Restart (RWR)
- Katz Index (global path-based)

**Output:** `outputs/gephi/metafam_full.gexf` with community labels

---

### Task 3: Rule Mining

**Objective:** Discover and validate logical horn-clause rules in the knowledge graph.

**Implementation:** `src/rules.py`, `notebooks/03_Rule_Mining.ipynb`

**Rules Validated:**

| Rule | Formula | Confidence |
|------|---------|------------|
| Grandmother | Mother(x,y) ∧ Mother(z,x) → Grandmother(z,y) | 100% |
| Sibling | Mother(z,x) ∧ Child(y,z) ∧ (x≠y) → Sibling(x,y) | 100% |
| Aunt | Mother(x,y) ∧ Mother(z,x) ∧ Daughter(w,z) → Aunt(w,y) | 100% |
| Parent/Child | Father(x,y) → Child(y,x) | 83% |
| Sibling Symmetry | Sibling(x,y) → Sibling(y,x) | 100% |
| Gender Inverse | Sister(x,y) ∧ isMale(y) → Brother(y,x) | 100% |
| Cousin Rules | Complex cousin chains | 0% (missing relation types) |

**Noise Analysis:** Adding irrelevant predicates causes 636× support explosion but confidence remains unchanged.

**Output:** `outputs/rules/rule_metrics.csv`, `outputs/rules/rule_report.txt`

---

### Task 4: Link Prediction

**Objective:** Train embedding models and evaluate with leakage-aware data splitting.

**Implementation:** `src/splitting.py`, `src/kge_models.py`, `src/gnn_models.py`, `src/train_eval.py`, `notebooks/04_Link_Pred.ipynb`

#### Data Splitting Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Naive Random** | 80/20 split, vocab from train only | Inductive evaluation |
| **Transductive** | 80/20 split, vocab from train+valid union | Standard KGE setup |
| **Inverse Leakage Removal** | Groups inverse pairs as units | Fair evaluation |
| **Full Train** | 100% train, no validation | Upper-bound baseline |

#### Models Implemented

**Knowledge Graph Embeddings (Custom):**
- **TransE** - Translation-based (h + r ≈ t)
- **DistMult** - Bilinear diagonal (symmetric)
- **ComplEx** - Complex-valued embeddings
- **RotatE** - Rotation in complex space

**Graph Neural Networks:**
- **RGCN + DistMult** - Relational GCN encoder with DistMult decoder
- **RGCN + RotatE** - RGCN encoder with RotatE decoder

**Library Comparison:**
- PyKEEN implementations of all 4 KGE models

#### Evaluation Metrics

- **MRR** (Mean Reciprocal Rank)
- **Hits@1** (% correct in top 1)
- **Hits@10** (% correct in top 10)

**Output:** `outputs/results/link_prediction_results.csv`

---

## Key Findings

### Task 1: Exploration
- Graph is a **forest of 50 isolated family trees** (no inter-family marriages)
- Each family has **26-27 members** across **4 generations**
- **High clustering coefficient (0.73)** indicates strong family cohesion
- **Gender split**: 51% female, 49% male (balanced)

### Task 2: Communities
- All three algorithms detect **~50 communities** matching family structure
- **Modularity ≈ 0.98** across all methods
- Louvain and Leiden produce **identical results** for this sparse graph
- **No bridge individuals** exist due to isolated family structure

### Task 3: Rules
- **5 rules with 100% confidence** (grandmother, sibling, aunt, symmetry, gender inverse)
- Parent/child inverse has **83% confidence** due to incomplete bidirectional edges
- Complex cousin rules have **0% confidence** (relation types missing from data)

### Task 4: Link Prediction
- **Inverse leakage removal** is critical for fair family graph evaluation
- Performance drops significantly when inverse shortcuts are removed
- **RotatE** is theoretically optimal for diverse relation patterns
- **Transductive** split shows best raw metrics but may be inflated

---

## Running the Notebooks

### Local Execution

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_Exploration.ipynb
# 2. notebooks/02_Communities.ipynb
# 3. notebooks/03_Rule_Mining.ipynb
# 4. notebooks/04_Link_Pred.ipynb (GPU recommended)
```

### Google Colab (for Task 4)

1. Upload `Precog_task_graphs` folder to Google Drive
2. Open `04_Link_Pred.ipynb` in Colab
3. Select GPU runtime (Runtime → Change runtime type → GPU)
4. Run Cell 1 to mount Drive and set up paths
5. Execute remaining cells sequentially

---

## Results

### Model Comparison (Test MRR)

Results vary by split type. The inverse-leakage-removal split provides the most reliable estimate of true generalization.

| Model | Naive Random | Transductive | Inverse Removed | Full Train |
|-------|-------------|--------------|-----------------|------------|
| TransE | - | - | - | - |
| DistMult | - | - | - | - |
| ComplEx | - | - | - | - |
| RotatE | - | - | - | - |
| RGCN+DistMult | - | - | - | - |
| RGCN+RotatE | - | - | - | - |

*Run `04_Link_Pred.ipynb` to populate results.*

### Visualizations

Generated plots are saved to `outputs/plots/`:
- `relationship_distribution.png` - Relation type frequencies
- `degree_distribution.png` - Node degree distribution
- `gender_distribution.png` - Inferred gender split
- `generation_distribution.png` - Generational structure
- `community_generation_histograms.png` - Generations within communities

Rule mining visualizations in `outputs/rules/`:
- `rule_confidence_chart.png` - Confidence comparison
- `support_vs_success.png` - Support vs success counts
- `noise_analysis.png` - Effect of irrelevant predicates

---

## References

### Community Detection
- Girvan, M. & Newman, M.E.J. (2002). Community structure in social and biological networks. *PNAS*.
- Blondel, V.D., et al. (2008). Fast unfolding of communities in large networks. *JSTAT*.
- Traag, V.A., et al. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*.

### Knowledge Graph Embeddings
- Bordes, A., et al. (2013). Translating embeddings for modeling multi-relational data. *NeurIPS*.
- Yang, B., et al. (2015). Embedding entities and relations for learning and inference in knowledge bases. *ICLR*.
- Trouillon, T., et al. (2016). Complex embeddings for simple link prediction. *ICML*.
- Sun, Z., et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR*.

### Graph Neural Networks
- Kipf, T.N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
- Schlichtkrull, M., et al. (2018). Modeling relational data with graph convolutional networks. *ESWC*.

### Rule Mining
- Galárraga, L., et al. (2013). AMIE: Association rule mining under incomplete evidence. *WWW*.

---

## License

This project was developed for the Precog Research Lab assignment.

---

## Acknowledgments

- Precog Research Lab for the assignment specification
- PyKEEN team for the KGE library
- NetworkX, PyTorch Geometric communities
