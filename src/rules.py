"""
Rule Mining Module
==================

This module implements logic-based rule mining and validation for the MetaFam
knowledge graph. It validates horn-clause rules against the dataset and
calculates metrics like support, confidence, and exceptions.

Theory:
-------
Horn-clause rules in knowledge graphs take the form:
    Premise₁ ∧ Premise₂ ∧ ... ∧ Premiseₙ → Conclusion

For genealogical knowledge graphs, these rules capture:
1. **Transitivity**: Chains of relations (e.g., mother's mother = grandmother)
2. **Composition**: Combining relations (e.g., mother's sister = aunt)
3. **Symmetry**: Bidirectional relations (e.g., sibling is symmetric)
4. **Inverse**: Complementary relations (e.g., parent ↔ child)

Metrics:
--------
- **Support**: Count of instances where the premise is satisfied
- **Success**: Count where premise AND conclusion are both satisfied
- **Confidence**: Success / Support (how often the rule holds)
- **Exceptions**: Cases where premise holds but conclusion doesn't

Edge Semantics:
---------------
In this graph, edge (h, t, relation) means: h IS [relation] OF t
- "olivia0 motherOf lisa5" → olivia0 is mother of lisa5 (parent → child)
- "nico4 sonOf olivia0" → nico4 is son of olivia0 (child → parent)

Author: MetaFam Analysis Team
"""

import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass, field
import time


@dataclass
class RuleResult:
    """Container for rule validation results."""
    rule_id: int
    rule_name: str
    rule_definition: str
    support: int
    success: int
    confidence: float
    examples: List[Tuple]  # Successful matches
    exceptions: List[Tuple]  # Failed matches (premise true, conclusion false)
    execution_time: float = 0.0
    
    def __str__(self) -> str:
        return (f"Rule {self.rule_id}: {self.rule_name}\n"
                f"  Definition: {self.rule_definition}\n"
                f"  Support: {self.support}, Success: {self.success}, "
                f"Confidence: {self.confidence:.4f}")


class RuleValidator:
    """
    Validates horn-clause rules against a knowledge graph.
    
    This class provides methods to evaluate logical patterns in genealogical
    knowledge graphs, calculating metrics like support, confidence, and
    identifying exceptions to rules.
    
    Parameters
    ----------
    G : nx.DiGraph
        The directed knowledge graph with 'relation' edge attributes.
        If edges have 'relations' attribute (list), all are considered.
        
    Attributes
    ----------
    G : nx.DiGraph
        The input graph.
    edge_index : Dict[str, Set[Tuple[str, str]]]
        Index mapping relation types to (head, tail) edge pairs.
    reverse_edge_index : Dict[str, Set[Tuple[str, str]]]
        Index mapping relation types to (tail, head) for reverse lookups.
    node_genders : Dict[str, str]
        Gender attribute for each node (if available).
        
    Relation Mapping
    ----------------
    The following relation pairs are semantically inverse:
    - motherOf ↔ sonOf, daughterOf
    - fatherOf ↔ sonOf, daughterOf  
    - grandmotherOf ↔ grandsonOf, granddaughterOf
    - grandfatherOf ↔ grandsonOf, granddaughterOf
    - uncleOf ↔ nephewOf, nieceOf
    - auntOf ↔ nephewOf, nieceOf
    - sisterOf ↔ brotherOf, sisterOf (both valid inverses)
    - brotherOf ↔ brotherOf, sisterOf (both valid inverses)
    """
    
    # Define inverse relation mappings
    INVERSE_RELATIONS = {
        'motherOf': ['sonOf', 'daughterOf'],
        'fatherOf': ['sonOf', 'daughterOf'],
        'sonOf': ['motherOf', 'fatherOf'],
        'daughterOf': ['motherOf', 'fatherOf'],
        'grandmotherOf': ['grandsonOf', 'granddaughterOf'],
        'grandfatherOf': ['grandsonOf', 'granddaughterOf'],
        'grandsonOf': ['grandmotherOf', 'grandfatherOf'],
        'granddaughterOf': ['grandmotherOf', 'grandfatherOf'],
        'greatGrandmotherOf': ['greatGrandsonOf', 'greatGranddaughterOf'],
        'greatGrandfatherOf': ['greatGrandsonOf', 'greatGranddaughterOf'],
        'greatGrandsonOf': ['greatGrandmotherOf', 'greatGrandfatherOf'],
        'greatGranddaughterOf': ['greatGrandmotherOf', 'greatGrandfatherOf'],
        'uncleOf': ['nephewOf', 'nieceOf'],
        'auntOf': ['nephewOf', 'nieceOf'],
        'nephewOf': ['uncleOf', 'auntOf'],
        'nieceOf': ['uncleOf', 'auntOf'],
        'sisterOf': ['brotherOf', 'sisterOf'],
        'brotherOf': ['brotherOf', 'sisterOf'],
    }
    
    # Relations grouped by type
    PARENT_RELATIONS = ['motherOf', 'fatherOf']
    CHILD_RELATIONS = ['sonOf', 'daughterOf']
    SIBLING_RELATIONS = ['sisterOf', 'brotherOf']
    GRANDPARENT_RELATIONS = ['grandmotherOf', 'grandfatherOf']
    GRANDCHILD_RELATIONS = ['grandsonOf', 'granddaughterOf']
    COUSIN_RELATIONS = ['boyCousinOf', 'girlCousinOf']
    FIRST_COUSIN_ONCE_REMOVED = ['boyFirstCousinOnceRemovedOf', 'girlFirstCousinOnceRemovedOf']
    
    def __init__(self, G: nx.DiGraph):
        """
        Initialize the RuleValidator with a knowledge graph.
        
        Parameters
        ----------
        G : nx.DiGraph
            Directed graph where edges have 'relation' or 'relations' attribute.
        """
        self.G = G
        self.edge_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.reverse_edge_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.node_genders: Dict[str, str] = {}
        
        self._build_indices()
        
    def _build_indices(self) -> None:
        """Build edge indices for efficient lookups."""
        print("Building edge indices for rule validation...")
        
        for u, v, data in self.G.edges(data=True):
            # Handle single relation or list of relations
            relations = data.get('relations', [data.get('relation', None)])
            if isinstance(relations, str):
                relations = [relations]
            
            for rel in relations:
                if rel:
                    self.edge_index[rel].add((u, v))
                    self.reverse_edge_index[rel].add((v, u))
        
        # Extract gender attributes
        for node in self.G.nodes():
            gender = self.G.nodes[node].get('gender', 'Unknown')
            self.node_genders[node] = gender
            
        print(f"  Indexed {len(self.edge_index)} relation types")
        print(f"  Total edges indexed: {sum(len(v) for v in self.edge_index.values())}")
        
    def has_relation(self, head: str, tail: str, relation: str) -> bool:
        """
        Check if a specific relation exists between two nodes.
        
        Parameters
        ----------
        head : str
            Source node.
        tail : str
            Target node.
        relation : str
            Relation type to check.
            
        Returns
        -------
        bool
            True if the relation exists.
        """
        return (head, tail) in self.edge_index.get(relation, set())
    
    def has_any_relation(self, head: str, tail: str, relations: List[str]) -> bool:
        """
        Check if any of the specified relations exists between two nodes.
        
        Parameters
        ----------
        head : str
            Source node.
        tail : str
            Target node.
        relations : List[str]
            List of relation types to check.
            
        Returns
        -------
        bool
            True if any relation exists.
        """
        for rel in relations:
            if (head, tail) in self.edge_index.get(rel, set()):
                return True
        return False
    
    def get_related_nodes(self, node: str, relation: str, 
                          direction: str = 'outgoing') -> Set[str]:
        """
        Get all nodes related to the given node by a specific relation.
        
        Parameters
        ----------
        node : str
            The source node.
        relation : str
            The relation type.
        direction : str
            'outgoing' for node -> x, 'incoming' for x -> node.
            
        Returns
        -------
        Set[str]
            Set of related nodes.
        """
        if direction == 'outgoing':
            return {t for (h, t) in self.edge_index.get(relation, set()) if h == node}
        else:  # incoming
            return {h for (h, t) in self.edge_index.get(relation, set()) if t == node}
    
    def get_edges_by_relation(self, relation: str) -> Set[Tuple[str, str]]:
        """Get all edges with a specific relation."""
        return self.edge_index.get(relation, set())
    
    # =========================================================================
    # GROUP A: TRANSITIVE & COMPOSITIONAL RULES
    # =========================================================================
    
    def validate_grandmother_rule(self) -> RuleResult:
        """
        Rule 1: Grandmother Logic
        
        Definition: Mother(x, y) ∧ Mother(z, x) → Grandmother(z, y)
        
        Logic: If z is mother of x, and x is mother of y, then z should be
        grandmother of y (grandmother = mother's mother).
        
        Theory:
        -------
        This is a transitivity rule capturing the composition of maternal
        relations across two generations. In a complete knowledge graph,
        this rule should have 100% confidence. Lower confidence indicates
        missing grandmother edges or data inconsistencies.
        
        Returns
        -------
        RuleResult
            Validation results with support, confidence, examples, exceptions.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get all motherOf edges: (mother, child)
        mother_edges = self.get_edges_by_relation('motherOf')
        
        # For each x such that Mother(x, y) holds
        for (x, y) in mother_edges:
            # Find all z where Mother(z, x) holds (z is mother of x)
            mothers_of_x = self.get_related_nodes(x, 'motherOf', 'incoming')
            
            for z in mothers_of_x:
                support += 1
                
                # Check if Grandmother(z, y) holds
                if self.has_relation(z, y, 'grandmotherOf'):
                    success += 1
                    if len(examples) < 100:  # Store up to 100 examples
                        examples.append((z, x, y))
                else:
                    if len(exceptions) < 100:
                        exceptions.append((z, x, y))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=1,
            rule_name="Grandmother Logic",
            rule_definition="Mother(x, y) ∧ Mother(z, x) → Grandmother(z, y)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    def validate_sibling_rule(self) -> RuleResult:
        """
        Rule 2: Sibling Logic
        
        Definition: Mother(z, x) ∧ Child(y, z) ∧ (x ≠ y) → Sibling(x, y)
        
        Logic: If z is mother of x, and y is a child of z (same mother),
        and x ≠ y, then x and y should be siblings.
        
        Theory:
        -------
        In this graph, Child(y, z) means y --daughterOf/sonOf--> z.
        We check both directions to find all children of z.
        Two different children of the same mother should be siblings.
        
        Note: This only checks maternal half-siblings. Full sibling check
        would require matching both parents.
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get all motherOf edges: (mother, child)
        mother_edges = self.get_edges_by_relation('motherOf')
        
        # Group children by mother
        mother_to_children: Dict[str, Set[str]] = defaultdict(set)
        for (mother, child) in mother_edges:
            mother_to_children[mother].add(child)
        
        # Also add children via child -> parent relations
        for rel in self.CHILD_RELATIONS:
            for (child, parent) in self.get_edges_by_relation(rel):
                # Check if parent is female (mother)
                if self.node_genders.get(parent) == 'Female':
                    mother_to_children[parent].add(child)
        
        # For each pair of children with same mother
        for mother, children in mother_to_children.items():
            for x in children:
                for y in children:
                    if x != y:
                        support += 1
                        
                        # Check if Sibling(x, y) holds
                        if self.has_any_relation(x, y, self.SIBLING_RELATIONS):
                            success += 1
                            if len(examples) < 100:
                                examples.append((mother, x, y))
                        else:
                            if len(exceptions) < 100:
                                exceptions.append((mother, x, y))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=2,
            rule_name="Sibling Logic",
            rule_definition="Mother(z, x) ∧ Child(y, z) ∧ (x ≠ y) → Sibling(x, y)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    def validate_aunt_rule(self) -> RuleResult:
        """
        Rule 3: Aunt Logic
        
        Definition: Mother(x, y) ∧ Mother(z, x) ∧ Daughter(w, z) → Aunt(w, y)
        
        Logic: 
        - x --motherOf--> y (x is mother of y)
        - z --motherOf--> x (z is grandmother of y, mother of x)
        - w --daughterOf--> z (w is daughter of z, sister of x)
        - Therefore: w is maternal aunt of y
        
        Theory:
        -------
        This rule captures the aunt relationship through the maternal line:
        If w is a sister of x (both daughters of z), and x has a child y,
        then w is an aunt of y.
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get motherOf edges
        mother_edges = self.get_edges_by_relation('motherOf')
        
        for (x, y) in mother_edges:  # x is mother of y
            # Find z where z is mother of x
            mothers_of_x = self.get_related_nodes(x, 'motherOf', 'incoming')
            
            for z in mothers_of_x:  # z is grandmother of y
                # Find w where w is daughter of z (but w != x)
                daughters_of_z = self.get_related_nodes(z, 'daughterOf', 'incoming')
                
                # Also check motherOf for other children
                other_children_of_z = self.get_related_nodes(z, 'motherOf', 'outgoing')
                
                # w must be female (daughter) and not x
                potential_aunts = daughters_of_z | {c for c in other_children_of_z 
                                                     if self.node_genders.get(c) == 'Female'}
                
                for w in potential_aunts:
                    if w != x:  # w is sister of x
                        support += 1
                        
                        # Check if Aunt(w, y) holds
                        if self.has_relation(w, y, 'auntOf'):
                            success += 1
                            if len(examples) < 100:
                                examples.append((x, y, z, w))
                        else:
                            if len(exceptions) < 100:
                                exceptions.append((x, y, z, w))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=3,
            rule_name="Aunt Logic",
            rule_definition="Mother(x, y) ∧ Mother(z, x) ∧ Daughter(w, z) → Aunt(w, y)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    # =========================================================================
    # GROUP B: INVERSE RULES
    # =========================================================================
    
    def validate_parent_child_inverse(self) -> RuleResult:
        """
        Rule 4: Parent/Child Inverse
        
        Definition: Father(x, y) → Child(y, x)
        
        Logic: If x is father of y (x --fatherOf--> y), then y should have
        a child relation to x (y --sonOf/daughterOf--> x).
        
        Theory:
        -------
        This tests inverse relation consistency. In a complete KG, every
        parent edge should have a corresponding child edge in the reverse
        direction. The specific child relation depends on the gender of y.
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get all fatherOf edges
        father_edges = self.get_edges_by_relation('fatherOf')
        
        for (x, y) in father_edges:  # x is father of y
            support += 1
            
            # Check if Child(y, x) holds (y --sonOf/daughterOf--> x)
            if self.has_any_relation(y, x, self.CHILD_RELATIONS):
                success += 1
                if len(examples) < 100:
                    examples.append((x, y))
            else:
                if len(exceptions) < 100:
                    exceptions.append((x, y))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=4,
            rule_name="Parent/Child Inverse",
            rule_definition="Father(x, y) → Child(y, x)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    def validate_sibling_symmetry(self) -> RuleResult:
        """
        Rule 5: Sibling Symmetry
        
        Definition: Sibling(x, y) → Sibling(y, x)
        
        Logic: If x is sibling of y, then y should be sibling of x
        (siblings is a symmetric relation).
        
        Theory:
        -------
        Sibling relations should be symmetric. If x --sisterOf/brotherOf--> y,
        then y should have some sibling relation to x. The specific relation
        depends on gender:
        - If y is male: y --brotherOf--> x
        - If y is female: y --sisterOf--> x
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get all sibling edges
        for rel in self.SIBLING_RELATIONS:
            for (x, y) in self.get_edges_by_relation(rel):
                support += 1
                
                # Check if Sibling(y, x) holds
                if self.has_any_relation(y, x, self.SIBLING_RELATIONS):
                    success += 1
                    if len(examples) < 100:
                        examples.append((x, y, rel))
                else:
                    if len(exceptions) < 100:
                        exceptions.append((x, y, rel))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=5,
            rule_name="Sibling Symmetry",
            rule_definition="Sibling(x, y) → Sibling(y, x)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    def validate_gender_inverse(self) -> RuleResult:
        """
        Rule 6: Gender Inverse
        
        Definition: SisterOf(x, y) → BrotherOf(y, x) [if y is Male]
        
        Logic: If x is sister of y, and y is male, then y should be
        brother of x.
        
        Theory:
        -------
        This rule tests gender-specific inverse relations. If x is female
        (implied by sisterOf) and y is male, then y should be brotherOf x.
        If y is female, y should be sisterOf x (but that's a different rule).
        
        This rule ONLY applies when y is Male, making it a conditional rule.
        Violations may indicate gender inference errors or data inconsistency.
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get all sisterOf edges
        sister_edges = self.get_edges_by_relation('sisterOf')
        
        for (x, y) in sister_edges:  # x is sister of y
            # Only check when y is Male
            if self.node_genders.get(y) == 'Male':
                support += 1
                
                # Check if BrotherOf(y, x) holds
                if self.has_relation(y, x, 'brotherOf'):
                    success += 1
                    if len(examples) < 100:
                        examples.append((x, y))
                else:
                    if len(exceptions) < 100:
                        exceptions.append((x, y))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=6,
            rule_name="Gender Inverse",
            rule_definition="SisterOf(x, y) → BrotherOf(y, x) [if y is Male]",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    # =========================================================================
    # GROUP C: COMPLEX/EXTENDED RULES
    # =========================================================================
    
    def validate_first_cousin_once_removed_type_a(self) -> RuleResult:
        """
        Rule 7: First Cousin Once Removed (Type A)
        
        Definition: Mother(a, b) ∧ Grandmother(c, a) ∧ Daughter(d, c) → FirstCousinOnceRemoved(d, b)
        
        Logic:
        - a --motherOf--> b (a is mother of b)
        - c --grandmotherOf--> a (c is grandmother of a)
        - d --daughterOf--> c (d is daughter of c)
        - d is c's daughter and c is a's grandmother
        - so d is great-aunt of b (sibling of a's grandparent)
        
        Critical Analysis:
        ------------------
        Wait - this logic actually describes a GREAT-AUNT relationship, not
        first cousin once removed. Let me verify:
        
        c (grandmother of a) → c is parent of a's parent
        d (daughter of c) → d is sibling of a's parent = AUNT of a, GREAT-AUNT of b
        
        However, we'll validate as specified and report findings.
        The "FirstCousinOnceRemoved" relations in the dataset may have
        different semantics.
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get motherOf edges
        mother_edges = self.get_edges_by_relation('motherOf')
        
        for (a, b) in mother_edges:  # a is mother of b
            # Find c where c is grandmother of a
            grandmothers_of_a = self.get_related_nodes(a, 'grandmotherOf', 'incoming')
            
            for c in grandmothers_of_a:
                # Find d where d is daughter of c (d != a's parent)
                # d --daughterOf--> c means d is daughter of c
                daughters_of_c = self.get_related_nodes(c, 'daughterOf', 'incoming')
                
                # Also get children via motherOf relation from c
                children_of_c = self.get_related_nodes(c, 'motherOf', 'outgoing')
                female_children = {ch for ch in children_of_c 
                                   if self.node_genders.get(ch) == 'Female'}
                
                all_daughters = daughters_of_c | female_children
                
                for d in all_daughters:
                    support += 1
                    
                    # Check if FirstCousinOnceRemoved(d, b) holds
                    if self.has_any_relation(d, b, self.FIRST_COUSIN_ONCE_REMOVED):
                        success += 1
                        if len(examples) < 100:
                            examples.append((a, b, c, d))
                    else:
                        if len(exceptions) < 100:
                            exceptions.append((a, b, c, d))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=7,
            rule_name="First Cousin Once Removed (Type A)",
            rule_definition="Mother(a, b) ∧ Grandmother(c, a) ∧ Daughter(d, c) → FirstCousinOnceRemoved(d, b)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    def validate_first_cousin_once_removed_type_b(self) -> RuleResult:
        """
        Rule 8: First Cousin Once Removed (Type B)
        
        Definition: Father(a, b) ∧ FirstCousin(c, a) ∧ Grandchild(d, c) → FirstCousinOnceRemoved(d, b)
        
        Logic:
        - a --fatherOf--> b (a is father of b)
        - c --*CousinOf--> a (c is first cousin of a)
        - d --grandsonOf/granddaughterOf--> c (d is grandchild of c)
        - d is two generations below c, who is same generation as a
        - So d and b are "first cousins twice removed"? Let's validate.
        
        Theory:
        -------
        First cousin once removed: Either a first cousin's child, or a 
        parent's first cousin. This rule seems to describe a first cousin's
        grandchild, which would be "first cousin twice removed".
        
        We validate as specified and report findings.
        
        Returns
        -------
        RuleResult
            Validation results.
        """
        start_time = time.time()
        
        support = 0
        success = 0
        examples = []
        exceptions = []
        
        # Get fatherOf edges
        father_edges = self.get_edges_by_relation('fatherOf')
        
        for (a, b) in father_edges:  # a is father of b
            # Find c where c is first cousin of a
            cousins_of_a = set()
            for rel in self.COUSIN_RELATIONS:
                cousins_of_a.update(self.get_related_nodes(a, rel, 'incoming'))
                cousins_of_a.update(self.get_related_nodes(a, rel, 'outgoing'))
            
            for c in cousins_of_a:
                # Find d where d is grandchild of c
                # d --grandsonOf/granddaughterOf--> c
                grandchildren_of_c = set()
                for rel in self.GRANDCHILD_RELATIONS:
                    grandchildren_of_c.update(
                        self.get_related_nodes(c, rel, 'incoming')
                    )
                
                for d in grandchildren_of_c:
                    support += 1
                    
                    # Check if FirstCousinOnceRemoved(d, b) holds
                    if self.has_any_relation(d, b, self.FIRST_COUSIN_ONCE_REMOVED):
                        success += 1
                        if len(examples) < 100:
                            examples.append((a, b, c, d))
                    else:
                        if len(exceptions) < 100:
                            exceptions.append((a, b, c, d))
        
        confidence = success / support if support > 0 else 0.0
        execution_time = time.time() - start_time
        
        return RuleResult(
            rule_id=8,
            rule_name="First Cousin Once Removed (Type B)",
            rule_definition="Father(a, b) ∧ FirstCousin(c, a) ∧ Grandchild(d, c) → FirstCousinOnceRemoved(d, b)",
            support=support,
            success=success,
            confidence=confidence,
            examples=examples,
            exceptions=exceptions,
            execution_time=execution_time
        )
    
    # =========================================================================
    # GROUP D: NOISE ANALYSIS
    # =========================================================================
    
    def test_random_insertion(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Rule 9: Random Variable Injection (Noise Analysis)
        
        Definition: Mother(x, y) ∧ Mother(z, x) ∧ Sister(a, b) → Grandmother(z, y)
        
        Experiment:
        -----------
        This rule introduces Sister(a, b) which is completely disconnected
        from the variables x, y, z in the grandmother rule.
        
        Hypothesis:
        -----------
        - Confidence should remain the same as Rule #1 (grandmother rule)
        - BUT the support space explodes combinatorially because a, b can
          be ANY pair of sisters, not just those connected to x, y, z
        
        Theory:
        -------
        Adding irrelevant predicates to a rule does NOT change its semantic
        validity (the conclusion follows from the relevant premises alone).
        However, it massively increases the search space:
        
        - Rule 1 support ≈ O(motherOf edges²)
        - Rule 9 support ≈ O(motherOf² × sisterOf edges)
        
        This demonstrates why rule mining algorithms must detect and prune
        irrelevant predicates - they are computationally expensive without
        adding information.
        
        Parameters
        ----------
        sample_size : int
            Maximum number of combinations to test (for tractability).
            
        Returns
        -------
        Dict[str, Any]
            Comparison results between Rule 1 and Rule 9.
        """
        print("\nNOISE ANALYSIS: Random Variable Injection")
        print("=" * 60)
        
        start_time = time.time()
        
        # First, get Rule 1 results for comparison
        rule1 = self.validate_grandmother_rule()
        print(f"\nBaseline (Rule 1): Grandmother Logic")
        print(f"  Support: {rule1.support}")
        print(f"  Confidence: {rule1.confidence:.4f}")
        
        # Now test with random injection
        mother_edges = list(self.get_edges_by_relation('motherOf'))
        sister_edges = list(self.get_edges_by_relation('sisterOf'))
        
        # Theoretical support space
        theoretical_support = 0
        for (x, y) in mother_edges:
            mothers_of_x = self.get_related_nodes(x, 'motherOf', 'incoming')
            theoretical_support += len(mothers_of_x) * len(sister_edges)
        
        print(f"\nWith Random Injection (Rule 9):")
        print(f"  Rule: Mother(x,y) ∧ Mother(z,x) ∧ Sister(a,b) → Grandmother(z,y)")
        print(f"  Theoretical support space: {theoretical_support:,}")
        print(f"  (= Rule1 support × |Sister edges| = {rule1.support} × {len(sister_edges)})")
        
        # Sample-based validation
        import random
        random.seed(42)
        
        sampled_support = 0
        sampled_success = 0
        
        # For each (x, y, z) from Rule 1, pair with random (a, b) sisters
        samples_per_triple = max(1, sample_size // max(rule1.support, 1))
        
        for (x, y) in mother_edges[:100]:  # Limit for tractability
            mothers_of_x = self.get_related_nodes(x, 'motherOf', 'incoming')
            
            for z in mothers_of_x:
                # Sample sister pairs
                sampled_sisters = random.sample(
                    sister_edges, 
                    min(samples_per_triple, len(sister_edges))
                )
                
                for (a, b) in sampled_sisters:
                    sampled_support += 1
                    
                    # Check conclusion (independent of a, b)
                    if self.has_relation(z, y, 'grandmotherOf'):
                        sampled_success += 1
        
        sampled_confidence = sampled_success / sampled_support if sampled_support > 0 else 0.0
        
        print(f"\n  Sampled validation (n={sampled_support}):")
        print(f"    Confidence: {sampled_confidence:.4f}")
        print(f"    Expected (same as Rule 1): {rule1.confidence:.4f}")
        
        execution_time = time.time() - start_time
        
        # Analysis
        print(f"\nANALYSIS:")
        print(f"  Confidence change: {abs(sampled_confidence - rule1.confidence):.6f}")
        print(f"  Support explosion factor: {len(sister_edges)}x")
        print(f"  Computational overhead: {theoretical_support / max(rule1.support, 1):.0f}x more checks")
        print(f"\n  Conclusion: Adding irrelevant predicates is:")
        print(f"    - Logically NEUTRAL (confidence unchanged)")
        print(f"    - Computationally EXPENSIVE ({len(sister_edges)}x more search space)")
        print(f"  This is why rule mining requires predicate pruning!")
        
        return {
            'rule1_support': rule1.support,
            'rule1_confidence': rule1.confidence,
            'rule9_theoretical_support': theoretical_support,
            'rule9_sampled_support': sampled_support,
            'rule9_sampled_confidence': sampled_confidence,
            'support_explosion_factor': len(sister_edges),
            'confidence_difference': abs(sampled_confidence - rule1.confidence),
            'execution_time': execution_time
        }
    
    # =========================================================================
    # MAIN VALIDATION INTERFACE
    # =========================================================================
    
    def validate_all_rules(self) -> List[RuleResult]:
        """
        Validate all defined rules and return results.
        
        Returns
        -------
        List[RuleResult]
            Results for all 8 rules (noise analysis is separate).
        """
        print("\n" + "=" * 70)
        print("RULE VALIDATION: Validating 8 Horn-Clause Rules")
        print("=" * 70)
        
        results = []
        
        # Group A: Transitive & Compositional
        print("\n--- GROUP A: Transitive & Compositional Rules ---")
        results.append(self.validate_grandmother_rule())
        print(f"  Rule 1: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        results.append(self.validate_sibling_rule())
        print(f"  Rule 2: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        results.append(self.validate_aunt_rule())
        print(f"  Rule 3: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        # Group B: Inverse Rules
        print("\n--- GROUP B: Inverse Rules ---")
        results.append(self.validate_parent_child_inverse())
        print(f"  Rule 4: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        results.append(self.validate_sibling_symmetry())
        print(f"  Rule 5: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        results.append(self.validate_gender_inverse())
        print(f"  Rule 6: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        # Group C: Complex/Extended
        print("\n--- GROUP C: Complex/Extended Rules ---")
        results.append(self.validate_first_cousin_once_removed_type_a())
        print(f"  Rule 7: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        results.append(self.validate_first_cousin_once_removed_type_b())
        print(f"  Rule 8: {results[-1].rule_name} - Confidence: {results[-1].confidence:.4f}")
        
        print("\n" + "=" * 70)
        print("Rule validation complete.")
        
        return results
    
    def results_to_dataframe(self, results: List[RuleResult]) -> pd.DataFrame:
        """
        Convert rule results to a pandas DataFrame.
        
        Parameters
        ----------
        results : List[RuleResult]
            Results from validate_all_rules().
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: rule_id, rule_name, definition, support,
            success, confidence, num_exceptions, execution_time.
        """
        data = []
        for r in results:
            data.append({
                'rule_id': r.rule_id,
                'rule_name': r.rule_name,
                'definition': r.rule_definition,
                'support': r.support,
                'success': r.success,
                'confidence': r.confidence,
                'num_exceptions': len(r.exceptions),
                'execution_time_sec': r.execution_time
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, results: List[RuleResult], 
                     output_path: str,
                     noise_analysis: Optional[Dict] = None) -> None:
        """
        Save rule validation results to CSV and TXT files.
        
        Parameters
        ----------
        results : List[RuleResult]
            Results from validate_all_rules().
        output_path : str
            Path to output directory.
        noise_analysis : Optional[Dict]
            Results from test_random_insertion().
        """
        from pathlib import Path
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics CSV
        df = self.results_to_dataframe(results)
        csv_path = output_dir / 'rule_metrics.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nSaved metrics to: {csv_path}")
        
        # Save detailed TXT report
        txt_path = output_dir / 'rule_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("METAFAM RULE MINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for r in results:
                f.write(f"\n{'='*60}\n")
                f.write(f"RULE {r.rule_id}: {r.rule_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Definition: {r.rule_definition}\n\n")
                f.write(f"METRICS:\n")
                f.write(f"  Support:    {r.support:,}\n")
                f.write(f"  Success:    {r.success:,}\n")
                f.write(f"  Confidence: {r.confidence:.6f}\n")
                f.write(f"  Exceptions: {len(r.exceptions):,}\n")
                f.write(f"  Time:       {r.execution_time:.4f}s\n\n")
                
                f.write("EXAMPLE MATCHES (up to 3):\n")
                for ex in r.examples[:3]:
                    f.write(f"  {ex}\n")
                
                f.write("\nEXAMPLE EXCEPTIONS (up to 3):\n")
                for ex in r.exceptions[:3]:
                    f.write(f"  {ex}\n")
            
            if noise_analysis:
                f.write(f"\n\n{'='*60}\n")
                f.write("NOISE ANALYSIS: Random Variable Injection\n")
                f.write(f"{'='*60}\n")
                f.write(f"Rule 1 Support: {noise_analysis['rule1_support']:,}\n")
                f.write(f"Rule 1 Confidence: {noise_analysis['rule1_confidence']:.6f}\n")
                f.write(f"Rule 9 Theoretical Support: {noise_analysis['rule9_theoretical_support']:,}\n")
                f.write(f"Rule 9 Sampled Confidence: {noise_analysis['rule9_sampled_confidence']:.6f}\n")
                f.write(f"Support Explosion Factor: {noise_analysis['support_explosion_factor']}x\n")
                f.write(f"Confidence Difference: {noise_analysis['confidence_difference']:.6f}\n")
        
        print(f"Saved detailed report to: {txt_path}")


def print_rule_details(result: RuleResult, num_examples: int = 3) -> None:
    """
    Print detailed information about a rule validation result.
    
    Parameters
    ----------
    result : RuleResult
        The rule result to print.
    num_examples : int
        Number of examples and exceptions to show.
    """
    print(f"\n{'='*70}")
    print(f"RULE {result.rule_id}: {result.rule_name}")
    print(f"{'='*70}")
    print(f"\nDefinition: {result.rule_definition}")
    print(f"\nMetrics:")
    print(f"  Support:    {result.support:,}")
    print(f"  Success:    {result.success:,}")
    print(f"  Confidence: {result.confidence:.6f} ({result.confidence*100:.2f}%)")
    print(f"  Exceptions: {len(result.exceptions):,}")
    print(f"  Time:       {result.execution_time:.4f}s")
    
    print(f"\nExample Matches ({min(num_examples, len(result.examples))} of {len(result.examples)}):")
    for ex in result.examples[:num_examples]:
        print(f"  {ex}")
    
    print(f"\nExample Exceptions ({min(num_examples, len(result.exceptions))} of {len(result.exceptions)}):")
    for ex in result.exceptions[:num_examples]:
        print(f"  {ex}")
