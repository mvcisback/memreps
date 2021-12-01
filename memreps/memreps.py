from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import (Any, Callable, Counter, Generator,
                    Iterable, Literal, Protocol, Union)
import attr
import exp4
import funcy as fn
import numpy as np
import networkx as nx
from exp4.algo import softmax
from dfa.utils import find_subset_counterexample

Atom = Any


class Concept(Protocol):
    def __contains__(self, atom: Atom) -> bool:
        ...

    def __xor__(self, other: Concept) -> Concept:
        ...

    def __invert__(self) -> Concept:
        ...

    def __iter__(self) -> Iterable[Atom]:
        ...

    def subset_of(self, other: Concept) -> Atom:
        ...

# ================== Learning API ==========================

# Types of learning queries.
MemQuery = tuple[Literal['∈'], Atom]
CmpQuery = tuple[Literal['≺'], tuple[Atom, Atom]]
EqQuery = tuple[Literal['≡'], Concept]
Query = Union[MemQuery, CmpQuery, EqQuery]

# Types of responses from teacher.
MemResponse = Literal['∈', '∉']
CmpResponse = Literal['≺', '≻', '=', '||']
EqResponse = Union[
    tuple[MemQuery, MemResponse],
    tuple[CmpQuery, CmpResponse],
]
Response = Union[MemResponse, CmpResponse, EqResponse]

LearningAPI = Generator[Query, Response, None]

# ================== Concept Class =========================

Assumption = EqResponse
Assumptions = list[Assumption]
# Note: Generated concepts should all be unique!
ConceptClass = Callable[[Assumptions], Iterable[Concept]]

# ====================== Learner ===========================


def find_distiguishing(concept1, concept2, atoms):
    if len(atoms) <= 1:
        return None

    x, *atoms = atoms
    polarity = x in concept1
    assert (x in concept2) != polarity
    
    for y in atoms:
        if (y in concept1) != polarity:
            assert (y in concept2) == polarity
            return x, y

def find_maximally_distinguishing(concept_gen, max_concepts=15, max_atoms=20):
    #find min cut of the hasse diagram from the concept generator
    # construct the Hasse diagram from the concepts
    concept_num = 0
    hasse = nx.DiGraph()
    for concept in concept_gen:
        if concept_num > max_concepts:
            break
        concept_num += 1
        hasse.add_node(concept_num, concept=concept)
        for candidate in hasse.nodes:
            if candidate == concept_num:
                continue # don't want to add self-edges
            if concept.subset_of(hasse.nodes[candidate]["concept"]) is None:
                hasse.add_edge(candidate, concept_num, capacity=1)
            elif hasse.nodes[candidate]["concept"].subset_of(concept) is None:
                hasse.add_edge(concept_num, candidate, capacity=1)
    hasse.add_node("T", concept=None)  # add source node
    hasse.add_node("⊥", concept=None)  # add sink node
    for concept_node in hasse.nodes:
        if hasse.in_degree(concept_node) == 0 and concept_node != "T":
            hasse.add_edge("T", concept_node, capacity=max_concepts)
        elif hasse.out_degree(concept_node) == 0 and concept_node != "⊥":
            hasse.add_edge(concept_node, "⊥", capacity=1)
    # find the min-cut
    extraneous_nodes = {"T", "⊥"}
    cut_value, partition = nx.minimum_cut(hasse, "T", "⊥")
    reachable, non_reachable = partition
    cutset = set([])
    for u, nbrs in ((n, hasse[n]) for n in reachable):
        for v in nbrs:
            if v in non_reachable:
                cutset.update([u, v])
    cutset = cutset.difference(extraneous_nodes)
    def get_best_atom(atom_list):
        best_atom, best_score = None, 0
        for atom in atom_list:
            accepting_num = 0
            for concept_node in hasse.nodes:
                if concept_node in extraneous_nodes:
                    continue
                if atom in hasse.nodes[concept_node]["concept"]:
                    accepting_num += 1
            score = (min(accepting_num, concept_num - accepting_num) / concept_num)
            best_atom = atom if score >= best_score else best_atom
            best_score = score if score >= best_score else best_score
        return best_atom
    # now, sample nodes from the symmetric difference
    symmetric_diff_concept = None
    atom_set1 = []
    atom_set2 = []
    for concept_node_id in cutset:
        if symmetric_diff_concept is None:
            symmetric_diff_concept = hasse.nodes[concept_node_id]["concept"]
            print("getting atoms")
            atom_set1 = fn.take(max_atoms, symmetric_diff_concept)
            print("getting atoms")
            atom_set2 = fn.take(max_atoms, ~symmetric_diff_concept)
            breakpoint()
            if len(atom_set1) == 0:
                atom_x = get_best_atom(atom_set2)
                atom_set2.remove(atom_x)
                atom_y = get_best_atom(atom_set2)
                return atom_x, atom_y
            if len(atom_set2) == 0:
                atom_x = get_best_atom(atom_set1)
                atom_set1.remove(atom_x)
                atom_y = get_best_atom(atom_set1)
                return atom_x, atom_y
        else:
            symmetric_diff_concept = symmetric_diff_concept ^ hasse.nodes[concept_node_id]["concept"]
        new_atom_set1 = fn.take(max_atoms, symmetric_diff_concept)
        new_atom_set2 = fn.take(max_atoms, ~symmetric_diff_concept)
        if len(new_atom_set1) == 0 or len(new_atom_set2) == 0:
            break
        atom_set1 = new_atom_set1
        atom_set2 = new_atom_set2
    # evaluate each atom sampled for its reduction in concept class size
    atm1, atm2 = get_best_atom(atom_set1), get_best_atom(atom_set2)
    breakpoint()
    return atm1, atm2
    #return get_best_atom(atom_set1), get_best_atom(atom_set2)

def create_learner(
        gen_concepts: ConceptClass,
        membership_cost: float,
        compare_cost: float,
        query_limit: int = 50,
        max_concepts: int = 15,
        max_atom_samples: int = 20) -> LearningAPI:
    """Create learner for interactiving learning a concept φ*.

    Assumes that atoms form a membership respecting pre-order, i.e.,

        (x ≼ y) ⇒ (x ∈ φ* → y ∈ φ*)

    Arguments:
      - membership_cost: Cost of testing if a atom is part of a concept.
      - compare_cost: Cost of comparing two atoms.

    Yields:
      One of three kinds of queries:
         1. Membership ('∈', [x]): Is x ∈ φ.

         2. Comparison ('≺', [x, y]): Compare x, y under ≺,
            e.g., x ≺ y or x || y.

         3. Equivalence ('≡', [φ]): Is φ ≡ φ*.

    Receives:
      One of the following depending on the type of query:
         1. Membership: Returns □ ∈ {'∈', '∉'} such that x □ φ*.

         2. Comparison: Returns □ ∈ {'≺', '≻', '=', '||'} such that x □ y.

         3. Equivalence: Returns (Query, Response) pair proving that φ ̸≡ φ*.
            If φ ≡ φ*, then no return is expected. The query and response
            correspond to membership or comparisons.
    """
    assumptions = []
    response = None
    gen_concepts = partial(gen_concepts, assumptions=assumptions)
    query_selector = QuerySelector(gen_concepts, membership_cost, compare_cost)
    known_queries = {}

    while True:
        concepts = gen_concepts()

        if (concept1 := next(concepts, None)) is None:
            return                                         # |Φ| = 0.

        queries = None
        if len(assumptions) > query_limit:  # max num. of queries reached
            yield '≡', concept1
            return

        if (concept2 := next(concepts, None)) is None:
            query = ('≡', concept1)                        # |Φ| = 1.
        else:
            concepts = gen_concepts()

            left, right = find_maximally_distinguishing(concepts)
            left = "" if left is None else left
            right = "" if right is None else right  # hacky fix to fn.take returning "()"
            queries = [('∈', left), ('≺', (left, right))]
            query = query_selector(queries)

        if query in known_queries:
            response = known_queries[query]
        else:
            response = yield query

        if queries is not None:
            query_selector.update(response)
        queries = None
        
        known_queries[query] = response
        if query[0] != '≡':  # Equiv responses contain query already.
            response = (query, response)

        assumptions.append(response)


# ===================== Bandit Details =====================


def worst_case_smax_comparable(summaries, _):
    vals = [max(v for k, v in s.items() if k != '||') for s in summaries]
    return softmax(-np.array(vals))


def historical_smax(summaries, hist):
    vals = []
    for summary in summaries:
        kind = '∈' if '∈' in summary else '≺'
        dist = hist[kind]
        val = sum(v * dist[k] for k, v in summary.items())
        val /= sum(dist.values())
        vals.append(val)
    return softmax(-np.array(vals))


EXPERTS = [
    worst_case_smax_comparable,
    historical_smax,
]


MemOrCmpResponse = Union[MemResponse, CmpResponse]
ResultMap = dict[MemOrCmpResponse, float]
Hist = dict[Literal['∈', '≺'], Counter[MemOrCmpResponse]]


@attr.s(auto_detect=True, auto_attribs=True)
class QuerySelector:
    # Parameters
    gen_concepts: ConceptClass
    mem_cost: float
    cmp_cost: float

    n_trials: int = 10  # Used for monte carlo summarization.

    # Internal State
    player: exp4.Player = attr.ib(factory=exp4.exp4)
    loss: float | None = None
    loss_map: Optional[ResultMap] = None
    hist: Hist = attr.ib(factory=lambda: {
        '∈': Counter(['∈', '∉']), '≺': Counter(['≺', '≻', '=', '||'])
    })

    def __call__(self, queries: list[Query]) -> Query:
        assert self.loss_map is None, "Must update selector with loss."

        loss_maps = []
        for q in queries:

            cost = self.mem_cost if q[0] == '∈' else self.cmp_cost
 
            loss_maps.append({
                k: cost*s for k, s in self.summarize(q).items()
            })

        advice = [expert(loss_maps, self.hist) for expert in EXPERTS]

        arm = self.player.send((self.loss, advice))

        # Setup loss_map.
        self.loss_map = loss_maps[arm]
        self.loss = None
        return queries[arm]

    def update(self, response: MemResponse | CmpResponse):
        assert self.loss is None, "Must first select a query!."

        if '∈' in self.loss_map:
            self.hist['∈'].update(response)
        else:
            self.hist['≺'].update(response)

            
        self.loss, self.loss_map = self.loss_map[response], None
        self.loss /= max(self.mem_cost, self.cmp_cost)

    def summarize(self, query: Query) -> ResultMap:
        if query[0] == '∈':
            x = query[1]
            tests = {'∈': lambda c: x in c, '∉': lambda c: x not in c}
        elif query[0] == '≺':
            left, right = query[1]
            tests = {
                '≺': lambda c: (left in c) <= (right in c),
                '≻': lambda c: (right in c) <= (left in c),
                '=': lambda c: (left in c) == (right in c),
                '||': lambda c: True,
            }
        else:
            raise NotImplementedError

        # Estimate concept class reduction per outcome via naïve monte carlo.
        trials = fn.take(self.n_trials, self.gen_concepts())
        return {k: sum(map(t, trials)) / len(trials) for k, t in tests.items()}
