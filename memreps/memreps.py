from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generator, Iterable, Literal, Protocol

import attr
import exp4


Atom = Any


class Concept(Protocol):
    def __in__(self, atom: Atom) -> bool:
        ...

    def __xor__(self, other: Concept) -> Concept:
        ...

    def __neg__(self) -> Concept:
        ...

    def __iter__(self) -> Iterable[Atom]:
        ...


# ================== Learning API ==========================

# Types of learning queries.
MemQuery = tuple[Literal['∈'], Atom]
CmpQuery = tuple[Literal['≺'], tuple[Atom, Atom]]
EqQuery = tuple[Literal['≡'], Concept]
Query = MemQuery | CmpQuery | EqQuery

# Types of responses from teacher.
MemResponse = Literal['∈', '∉']
CmpResponse = Literal['≺', '≻', '=', '||']
EqResponse = tuple[MemQuery, MemResponse] | tuple[CmpQuery, CmpResponse]
Response = MemResponse | CmpResponse | EqResponse

LearningAPI = Generator[Query, Response, None]

# ================== Concept Class =========================

Assumption = EqResponse
Assumptions = list[Assumption]
# Note: Generated concepts should all be unique!
ConceptClass = Callable[[Assumptions], Iterable[Concept]]

# ====================== Learner ===========================


def create_learner(
        gen_concepts: ConceptClass,
        membership_cost: float,
        compare_cost: float) -> LearningAPI:
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

         3. Equivalence: Returns (Query, Response) pair proving that φ ≠ φ*.
            If φ = φ*, then no return is expected. The query and response
            correspond to membership or comparisons.
    """
    assumptions = []
    response = None
    gen_concepts = partial(gen_concepts, assumptions=assumptions)
    query_selector = QuerySelector(gen_concepts, membership_cost, compare_cost)

    while True:
        concepts = gen_concepts()

        if (concept1 := next(concepts, None)) is None:
            return                                         # |Φ| = 0.

        if (concept2 := next(concepts, None)) is None:
            query = ('≡', concept1)                        # |Φ| = 1.
        else:
            concept12 = concept1 ^ concept2
            atom1, atom2 = next(concept12), next(~concept12)

            queries = [('∈', atom1), ('≺', (atom1, atom2))]
            query = query_selector(queries)

        response = yield query
        
        if query[0] != '≡':  # Equiv responses contain query already. 
            query_selector.update(response)
            response = (query, response)
 
        assumptions.append(response)


# ===================== Bandit Details =====================

# TODO: implement experts.
EXPERTS = []


@attr.s(auto_detect=True, auto_attribs=True)
class QuerySelector:
    gen_concepts: ConceptClass
    mem_cost: float
    cmp_cost: float

    # Internal State
    player: exp4.Player = attr.ib(factory=exp4.exp4)
    loss: float | None = None
    loss_map: dict[MemResponse | CmpResponse, float] = attr.ib(factory=dict)

    def __call__(self, queries: list[Query]) -> Query:
        summaries = [self.summarize(q) for q in queries]
        advice = [expert(summary) for expert in EXPERTS]

        arm = self.player.send((self.loss, advice))
        query = queries[arm]

        # Setup loss_map
        query_cost = self.mem_cost if query[0] == '∈' else self.cmp_cost
        query_cost /= max(self.mem_cost, self.cmp_cost)
        self.loss_map = {k: (query_cost + s) / 2 for k, s in summaries[arm]}

        return query

    def update(self, response: MemResponse | CmpResponse):
        self.loss = self.loss_map[response]

    def summarize(self) -> dict[MemResponse | CmpResponse, float]:
        ...
