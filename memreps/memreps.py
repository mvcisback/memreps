from __future__ import annotations

from typing import (Any, Callable, Generator, Iterable, Literal,
                    Protocol, Sequence, TypeVar)

from exp4 import exp4


Atom = Any


class Concept(Protocol):
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
Assumptions = Sequnce[Assumption]
# Note: Generated concepts should all be unique!
ConceptClass = Callable[[Assumptions], Iterable[Concept]]


# ====================== Learner ===========================


def create_learner(
        find_concept: ConceptClass,
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
    bandit_alg = exp4()
    assumptions = []

    def choose_query(atom1, atom2) -> Query:
        pass

    while True:
        concepts = find_concept(assumptions)

        if (concept1 := next(concepts, None)) is None:
            return                                         # |Φ| = 0.

        if (concept2 := next(concepts, None)) is None:
            query = ('≡', concept1)                        # |Φ| = 1.
        else:
            concept12 = concept1 ^ concept2
            atom1, atom2 = next(concept12), next(~concept12)
            query = choose_query(atom1, atom2)

        response = yield query
        
        if query[0] != '≡':  # Equiv responses contain query already. 
            response = (query, response)
 
        assumptions.append(response)
