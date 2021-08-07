from __future__ import annotations

from functools import partial
from typing import (Any, Callable, Counter, Generator,
                    Iterable, Literal, Protocol)

import attr
import exp4
import funcy as fn
from exp4.algo import softmax


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

         3. Equivalence: Returns (Query, Response) pair proving that φ ̸≡ φ*.
            If φ ≡ φ*, then no return is expected. The query and response
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


def worst_case_smax(summaries, _):
    vals = [min(s.values()) for s in summaries]
    return softmax(vals)


def worst_case_smax_comparable(summaries, _):
    vals = [min(v for k, v in s.items() if k != '||') for s in summaries]
    return softmax(vals)


def historical_smax(summaries, hist):
    vals = []
    for summary in summaries:
        kind = '∈' if '∈' in summary else '≺'
        val = sum(v * hist[k] for k, v in summary.items())
        vals.append(val)
    return softmax(vals)


EXPERTS = [
    lambda summaries, _: [float('∈' in s) for s in summaries],
    lambda summaries, _: [float('≺' in s) for s in summaries],
    worst_case_smax,
    worst_case_smax_comparable,
    historical_smax,
]


ResultMap = dict[MemResponse | CmpResponse, float]
Hist = dict[Literal['∈', '≺'], Counter[MemResponse, CmpResponse, float]]


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
    loss_map: ResultMap = attr.ib(factory=dict)
    hist: Hist = attr.ib(factory=lambda: {'∈': Counter(), '≺': Counter()})

    def __call__(self, queries: list[Query]) -> Query:
        summaries = [self.summarize(q) for q in queries]
        advice = [expert(summaries, self.hist) for expert in EXPERTS]

        arm = self.player.send((self.loss, advice))
        query = queries[arm]

        # Setup loss_map.
        query_cost = self.mem_cost if query[0] == '∈' else self.cmp_cost
        query_cost /= max(self.mem_cost, self.cmp_cost)
        self.loss_map = {k: (query_cost + s) / 2 for k, s in summaries[arm]}

        return query

    def update(self, response: MemResponse | CmpResponse):
        self.loss = self.loss_map[response]
        if '∈' in self.loss_map:
            self.hist['∈'].update(response)
        else:
            self.hist['≺'].update(response)

    def summarize(self, query: Query) -> ResultMap:
        match query:
            case ('∈', atom):
                tests = {'∈': lambda c: x in c, '∉': lambda c: x not in c}

            case ('≺', (left, right)):
                tests = {
                    '≺': lambda c: (left in c) <= (right in c),
                    '≻': lambda c: (right in c) <= (left in c),
                    '=': lambda c: (left in c) == (right in c),
                    '||': lambda c: True,
                }

            case _:
                raise NotImplementedError

        # Estimate concept class reduction per outcome via naïve monte carlo.
        concepts = fn.take(self.n_trials, gen_concepts())
        return {k: sum(map(t, concepts)) / len(concepts) for k, t in tests}
