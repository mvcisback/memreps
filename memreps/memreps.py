from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import (Any, Callable, Counter, Generator,
                    Iterable, Literal, Protocol, Union)

import attr
import exp4
import funcy as fn
import numpy as np
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


def create_learner(
        gen_concepts: ConceptClass,
        membership_cost: float,
        compare_cost: float,
        query_limit: int = 50) -> LearningAPI:
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

        queries = None
        if len(assumptions) > query_limit:  # max num. of queries reached
            yield next(concepts, None)
            return

        if (concept2 := next(concepts, None)) is None:
            query = ('≡', concept1)                        # |Φ| = 1.
        else:
            concept12 = concept1 ^ concept2
            atoms1 = fn.take(1, concept12)
            atoms2 = fn.take(1, ~concept12)

            if len(atoms1) == 0:
                query = ('∈', atoms2[0])
            elif len(atoms2) == 0:
                query = ('∈', atoms1[0])
            else:
                left, right = atoms1[0], atoms2[0]
                queries = [('∈', left), ('≺', (left, right))]
                query = query_selector(queries)

        response = yield query

        if queries is not None:
            query_selector.update(response)
        
        if query[0] != '≡':  # Equiv responses contain query already.
            response = (query, response)

        assumptions.append(response)


# ===================== Bandit Details =====================


def worst_case_smax(summaries, _):
    vals = [max(s.values()) for s in summaries]
    return softmax(1 - np.array(vals))


def worst_case_smax_comparable(summaries, _):
    vals = [min(v for k, v in s.items() if k != '||') for s in summaries]
    return softmax(1 - np.array(vals))


def historical_smax(summaries, hist):
    vals = []
    for summary in summaries:
        kind = '∈' if '∈' in summary else '≺'
        dist = hist[kind]
        val = sum((1 - v) * dist[k] for k, v in summary.items())
        val /= sum(dist.values())
        vals.append(val)
    return softmax(vals)


EXPERTS = [
    lambda summaries, _: [float('∈' in s) for s in summaries],
    lambda summaries, _: [float('≺' in s) for s in summaries],
    worst_case_smax,
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

        summaries = [self.summarize(q) for q in queries]
        advice = [expert(summaries, self.hist) for expert in EXPERTS]

        arm = self.player.send((self.loss, advice))
        query = queries[arm]

        # Setup loss_map.
        query_cost = self.mem_cost if query[0] == '∈' else self.cmp_cost
        query_cost /= max(self.mem_cost, self.cmp_cost)
        summary = summaries[arm]
        self.loss_map = {k: (query_cost + s) / 2 for k, s in summary.items()}
        self.loss = None

        return query

    def update(self, response: MemResponse | CmpResponse):
        assert self.loss is None, "Must first select a query!."

        if '∈' in self.loss_map:
            self.hist['∈'].update(response)
        else:
            self.hist['≺'].update(response)

        self.loss, self.loss_map = self.loss_map[response], None

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
