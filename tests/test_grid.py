import operator as op
from functools import reduce

import attr
import numpy as np

from memreps import memreps
from memreps import grid


def parity(xs) -> bool:
    return reduce(op.xor, xs)


@attr.frozen
class ThresholdFamily2d:
    params: frozenset[tuple[float, float]] = attr.ib(
        converter=lambda x: frozenset([tuple(x)])
    )  # List of parameters. This concept is xor or xnor of indexed concepts.
    negated: bool = False  # Picks xor or xnor.

    def __contains__(self, xs):
        val = parity(all(x >= p for x, p in zip(xs, ps)) for ps in self.params)
        return val ^ self.negated

    def __xor__(self, other):
        assert not (self.negated or other.negated)
        return attr.evolve(self, params=self.params | other.params)

    def __invert__(self):
        return attr.evolve(self, negated=not self.negated)

    def __iter__(self):
        while True:
            point = tuple(np.random.rand(2))
            if point in self:
                yield point


def test_simple_grid_concept_class():
    gen_concepts = grid.create_grid_concept_class(
        family=ThresholdFamily2d,
        dim=2,
        num_ticks=4,
    )

    concept = next(gen_concepts())

    assert (1, 1) in concept
    params = np.array(list(concept.params)[0])
    assert params in concept
    assert params - 0.1 not in concept
    assert params + 0.1 in concept 

    left, right = (0, 0), (0.6, 0.6)
    assumptions = [(('≺', (left, right)), '≻')]

    assert set(gen_concepts(assumptions)) < set(gen_concepts())
