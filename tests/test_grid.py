import attr
import numpy as np

from memreps import memreps
from memreps import grid


@attr.frozen
class Predicate:
    params: tuple[float, float] = attr.ib(converter=tuple)

    def __call__(self, xs):
        return all(x >= p for x, p in zip(xs, self.params))


def point_sampler(pred):
    while True:
        yield tuple(np.random.rand(2))


def test_simple_grid_concept_class():
    gen_concepts = grid.create_grid_concept_class(
        family=Predicate,
        dim=2,
        elems=point_sampler,
        num_ticks=4,
    )

    concept = next(gen_concepts())

    assert (1, 1) in concept
