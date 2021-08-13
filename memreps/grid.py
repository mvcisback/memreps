from __future__ import annotations

from itertools import product, repeat
from typing import Callable, Protocol, Iterable, Sequence

import attr
import numpy as np


from memreps import Atom, ConceptClass
from memreps.implicit import create_implicit_concept_class, Predicate


Parameters = Sequence[float]  # Should be between 0 and 1.
ParameterizedPredicate = Callable[[Parameters], Predicate]


def create_grid_concept_class(
    family: ParameterizedPredicate,
    dim: int,
    elems: Callable[[Predicate], Iterable[Atom]],
    num_ticks: int) -> ConceptClass:

    """Creates a concept class based on a `ticks` × … × `ticks` grid of predicates.

    In particular, the grid is formed by product of `dim` uniform
    discretizations of [0, 1], i.e., {k/i : 0 ≤ k ≤ i}.
    """

    # Create 1d discretization and take product with itself dim times.
    points1d = np.linspace(0, 1, num_ticks)
    concepts = (family(p) for p in product(*repeat(points1d, dim)))

    return create_implicit_concept_class(elems, concepts)
