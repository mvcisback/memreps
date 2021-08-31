from __future__ import annotations

from itertools import product, repeat
from typing import Callable, Protocol, Iterable, Sequence

import attr
import numpy as np


from memreps import Atom, Concept, ConceptClass
from memreps.implicit import create_finite_concept_class, Predicate


Parameters = Sequence[float]  # Should be between 0 and 1.
ParameterizedConceptFamily = Callable[[Parameters], Concept]


def create_grid_concept_class(
    family: ParameterizedConceptFamily,
    dim: int,
    num_ticks: int) -> ConceptClass:

    """Creates a concept class based on a `ticks` × … × `ticks` grid of predicates.

    In particular, the grid is formed by product of `dim` uniform
    discretizations of [0, 1], i.e., {k/i : 0 ≤ k ≤ i}.
    """

    # Create 1d discretization and take product with itself dim times.
    points_1d = np.linspace(0, 1, num_ticks)
    points_nd = product(*repeat(points_1d, dim))
    concepts = map(family, points_nd)
    return create_finite_concept_class(concepts)

