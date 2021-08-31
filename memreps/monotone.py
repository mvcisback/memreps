from __future__ import annotations

from typing import Callable, Sequence

import attr
import numpy as np
import funcy as fn

from memreps.memreps import Atom
from memreps.grid import create_grid_concept_class


@attr.frozen
class MonotoneConcept:
    points: frozenset[Sequence[float]]
    negated: bool = False  # Picks xor or xnor.
    eta: float = 1e-12

    @staticmethod
    def from_point(point: Sequence[float]) -> MonotoneConcept:
       return MonotoneConcept(frozenset([tuple(point)])) 

    def __xor__(self, other):
        points = self.points | other.points
        negated = self.negated ^ other.negated
        return attr.evolve(self, points=points, negated=negated)

    def __invert__(self):
        return attr.evolve(self, negated=not self.negated)

    def __contains__(self, plane):
        val = self.negated
        for point in self.points:
            val ^= hyperplane_mem(point, plane)
        return val

    def __iter__(self):
        points = self.points

        assert 1 <= len(points) <= 2

        if len(points) == 1:
            left = np.array(fn.first(points))
            right = np.zeros_like(left)
        else:
            left, right = map(np.array, points)

        get_plane = nonsep_hyperplane if self.negated else sep_hyperplane
        while True:
            yield get_plane(left, right, self.eta)
        


def sep_hyperplane(p1, p2, eta: float = 1e-12) -> np.ndarray:
    """Given 2 points, p1 and p2, generate a separating hyperplane for the two.

    To extract p1 and p2, we have to go into Concept, extract Predicate, and
    look at Parameters .
    
    Hyperplane is represented as: x_0 = a_1*x_1 + a_2*x_2 + ... + a_n, where
    a_i are float parameters:

    Constraints:
        - all gradients (ie parameters other than a_n) must be negative
        - hyperplane must be separating

    Returns:
      ndarray containing a_1, a_2, ...
    """
    limit = 10.0
    dim = p1.shape[0]
    
    A = np.zeros((dim))
    h1 = h2 = 0
    while abs(h1 - h2) < eta:
        # Generating random gradients (a_n = 0 for now)
        A[:-1] = -np.random.default_rng().uniform(0, limit, size = (dim-1))

        # Check if separating hyperplane is possible with generated gradients
        # We do this by calculating x_0 of hyperplane at p1 and p2's locations
        # These must differ by at least eta, for there to be some a_n that
        # separates the two.
        h1 = p1[1:] @ A[:-1]
        h2 = p2[1:] @ A[:-1]

    delta1 = p1[0] - h1
    delta2 = p2[0] - h2
    A[-1] = np.random.default_rng().uniform(min(delta1, delta2), max(delta1, delta2))
    return A


def nonsep_hyperplane(p1: np.ndarray, p2: np.ndarray, eta: float = 1e-12) -> np.ndarray:
    """Given 2 points, p1 and p2, generate a non-separating hyperplane.

    To extract p1 and p2, we have to go into Concept, extract Predicate, and
    look at Parameters.

    This basically finds atoms that are not in symmetric difference of 2
    concepts

    Hyperplane is represented as: x_0 = a_1*x_1 + a_2*x_2 + ... + a_n,
    where a_i are float parameters.
    
    Constraints:
        - all gradients (ie parameters other than a_n) must be negative
        - hyperplane must be separating

    Returns:
      ndarray containing a_1, a_2, ...
    """
    limit = 10.0
    dim = p1.shape[0]
    
    A = np.zeros((dim))
    A[:-1] = -np.random.default_rng().uniform(0, limit, size = (dim-1))
    h1 = p1[1:] @ A[:-1]
    h2 = p2[1:] @ A[:-1]

    delta1 = p1[0] - h1
    delta2 = p2[0] - h2
    lower = min(delta1, delta2)
    upper = max(delta1, delta2)
    if np.random.default_rng().integers(2):
        A[-1] = np.random.default_rng().uniform(0, lower - eta)
    else:
        A[-1] = np.random.default_rng().uniform(upper + eta, 1 - np.sum(A[:-1]))
    return A


def hyperplane_mem(concept, plane, eta: float = 1e-12) -> bool:
    """
    Tests whether a specified atom (hyperplane) belongs to a concept (point in parameter space)
    Here we assume that membership holds if point is below the plane (ie higher values of parameter result in stricter predicate)
    """
    height = concept[1:] @ plane[:-1] + plane[-1]
    return concept[0] - height < -eta
