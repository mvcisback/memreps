from __future__ import annotations

from itertools import product, repeat
from typing import Callable, Protocol, Iterable, Sequence

import attr
import numpy as np


from memreps import Atom, ConceptClass
from memreps.implicit import create_implicit_concept_class, Predicate


Parameters = Sequence[float]  # Should be between 0 and 1.
ParameterizedPredicate = Callable[[Parameters], Predicate]
eta = 1e-12

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

def h(x: np.ndarray, p: np.ndarray) -> float:
    """
    Computes height of plane at point specified by x
    Note that x has size n-1, p has size n, where n is number of dimensions of parameter space
    """
    return np.dot(x, p[:-1]) + p[-1]

def sep_hyperplane(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Given 2 points, p1 and p2, generate a separating hyperplane for the two
    To extract p1 and p2, we have to go into Concept, extract Predicate, and look at Parameters 
    
    Hyperplane is represented as: x_0 = a_1*x_1 + a_2*x_2 + ... + a_n, where a_i are float parameters
    Constraints:
        - all gradients (ie parameters other than a_n) must be negative
        - hyperplane must be separating

    output:
    ndarray containing a_1, a_2, ...
    """
    limit = 10.0
    dim = p1.shape[0]
    
    A = np.zeros((dim))
    h1 = 0
    h2 = 0
    while abs(h1 - h2) < eta:
        # Generating random gradients (a_n = 0 for now)
        A[:-1] = -np.random.default_rng().uniform(0, limit, size = (dim-1))

        # Check if separating hyperplane is possible with generated gradients
        # We do this by calculating x_0 of hyperplane at p1 and p2's locations
        # These must differ by at least eta, for there to be some a_n that separates the two
        h1 = np.dot(p1[1:], A[:-1])
        h2 = np.dot(p2[1:], A[:-1])

    delta1 = p1[0] - h1
    delta2 = p2[0] - h2
    A[-1] = np.random.default_rng().uniform(min(delta1, delta2), max(delta1, delta2))
    return A

def nonsep_hyperplane(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Given 2 points, p1 and p2, generate a non-separating hyperplane for the two
    To extract p1 and p2, we have to go into Concept, extract Predicate, and look at Parameters 
    This basically finds atoms that are not in symmetrix difference of 2 concepts

    Hyperplane is represented as: x_0 = a_1*x_1 + a_2*x_2 + ... + a_n, where a_i are float parameters
    Constraints:
        - all gradients (ie parameters other than a_n) must be negative
        - hyperplane must be separating

    output:
    ndarray containing a_1, a_2, ...
    """
    limit = 10.0
    dim = p1.shape[0]
    
    A = np.zeros((dim))
    A[:-1] = -np.random.default_rng().uniform(0, limit, size = (dim-1))
    h1 = np.dot(p1[1:], A[:-1])
    h2 = np.dot(p2[1:], A[:-1])

    delta1 = p1[0] - h1
    delta2 = p2[0] - h2
    lower = min(delta1, delta2)
    upper = max(delta1, delta2)
    if np.random.default_rng().integers(2):
        A[-1] = np.random.default_rng().uniform(0, lower - eta)
    else:
        A[-1] = np.random.default_rng().uniform(upper + eta, 1 - np.sum(A[:-1]))
    return A

def hyperplane_mem(concept: np.ndarray, plane: np.ndarray) -> bool:
    """
    Tests whether a specified atom (hyperplane) belongs to a concept (point in parameter space)
    Here we assume that membership holds if point is below the plane (ie higher values of parameter result in stricter predicate)
    """
    return (concept[0] - h(concept[1:], plane) < -eta)
