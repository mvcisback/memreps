from itertools import combinations

import numpy as np
import funcy as fn

from memreps import memreps
from memreps.monotone import create_monotone_concept_class, MonotoneConcept


def h(x, p):
    return x @ p[:-1] + p[-1]


def validate_concept_pair(concept1, concept2):
    assert len(concept1.points) == len(concept2.points) == 1
    p1 = fn.first(concept1.points)
    p2 = fn.first(concept2.points)

    # Test ability to generate seperating hyperplane (atom).
    concept12 = concept1 ^ concept2
    plane = next(iter(concept12))
    h1 = h(p1[1:], plane) - p1[0]
    h2 = h(p2[1:], plane) - p2[0]
    assert h1 * h2 < 0
    assert (plane in concept1) != (plane in concept2)

    # Test ability to generate non-seperating hyperplane (atom).
    concept12 = ~concept12
    plane = next(iter(concept12))
    h1 = h(p1[1:], plane) - p1[0]
    h2 = h(p2[1:], plane) - p2[0]
    assert h1 * h2 > 0
    assert (plane in concept1) == (plane in concept2)


def test_random_points():
    # 3D testing
    dim = 3
    for _ in range(10):
        p1 = np.random.default_rng().uniform(0, 1, size = (dim))
        p2 = np.random.default_rng().uniform(0, 1, size = (dim))

        concept1 = MonotoneConcept.from_point(p1)
        concept2 = MonotoneConcept.from_point(p2)

        validate_concept_pair(concept1, concept2)


def test_monotone_grid():
    concept_class = create_monotone_concept_class(dim=3, num_ticks=4)
    
    for concept1, concept2 in combinations(concept_class(), 2):
        pass
        # validate_concept_pair(concept1, concept2)
