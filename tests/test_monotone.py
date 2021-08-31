import numpy as np

from memreps import memreps
from memreps.monotone import nonsep_hyperplane, MonotoneConcept


def h(x, p):
    return x @ p[:-1] + p[-1]


def test_sep_hyperplane():
    # 3D testing
    dim = 3
    for _ in range(10):
        p1 = np.random.default_rng().uniform(0, 1, size = (dim))
        p2 = np.random.default_rng().uniform(0, 1, size = (dim))

        concept1 = MonotoneConcept.from_point(p1)
        concept2 = MonotoneConcept.from_point(p2)

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
