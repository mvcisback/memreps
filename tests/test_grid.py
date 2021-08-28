import attr
import numpy as np
import matplotlib.pyplot as plt

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
    params = np.array(concept.pred.params)
    assert params in concept
    assert params - 0.1 not in concept
    assert params + 0.1 in concept 

    left, right = (0, 0), (0.6, 0.6)
    assumptions = [(('≺', (left, right)), '≻')]

    assert set(gen_concepts(assumptions)) < set(gen_concepts())

def test_hyperplane():
    # 3D testing
    dim = 3
    for _ in range(10):
        p1 = np.random.default_rng().uniform(0, 1, size = (dim))
        p2 = np.random.default_rng().uniform(0, 1, size = (dim))
        plane = grid.sep_hyperplane(p1, p2)

        def h(x, p):
            return np.dot(x, p[:-1]) + p[-1]

        # print(p1, p2, plane)
        # checks whether hyperplane is really separating
        h1 = h(p1[1:], plane) - p1[0]
        h2 = h(p2[1:], plane) - p2[0]
        # h1 = np.dot(p1[1:], plane[:-1]) + plane[-1] - p1[0]
        # h2 = np.dot(p2[1:], plane[:-1]) + plane[-1] - p2[0]
        assert h1 * h2 < 0

    print(p1, p2, plane)
    print(h1, h2)

    # # plotting stuff
    # points = np.stack((p1, p2))
    # x = np.arange(0, 1.1, 0.2)
    # y = np.arange(0, 1.1, 0.2)
    # x, y = np.meshgrid(x, y)
    # z = np.reshape(h(np.stack((np.reshape(x,-1),np.reshape(y,-1))).T, plane), x.shape)
    # ax = plt.axes(projection = "3d")
    # ax.scatter3D(points[:,1], points[:,2], points[:,0])
    # ax.plot_surface(x, y, z)
    # ax.set_title(plane)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()