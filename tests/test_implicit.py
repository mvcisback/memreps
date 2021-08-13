from memreps import memreps
from memreps import implicit


def test_simple_implicit_concept_class():
    gen_concepts = implicit.create_implicit_concept_class(
        elems=lambda pred: filter(pred, {1, 2}),
        concepts = [
            lambda x: x == 2,
            lambda x: x == 1,
            lambda x: x in {1, 2},
        ]
    )

    concept = next(gen_concepts())
    assert (1 in concept) or (2 in concept)

    assumptions = [(('≺', (1, 2)), '≺')]

    assert set(gen_concepts()) > set(gen_concepts(assumptions))

    assumptions = [(('≺', (1, 2)), '≺'), (('∈', 1), '∈')]
    assert len(set(gen_concepts(assumptions))) == 1
    
    assumptions = [(('≺', (1, 2)), '≺'), (('∈', 2), '∉')]
    assert set(gen_concepts(assumptions)) == set()

    assumptions = [(('≺', (1, 2)), '≺'), (('≺', (1, 2)), '≻')]
    assert set(gen_concepts(assumptions)) == set()

