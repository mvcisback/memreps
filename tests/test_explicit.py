from memreps import memreps
from memreps import explicit
from memreps import finite


def test_simple_concept_class():
    gen_concepts = explicit.create_explict_concept_class(
        universe = {1, 2},
        concepts = [{2}, {1}, {1, 2}],
    )

    concept = next(gen_concepts())
    assert concept.elements == {2}

    assumptions = [(('≺', (1, 2)), '≺')]
    lpreset = finite.LabeledPreSet.from_assumptions(assumptions)
    assert lpreset.support == {1, 2}

    assert set(gen_concepts()) > set(gen_concepts(assumptions))

    assumptions = [(('≺', (1, 2)), '≺'), (('∈', 1), '∈')]
    assert len(set(gen_concepts(assumptions))) == 1
    
    assumptions = [(('≺', (1, 2)), '≺'), (('∈', 2), '∉')]
    assert set(gen_concepts(assumptions)) == set()

    assumptions = [(('≺', (1, 2)), '≺'), (('≺', (1, 2)), '≻')]
    assert set(gen_concepts(assumptions)) == set()

