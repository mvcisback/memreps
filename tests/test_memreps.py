from memreps import memreps
from memreps import explicit


def test_memreps_smoke():
    gen_concepts = explicit.create_explict_concept_class(
        universe = {1, 2},
        concepts = [{2}, {1}, {1, 2}],
    )

    learner = memreps.create_learner(
        gen_concepts,
        membership_cost=10,
        compare_cost=1,
    )
 
    query = learner.send(None)

    for _ in range(2):
        kind, payload = query
        assert kind in {'∈', '≺'}

        if kind == '≺':
            assert set(payload) == {1, 2}
            query = learner.send('=')
        else:
            assert payload in {1, 2}
            query = learner.send('∈')

    kind, payload = query
    assert kind == '≡'
    assert payload.elements == {1, 2}
    breakpoint()
