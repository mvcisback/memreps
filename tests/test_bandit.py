from scipy.special import softmax

from memreps import memreps
from memreps import explicit


def test_experts():
    summaries = [
       {'∈': 0.4, '∉': 0.6},
       {'≺': 0.3, '≻': 0.4, '=': 0.1, '||': 1}
    ]
    hist = {
        '∈': {'∈': 3, '∉': 4},
        '≺': {'≺': 2, '≻': 3, '=': 7, '||': 0},
    }
    advice = memreps.worst_case_smax(summaries, hist)
    assert advice.argmax() == 0

    advice = memreps.worst_case_smax_comparable(summaries, hist)
    assert advice.argmax() == 1

    advice = memreps.historical_smax(summaries, hist)
    assert advice.argmax() == 1
    
    # Inflate frequency of incomparable.
    hist = {
        '∈': {'∈': 3, '∉': 4},
        '≺': {'≺': 2, '≻': 3, '=': 1, '||': 100},
    }
    advice = memreps.historical_smax(summaries, hist)
    assert advice.argmax() == 0

    # Always membership.
    advice = memreps.EXPERTS[0](summaries, hist)
    assert advice == [1, 0]

    advice = memreps.EXPERTS[1](summaries, hist)
    assert advice == [0, 1]


def test_query_selector():
    gen_concepts = explicit.create_explicit_concept_class(
        universe = {1,2,3},
        concepts = [set(), {2}, {1}, {1, 2}],
    )
    assert len(list(gen_concepts())) == 4
    
    selector = memreps.QuerySelector(gen_concepts, 10, 1)

    queries = [('∈', 1), ('≺', (1, 2))]

    assert selector.summarize(queries[0]) == {'∈': 1/2, '∉': 1/2}
    assert selector.summarize(queries[1]) == {
        '≺': 3/4, '≻': 3/4, '=': 1/2, '||': 1,
    }
    
    for _queries in [queries, queries, queries]:
        query = selector(_queries)
        assert query in _queries

        if query[0] == '∈':
            assert selector.loss_map == {
                '∈': (1/2 + 1) / 2,
                '∉': (1/2 + 1) / 2,
            }
            selector.update('∈')
        else:
            assert selector.loss_map == {
                '≺': (3/4 + 1/10)/2,
                '≻': (3/4 + 1/10)/2,
                '=': (1/2 + 1/10)/2,
                '||': (1 + 1/10)/2,
            }
            selector.update('≺')
