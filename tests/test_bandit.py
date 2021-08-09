from scipy.special import softmax

from memreps import memreps


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
