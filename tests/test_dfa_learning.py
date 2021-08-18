from dfa_identify import find_dfa, find_dfas
from dfa.utils import find_equiv_counterexample, find_subset_counterexample
from dfa import dict2dfa
from memreps.dfa_learning import dfa_memreps
import numpy as np

def test_equivalence_memreps():
    def oracle(full_query):
        query_symbol, query = full_query
        if query_symbol == '∈':
            # mem query case
            return membership_fxn(query)
        elif query_symbol == '≺':
            # cmp query case
            return pref_fxn(query[0], query[1])
        else:
            # eq query case
            resp = equivalence_fxn(query.dfa)
            if resp is None:
                return None
            else:
                counterex, label = resp
                return ('∈', counterex), label

    transition_dict = {0: (True, {'a': 1, 'b': 0}),
                       1: (True, {'a' : 4, 'b': 2}),
                       2: (True, {'a' : 5, 'b': 3}),
                       3: (False, {'a' : 3, 'b': 3}),
                       4: (True, {'a' : 2, 'b': 4}),
                       5: (True, {'a' : 3, 'b': 5})}

    true_dfa = dict2dfa(transition_dict, 0)

    def pref_fxn(word1, word2):
        if true_dfa.label(word1) == true_dfa.label(word2):
            return '||'
        elif true_dfa.label(word1):
            return '≻'
        else:
            return '≺'

    def membership_fxn(word):
        if true_dfa.label(word):
            return '∈'
        else:
            return '∉'

    def equivalence_fxn(candidate):
        cex = find_equiv_counterexample(candidate, true_dfa)
        if cex is None:
            return None
        else:
            if true_dfa.label(cex):
                return cex, '∈'
            else:
                return cex, '∉'

    accepting = ['b', 'aa', 'a']
    rejecting = ['aaaaa', 'abb']

    resulting_dfa = dfa_memreps(accepting, rejecting, oracle, 2, 1)
    assert find_equiv_counterexample(true_dfa, resulting_dfa.dfa) is None


def test_gridworld_dfa():
    def oracle(full_query):
        query_symbol, query = full_query
        if query_symbol == '∈':
            # mem query case
            return membership_fxn(query)
        elif query_symbol == '≺':
            # cmp query case
            return pref_fxn(query[0], query[1])
        else:
            # eq query case
            resp = equivalence_fxn(query.dfa)
            if resp is None:
                return None
            else:
                counterex, label = resp
                return ('∈', counterex), label
    transition_dict = {0: (False, {'R': 0, 'B': 0, 'O': 0, 'W': 0, 'Y': 0}),
                       1: (False, {'R': 0, 'B': 2, 'O': 1, 'W': 1, 'Y': 1}),
                       2: (False, {'R': 0, 'B': 2, 'O': 3, 'W': 2, 'Y': 0}),
                       3: (False, {'R': 0, 'B': 2, 'O': 4, 'W': 3, 'Y': 0}),
                       4: (False, {'R': 0, 'B': 2, 'O': 4, 'W': 4, 'Y': 5}),
                       5: (True,  {'R': 0, 'B': 5, 'O': 5, 'W': 5, 'Y': 5})}

    true_dfa = dict2dfa(transition_dict, 1)

    def pref_fxn(word1, word2):
        trace1 = list(true_dfa.trace(word1))
        trace2 = list(true_dfa.trace(word2))
        if trace1[-1] > trace2[-1]:
            return '≻'
        elif trace2[-1] > trace1[-1]:
            return '≺'
        else:
            return '||'

    def membership_fxn(word):
        if true_dfa.label(word):
            return '∈'
        else:
            return '∉'

    def equivalence_fxn(candidate):
        cex = find_equiv_counterexample(candidate, true_dfa)
        if cex is None:
            return None
        else:
            if true_dfa.label(cex):
                return cex, '∈'
            else:
                return cex, '∉'

    accepting = ['WBOOYY', 'WBWBWOWOWY']
    rejecting = ['R', 'WYBY']

    resulting_dfa = dfa_memreps(accepting, rejecting, oracle, 2, 1, query_limit=150)
    assert find_equiv_counterexample(true_dfa, resulting_dfa.dfa) is None