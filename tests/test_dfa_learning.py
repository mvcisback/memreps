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
            return equivalence_fxn(query)

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
                return ('∈', cex)
            else:
                return ('∉', cex)

    accepting = ['b', 'aa', 'a']
    rejecting = ['aaaaa', 'abb']

    resulting_dfa = dfa_memreps(accepting, rejecting, oracle, 10,1)
    assert find_equiv_counterexample(true_dfa, resulting_dfa) is None