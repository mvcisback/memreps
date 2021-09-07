from dfa_identify import find_dfa, find_dfas
from dfa.utils import find_equiv_counterexample, find_subset_counterexample
from dfa import dict2dfa
from memreps.dfa_learning import dfa_memreps, create_dfa_concept_class
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

    resulting_dfa = dfa_memreps(oracle, 2, 1, accepting=accepting, rejecting=rejecting)
    assert find_equiv_counterexample(true_dfa, resulting_dfa.dfa) is None


def test_gridworld_dfa(membership_cost: int = 1, force_membership: bool = False):
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
    transition_dict = {0: (False, {'R': 0, 'B': 0, 'O': 0, 'Y': 0}),
                       1: (False, {'R': 0, 'B': 1, 'O': 2, 'Y': 0}),
                       2: (False, {'R': 0, 'B': 1, 'O': 2, 'Y': 3}),
                       3: (True, {'R': 0, 'B': 1, 'O': 3, 'Y': 3})}


    true_dfa = dict2dfa(transition_dict, 2)

    def pref_fxn(word1, word2):
        if np.random.rand() < 0.1:
            response = '||'

        left, right = true_dfa.label(word1), true_dfa.label(word2)
        if left < right:
            return '≺'
        elif right < left:
            return '≻'
        else:
            return '=' if np.random.rand() > 0.3 else '≺'

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

    accepting = ['Y', 'YY', 'OY']
    rejecting = ['', 'R', 'RY', 'YR', 'BR', 'RB', 'OR', 'RO']

    resulting_dfa, query_hist = dfa_memreps(oracle, membership_cost, 1, query_limit=250,
                                            accepting=accepting, rejecting=rejecting, force_membership=force_membership)
    assert find_equiv_counterexample(true_dfa, resulting_dfa.dfa) is None
    return query_hist
