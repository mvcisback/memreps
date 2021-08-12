from itertools import groupby
import attr
from typing import Any, Optional, Tuple, Callable, List, Dict, Iterable
from dfa import DFA
from dfa.utils import find_equiv_counterexample, find_subset_counterexample, find_word
from memreps.memreps import create_learner, MemQuery, CmpQuery, EqQuery
from memreps.memreps import Atom, Assumptions, Response, Query, Concept, Literal

from dfa_identify.identify import find_dfa, find_dfas

class DFAConcept(Concept):
    def __init__(self, dfa: DFA):
        self.dfa = dfa

    def __in__(self, atom: Atom) -> bool:
        return self.dfa.label(atom)

    def __xor__(self, other: Concept) -> Concept:
        return (self.dfa ^ other)

    def __neg__(self) -> Concept:
        return ~self.dfa

    def __iter__(self) -> Iterable[Atom]:
        yield find_word(self.dfa)


def dfa_memreps(
        accepting: List[Atom],
        rejecting: List[Atom],
        oracle: Callable[[Query], Response],
        membership_cost: float,
        compare_cost: float,
        query_limit: int = 50,
        strong_memrep: bool = False,
        ordered_preference_words: list[Tuple[Atom, Atom]] = None,
        incomparable_preference_words: list[Tuple[Atom, Atom]] = None
):
    if ordered_preference_words is None:
        ordered_preference_words = []
    if incomparable_preference_words is None:
        incomparable_preference_words = []

    # create wrapper for DFA concept class
    def concept_class_wrapper(assumptions: Assumptions):
        tmp_accepting = accepting[:]
        tmp_rejecting = rejecting[:]
        tmp_ordered_preference_words = ordered_preference_words[:]
        tmp_incomparable_preference_words = incomparable_preference_words[:]

        for query, response in assumptions:
            query_symbol, query_atom = query
            if query_symbol == '∈':
                query_id, word = query
                if response == '∈':  # accepting case
                    tmp_accepting.append(word)
                else:
                    tmp_rejecting.append(word)
            else:  # we are in a CmpQuery
                query_id, word_pair = query
                if response == '≺':
                    tmp_ordered_preference_words.append(word_pair)  # default is (less, more)
                elif response == '≻':
                    tmp_ordered_preference_words.append((word_pair[1], word_pair[0]))
                elif response == '=':
                    tmp_incomparable_preference_words.append(word_pair)
                elif response == '||':
                    if strong_memrep:
                        tmp_incomparable_preference_words.append(word_pair)
        gnr = find_dfas(tmp_accepting, tmp_rejecting, tmp_ordered_preference_words,
                         tmp_incomparable_preference_words)
        yield DFAConcept(next(gnr))
    #  create initial learner and generate initial query
    dfa_learner = create_learner(concept_class_wrapper, membership_cost, compare_cost)
    query = dfa_learner.send(None)
    for _ in range(query_limit):
        # get a response from the oracle
        response = oracle(query)
        query_type, concept = query
        if query_type == '≡':
            # we are in an equivalence query. return if we are indeed equivalent
            if response is None:
                return concept
        query = dfa_learner.send(response)
    query_type, concept = query
    return concept
