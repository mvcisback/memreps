import random
from itertools import groupby
from typing import Any, Optional, Tuple, Callable, List, Dict, Iterable

import attr
import funcy as fn
from dfa import DFA
from dfa.utils import find_equiv_counterexample, find_subset_counterexample, find_word, words
from memreps.memreps import create_learner, MemQuery, CmpQuery, EqQuery
from memreps.memreps import Atom, Assumptions, Response, Query, Concept, Literal
from collections import Counter
from dfa_identify.identify import find_dfa, find_dfas


@attr.frozen
class DFAConcept:
    dfa: DFA

    def __contains__(self, atom: Atom) -> bool:
        return self.dfa.label(atom)

    def __xor__(self, other) -> Concept:
        return DFAConcept(self.dfa ^ other.dfa)

    def __invert__(self) -> Concept:
        return DFAConcept(~self.dfa)

    def __iter__(self) -> Iterable[Atom]:
        yield from words(self.dfa)

# create wrapper for DFA concept class
def create_dfa_concept_class(
    strong_memrep: bool = False,
    accepting: Optional[list[Atom]] = None,
    rejecting: Optional[list[Atom]] = None,
    ordered_preference_words: Optional[list[Tuple[Atom, Atom]]] = None,
    incomparable_preference_words: Optional[list[Tuple[Atom, Atom]]] = None,
    alphabet: Iterable[Atom] = None
):
    if accepting is None:
       accepting = []
    if rejecting is None:
        rejecting = []
    if ordered_preference_words is None:
        ordered_preference_words = []
    if incomparable_preference_words is None:
        incomparable_preference_words = []

    def concept_class(assumptions):
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
                         tmp_incomparable_preference_words, alphabet=alphabet)
        gnr = map(DFAConcept, gnr)

        for batch in fn.chunks(20, gnr):
           random.shuffle(batch)
           yield from batch

    return concept_class




def dfa_memreps(
    oracle: Callable[[Query], Response],
    membership_cost: float,
    compare_cost: float,
    query_limit: int = 50,
    strong_memrep: bool = False,
    accepting: Optional[list[Atom]] = None,
    rejecting: Optional[list[Atom]] = None,
    ordered_preference_words: Optional[list[Tuple[Atom, Atom]]] = None,
    incomparable_preference_words: Optional[list[Tuple[Atom, Atom]]] = None,
    force_membership: bool = False,
    alphabet: Iterable[Atom] = None,
):
    if force_membership:
        compare_cost = 100 * membership_cost
    concept_class = create_dfa_concept_class(
        strong_memrep=strong_memrep,
        accepting=accepting,
        rejecting=rejecting,
        ordered_preference_words=ordered_preference_words,
        incomparable_preference_words=incomparable_preference_words,
        alphabet=alphabet
    )

     #  create initial learner and generate initial query
    dfa_learner = create_learner(concept_class, membership_cost, compare_cost, query_limit=query_limit)
    query = dfa_learner.send(None)
    query_histogram = Counter()
    for itr in range(query_limit):
        # get a response from the oracle
        response = oracle(query)
        query_type, concept = query
        query_histogram.update(query_type)
        if query_type == '≡':
            # we are in an equivalence query. return if we are indeed equivalent
            if response is None:
                return concept, query_histogram
        query = dfa_learner.send(response)
        print("On iteration: ", itr)
        print(query)
    query_type, concept = query
    return concept, query_histogram
