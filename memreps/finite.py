from __future__ import annotations

from typing import Any, Iterable, Sequence

import attr
import networkx as nx
from hasse import PoSet

from memreps import Atom, Assumptions, Concept, ConceptClass


__all__ = ['LabeledPreSet', 'create_finite_concept_class']


@attr.frozen
class LabeledPreSet:
    poset: Poset
    labels: dict[Any, bool]
    equivs: list[set[Any]]

    @property
    def edges(self) -> Sequence[tuple[Any, Any]]:
        return list(self.poset.hasse.edges)

    @property
    def support(self) -> frozenset[Any]:
        support = set(self.poset) | set(self.labels)
        if self.equivs:
            support |= set.union(*self.equivs)
        return support

    def is_memrep(self, concept: Concept) -> bool:
        labels = {x: x in concept for x in self.support}

        # Check equivalence classes yield same label under concept.
        if any(len({labels[x] for x in group}) != 1 for group in self.equivs):
            return False

        # Check concept's labels match observed labels.
        if any(labels[x] != label for x, label in self.labels.items()):
            return False

        # Check monotonicity of acceptance on chains.
        return all(labels[x] <= labels[y] for x, y in self.edges)

    @staticmethod
    def from_assumptions(assumptions: Assumptions):
        labels, comparisons, equivs = {}, [], []
        for query, response in assumptions:
            if query[0] == '∈':
                labels[query[1]] = (response == '∈')
                continue
            elif response == '||':
                continue

            left, right = query[1]
            
            if response == '≺':
                comparisons.append((left, right))
            elif response == '≻':
                comparisons.append((right, left))
            else:
                assert response == '='
                new_group = {left, right}
                for group in equivs:
                    if group & new_group: 
                        group |= {left, right}  # Add to existing group.
                        continue
                equivs.append(new_group)
        return LabeledPreSet(
            labels=labels,
            equivs=equivs,
            poset=PoSet.from_chains(*comparisons),
        )


def create_finite_concept_class(concepts: Iterable[Concept]) -> ConceptClass:
    concepts = list(concepts)

    def concept_class(assumptions=()):
        try:
            lposet = LabeledPreSet.from_assumptions(assumptions)
        except nx.NetworkXError:
            return
        yield from (c for c in concepts if lposet.is_memrep(c))

    return concept_class

