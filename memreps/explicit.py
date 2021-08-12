from __future__ import annotations

from typing import Any, Iterable, Sequence

import attr
from hasse import PoSet

from memreps import Atom, Assumption, Assumptions, Concept, ConceptClass


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
        labels, comparisons, equivs = {}, [], set()
        for query, response in assumptions:
            if query[0] == '∈':
                labels[query[1]] = (response == '∈')
                continue
            elif response == '||':
                continue

            left, right = query
            
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
            poset=PoSet.from_chains(comparisons),
        )

def create_explict_concept_class(
    universe: Sequence[Atom],
    concepts: Sequence[frozenset[Atom]]) -> ConceptClass:

    universe = frozenset(universe)

    @attr.frozen
    class ExplicitConcept:
        elements: frozenset[Atom]

        def __in__(self, atom: Atom) -> bool:
            return atom in self.elements

        def __xor__(self, other: ExplicitConcept) -> ExplicitConcept:
            return attr.evolve(self, elements=self.elements ^ other.elements)

        def __invert__(self) -> ExplicitConcept:
            return attr.evolve(self, elements=self.elements ^ universe)

        def __iter__(self) -> Iterable[Atom]:
            yield from self.elements

    concepts = [ExplicitConcept(c) for c in concepts]

    def concept_class(assumptions=()):
        lposet = LabeledPreSet.from_assumptions(assumptions)
        yield from (c for c in concepts if lposet.is_memrep(c))

    return concept_class
