from __future__ import annotations

from typing import Iterable, Sequence

import attr

from memreps.memreps import Atom, Assumption, Assumptions, ConceptClass


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class ExplicitConcept:
    universe: frozenset[Atom]
    elements: frozenset[Atom]

    def __in__(self, atom: Atom) -> bool:
        return atom in self.elements

    def __xor__(self, other: ExplicitConcept) -> ExplicitConcept:
        return attr.evolve(self, elements=self.elements ^ other.elements)

    def __neg__(self) -> ExplicitConcept:
        return attr.evolve(self, elements=self.elements ^ self.universe)

    def __iter__(self) -> Iterable[Atom]:
        yield from self.elements

    def _satisifies(self, assumption: Assumption) -> bool:
        query, response = assumption
        if query[0] == '∈':
            return (query[1] in self) == ('∈' == response)
        else:
            left, right = query[1]
            left, right = (left in self), (right in self)

            if response == '≺':
                return left <= right
            elif response == '≻':
                return right <= left
            elif response == '=':
                return left == right
            return True

    def satisifies(self, assumptions: Assumptions) -> bool:
        return all(map(self._satisifies, assumptions))


def create_explict_concept_class(
    universe: Sequence[Atom],
    concepts: Sequence[frozenset[Atom]]) -> ConceptClass:

    universe = frozenset(universe)
    concepts = [ExplicitConcept(universe, c) for c in concepts]

    def concept_class(assumptions=()):
        yield from (c for c in concepts if c.satisifies(assumptions))

    return concept_class
