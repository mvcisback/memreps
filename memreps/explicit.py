from __future__ import annotations

from typing import Any, Iterable

import attr

from memreps import Atom,  ConceptClass
from memreps.finite import create_finite_concept_class



def create_explict_concept_class(
    universe: Iterable[Atom],
    concepts: Iterable[frozenset[Atom]]) -> ConceptClass:

    universe = frozenset(universe)

    @attr.frozen
    class ExplicitConcept:
        elements: frozenset[Atom] = attr.ib(converter=frozenset)

        def __in__(self, atom: Atom) -> bool:
            return atom in self.elements

        def __xor__(self, other: ExplicitConcept) -> ExplicitConcept:
            return attr.evolve(self, elements=self.elements ^ other.elements)

        def __invert__(self) -> ExplicitConcept:
            return attr.evolve(self, elements=self.elements ^ universe)

        def __iter__(self) -> Iterable[Atom]:
            yield from self.elements

    return create_finite_concept_class(ExplicitConcept(c) for c in concepts)
