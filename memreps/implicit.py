from __future__ import annotations

from typing import Any, Iterable, Callable

import attr

from memreps import Atom, ConceptClass
from memreps.finite import create_finite_concept_class


__all__ = ['create_implicit_concept_class']


Predicate = Callable[[Atom], bool]


def create_implicit_concept_class(
    elems: Callable[[Predicate], Iterable[Atom]],
    concepts: Iterable[Predicate]) -> ConceptClass:


    @attr.frozen
    class ImplicitConcept:
        pred: Predicate

        def __in__(self, atom: Atom) -> bool:
            return Predicate(atom) 

        def __xor__(self, other: ImplicitConcept) -> ImplicitConcept:
            def pred(x):
                return self.pred(x) ^ other.pred(x)

            return attr.evolve(self, pred=pred)

        def __invert__(self) -> ImplicitConcept:
            return attr.evolve(self, pred=lambda x: not self.pred(x))

        def __iter__(self) -> Iterable[Atom]:
            yield from elems(self.pred)

    return create_finite_concept_class(ImplicitConcept(c) for c in concepts)
