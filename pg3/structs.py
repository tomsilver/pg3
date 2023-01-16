"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type."""
    name: str
    parent: Optional[Type] = field(default=None, repr=False)

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True, order=True, repr=False)
class _TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).

    Should not be instantiated externally.
    """
    name: str
    type: Type

    @cached_property
    def _str(self) -> str:
        return f"{self.name}:{self.type.name}"

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def is_instance(self, t: Type) -> bool:
        """Return whether this entity is an instance of the given type, taking
        hierarchical typing into account."""
        cur_type: Optional[Type] = self.type
        while cur_type is not None:
            if cur_type == t:
                return True
            cur_type = cur_type.parent
        return False


@dataclass(frozen=True, order=True, repr=False)
class Object(_TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name does
    not start with "?"."""

    def __post_init__(self) -> None:
        assert not self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Variable(_TypedEntity):
    """Struct defining a Variable, which is just a _TypedEntity whose name
    starts with "?"."""

    def __post_init__(self) -> None:
        assert self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Predicate:
    """Struct defining a predicate."""
    name: str
    types: Sequence[Type]

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __hash__(self) -> int:
        return self._hash

    @cached_property
    def arity(self) -> int:
        """The arity of this predicate (number of arguments)."""
        return len(self.types)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom:
    """Struct defining an atom (a predicate applied to either variables or
    objects).

    Should not be instantiated externally.
    """
    predicate: Predicate
    entities: Sequence[_TypedEntity]

    def __post_init__(self) -> None:
        if isinstance(self.entities, _TypedEntity):
            raise ValueError("Atoms expect a sequence of entities, not a "
                             "single entity.")
        assert len(self.entities) == self.predicate.arity
        for ent, pred_type in zip(self.entities, self.predicate.types):
            assert ent.is_instance(pred_type)

    @property
    def _str(self) -> str:
        raise NotImplementedError("Override me")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) < str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom):
    """Struct defining a lifted atom (a predicate applied to variables)."""

    @cached_property
    def variables(self) -> List[Variable]:
        """Arguments for this lifted atom.

        A list of "Variable"s.
        """
        return list(cast(Variable, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return (str(self.predicate) + "(" +
                ", ".join(map(str, self.variables)) + ")")

    def ground(self, sub: VarToObjSub) -> GroundAtom:
        """Create a GroundAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return GroundAtom(self.predicate, [sub[v] for v in self.variables])


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom):
    """Struct defining a ground atom (a predicate applied to objects)."""

    @cached_property
    def objects(self) -> List[Object]:
        """Arguments for this ground atom.

        A list of "Object"s.
        """
        return list(cast(Object, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return (str(self.predicate) + "(" + ", ".join(map(str, self.objects)) +
                ")")

    def lift(self, sub: ObjToVarSub) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.objects).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[o] for o in self.objects])


@dataclass(frozen=True, repr=False, eq=False)
class STRIPSOperator:
    """Struct defining a (lifted) symbolic operator (as in STRIPS)."""
    name: str
    parameters: Sequence[Variable]
    preconditions: Set[LiftedAtom]
    add_effects: Set[LiftedAtom]
    delete_effects: Set[LiftedAtom]

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> _GroundSTRIPSOperator:
        """Ground into a _GroundSTRIPSOperator, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        return _GroundSTRIPSOperator(self, list(objects), preconditions,
                                     add_effects, delete_effects)

    @cached_property
    def _str(self) -> str:
        return f"""STRIPS-{self.name}:
    Parameters: {self.parameters}
    Preconditions: {sorted(self.preconditions, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, STRIPSOperator)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, STRIPSOperator)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, STRIPSOperator)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class _GroundSTRIPSOperator:
    """A STRIPSOperator + objects.

    Should not be instantiated externally.
    """
    parent: STRIPSOperator
    objects: Sequence[Object]
    preconditions: Set[GroundAtom]
    add_effects: Set[GroundAtom]
    delete_effects: Set[GroundAtom]

    @cached_property
    def _str(self) -> str:
        return f"""GroundSTRIPS-{self.name}:
    Parameters: {self.objects}
    Preconditions: {sorted(self.preconditions, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this ground STRIPSOperator."""
        return self.parent.name

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundSTRIPSOperator)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundSTRIPSOperator)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundSTRIPSOperator)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LDLRule:
    """A lifted decision list rule."""
    name: str
    parameters: Sequence[Variable]  # a superset of the operator parameters
    pos_state_preconditions: Set[LiftedAtom]  # a superset of the preconds
    neg_state_preconditions: Set[LiftedAtom]
    goal_preconditions: Set[LiftedAtom]
    operator: STRIPSOperator

    def __post_init__(self) -> None:
        assert set(self.parameters).issuperset(self.operator.parameters)
        assert self.pos_state_preconditions.issuperset(
            self.operator.preconditions)
        # The preconditions and goal preconditions should only use variables in
        # the rule parameters.
        for atom in self.pos_state_preconditions | \
            self.neg_state_preconditions | self.goal_preconditions:
            assert all(v in self.parameters for v in atom.variables)

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> _GroundLDLRule:
        """Ground into a _GroundLDLRule, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        pos_pre = {atom.ground(sub) for atom in self.pos_state_preconditions}
        neg_pre = {atom.ground(sub) for atom in self.neg_state_preconditions}
        goal_pre = {atom.ground(sub) for atom in self.goal_preconditions}
        op_objects = tuple(sub[v] for v in self.operator.parameters)
        ground_op = self.operator.ground(op_objects)
        return _GroundLDLRule(self, list(objects), pos_pre, neg_pre, goal_pre,
                              ground_op)

    @cached_property
    def _str(self) -> str:
        op_param_str = ", ".join([str(v) for v in self.operator.parameters])
        return f"""LDLRule-{self.name}:
    Parameters: {self.parameters}
    Pos State Pre: {sorted(self.pos_state_preconditions, key=str)}
    Neg State Pre: {sorted(self.neg_state_preconditions, key=str)}
    Goal Pre: {sorted(self.goal_preconditions, key=str)}
    Operator: {self.operator.name}({op_param_str})"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class _GroundLDLRule:
    """A ground LDL rule is an LDLRule + objects.

    Should not be instantiated externally.
    """
    parent: LDLRule
    objects: Sequence[Object]
    pos_state_preconditions: Set[GroundAtom]
    neg_state_preconditions: Set[GroundAtom]
    goal_preconditions: Set[GroundAtom]
    ground_operator: _GroundSTRIPSOperator

    @cached_property
    def _str(self) -> str:
        op_obj_str = ", ".join([str(o) for o in self.ground_operator.objects])
        return f"""GroundLDLRule-{self.name}:
    Parameters: {self.objects}
    Pos State Pre: {sorted(self.pos_state_preconditions, key=str)}
    Neg State Pre: {sorted(self.neg_state_preconditions, key=str)}
    Goal Pre: {sorted(self.goal_preconditions, key=str)}
    Operator: {self.ground_operator.name}({op_obj_str})"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this ground LRL rule."""
        return self.parent.name

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) > str(other)


@dataclass(frozen=True)
class LiftedDecisionList:
    """A goal-conditioned policy from abstract states to ground operators
    implemented with a lifted decision list.

    The logic described above is implemented in utils.query_ldl().
    """
    rules: Sequence[LDLRule]

    @cached_property
    def _hash(self) -> int:
        return hash(tuple(self.rules))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, LiftedDecisionList)
        if len(self.rules) != len(other.rules):
            return False
        return all(r1 == r2 for r1, r2 in zip(self.rules, other.rules))

    def __str__(self) -> str:
        rule_str = "\n".join(str(r) for r in self.rules)
        return f"LiftedDecisionList[\n{rule_str}\n]"


VarToObjSub = Dict[Variable, Object]
ObjToVarSub = Dict[Object, Variable]
