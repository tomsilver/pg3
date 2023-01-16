"""Utility functions."""
from __future__ import annotations

import functools
import itertools
import logging
from dataclasses import dataclass
from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set

from pyperplan.heuristics.heuristic_base import \
    Heuristic as _PyperplanBaseHeuristic
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS

from pg3.structs import GroundAtom, LDLRule, LiftedAtom, LiftedDecisionList, \
    Object, ObjectOrVariable, Predicate, STRIPSOperator, Type, Variable, \
    _GroundLDLRule, _GroundSTRIPSOperator


def all_ground_ldl_rules(
    rule: LDLRule,
    objects: Collection[Object],
    static_predicates: Optional[Collection[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> List[_GroundLDLRule]:
    """Get all possible groundings of the given rule with the given objects.

    If provided, use the static predicates and init_atoms to avoid
    grounding rules that will never have satisfied preconditions in any
    state.
    """
    if static_predicates is None:
        static_predicates = set()
    if init_atoms is None:
        init_atoms = set()
    return _cached_all_ground_ldl_rules(rule, frozenset(objects),
                                        frozenset(static_predicates),
                                        frozenset(init_atoms))


@functools.lru_cache(maxsize=None)
def _cached_all_ground_ldl_rules(
        rule: LDLRule, objects: FrozenSet[Object],
        static_predicates: FrozenSet[Predicate],
        init_atoms: FrozenSet[GroundAtom]) -> List[_GroundLDLRule]:
    """Helper for all_ground_ldl_rules() that caches the outputs."""
    ground_rules = []
    # Use static preconds to reduce the map of parameters to possible objects.
    # For example, if IsBall(?x) is a positive state precondition, then only
    # the objects that appear in init_atoms with IsBall could bind to ?x.
    # For now, we just check unary static predicates, since that covers the
    # common case where such predicates are used in place of types.
    # Create map from each param to unary static predicates.
    param_to_pos_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in rule.parameters
    }
    param_to_neg_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in rule.parameters
    }
    for (preconditions, param_to_preds) in [
        (rule.pos_state_preconditions, param_to_pos_preds),
        (rule.neg_state_preconditions, param_to_neg_preds),
    ]:
        for atom in preconditions:
            pred = atom.predicate
            if pred in static_predicates and pred.arity == 1:
                param = atom.variables[0]
                param_to_preds[param].add(pred)
    # Create the param choices, filtering based on the unary static atoms.
    param_choices = []  # list of lists of possible objects for each param
    # Preprocess the atom sets for faster lookups.
    init_atom_tups = {(a.predicate, tuple(a.objects)) for a in init_atoms}
    for param in rule.parameters:
        choices = []
        for obj in objects:
            # Types must match, as usual.
            if obj.type != param.type:
                continue
            # Check the static conditions.
            binding_valid = True
            for pred in param_to_pos_preds[param]:
                if (pred, (obj, )) not in init_atom_tups:
                    binding_valid = False
                    break
            for pred in param_to_neg_preds[param]:
                if (pred, (obj, )) in init_atom_tups:
                    binding_valid = False
                    break
            if binding_valid:
                choices.append(obj)
        # Must be sorted for consistency with other grounding code.
        param_choices.append(sorted(choices))
    for choice in itertools.product(*param_choices):
        ground_rule = rule.ground(choice)
        ground_rules.append(ground_rule)
    return ground_rules


def query_ldl(
    ldl: LiftedDecisionList,
    atoms: Set[GroundAtom],
    objects: Set[Object],
    goal: Set[GroundAtom],
    static_predicates: Optional[Set[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> Optional[_GroundSTRIPSOperator]:
    """Queries a lifted decision list representing a goal-conditioned policy.

    Given an abstract state and goal, the rules are grounded in order.
    The first applicable ground rule is used to return a ground
    operator. If static_predicates is provided, it is used to avoid
    grounding rules with nonsense preconditions like IsBall(robot). If
    no rule is applicable, returns None.
    """
    for rule in ldl.rules:
        for ground_rule in all_ground_ldl_rules(
                rule,
                objects,
                static_predicates=static_predicates,
                init_atoms=init_atoms):
            if ground_rule.pos_state_preconditions.issubset(atoms) and \
               not ground_rule.neg_state_preconditions & atoms and \
               ground_rule.goal_preconditions.issubset(goal):
                return ground_rule.ground_operator
    return None


def _get_entity_combinations(
        entities: Collection[ObjectOrVariable],
        types: Sequence[Type]) -> Iterator[List[ObjectOrVariable]]:
    """Get all combinations of entities satisfying the given types sequence."""
    sorted_entities = sorted(entities)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_object_combinations(objects: Collection[Object],
                            types: Sequence[Type]) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(objects, types)


def get_variable_combinations(
        variables: Collection[Variable],
        types: Sequence[Type]) -> Iterator[List[Variable]]:
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(variables, types)


def get_all_lifted_atoms_for_predicate(
        predicate: Predicate,
        variables: FrozenSet[Variable]) -> Set[LiftedAtom]:
    """Get all groundings of the predicate given variables.

    Note: we don't want lru_cache() on this function because we might want
    to call it with stripped predicates, and we wouldn't want it to return
    cached values.
    """
    lifted_atoms = set()
    for args in get_variable_combinations(variables, predicate.types):
        lifted_atom = LiftedAtom(predicate, args)
        lifted_atoms.add(lifted_atom)
    return lifted_atoms


def create_new_variables(
    types: Sequence[Type],
    existing_vars: Optional[Collection[Variable]] = None,
    var_prefix: str = "?x",
) -> List[Variable]:
    """Create new variables of the given types, avoiding name collisions with
    existing variables. By convention, all new variables are of the form.

    <var_prefix><number>.
    """
    pre_len = len(var_prefix)
    existing_var_nums = set()
    if existing_vars:
        for v in existing_vars:
            if v.name.startswith(var_prefix) and v.name[pre_len:].isdigit():
                existing_var_nums.add(int(v.name[pre_len:]))
    if existing_var_nums:
        counter = itertools.count(max(existing_var_nums) + 1)
    else:
        counter = itertools.count(0)
    new_vars = []
    for t in types:
        new_var_name = f"{var_prefix}{next(counter)}"
        new_var = Variable(new_var_name, t)
        new_vars.append(new_var)
    return new_vars


def apply_operator(op: _GroundSTRIPSOperator,
                   atoms: Set[GroundAtom]) -> Set[GroundAtom]:
    """Get a next set of atoms given a current set and a ground operator."""
    new_atoms = set(atoms)
    for atom in op.delete_effects:
        new_atoms.discard(atom)
    for atom in op.add_effects:
        new_atoms.add(atom)
    return new_atoms


def get_applicable_operators(
        ground_ops: Collection[_GroundSTRIPSOperator],
        atoms: Collection[GroundAtom]) -> Iterator[_GroundSTRIPSOperator]:
    """Iterate over ground operators whose preconditions are satisfied.

    Note: the order may be nondeterministic. Users should be invariant.
    """
    for op in ground_ops:
        applicable = op.preconditions.issubset(atoms)
        if applicable:
            yield op


def all_ground_operators(
        operator: STRIPSOperator,
        objects: Collection[Object]) -> Iterator[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given
    objects."""
    types = [p.type for p in operator.parameters]
    for choice in get_object_combinations(objects, types):
        yield operator.ground(tuple(choice))


def get_static_atoms(ground_ops: Collection[_GroundSTRIPSOperator],
                     atoms: Collection[GroundAtom]) -> Set[GroundAtom]:
    """Get the subset of atoms from the given set that are static with respect
    to the given ground operators.

    Note that this can include MORE than simply the set of atoms whose
    predicates are static, because now we have ground operators.
    """
    static_atoms = set()
    for atom in atoms:
        # This atom is not static if it appears in any op's effects.
        if any(
                any(atom == eff for eff in op.add_effects) or any(
                    atom == eff for eff in op.delete_effects)
                for op in ground_ops):
            continue
        static_atoms.add(atom)
    return static_atoms


def get_all_ground_atoms_for_predicate(
        predicate: Predicate, objects: FrozenSet[Object]) -> Set[GroundAtom]:
    """Get all groundings of the predicate given objects.

    Note: we don't want lru_cache() on this function because we might want
    to call it with stripped predicates, and we wouldn't want it to return
    cached values.
    """
    ground_atoms = set()
    for args in get_object_combinations(objects, predicate.types):
        ground_atom = GroundAtom(predicate, args)
        ground_atoms.add(ground_atom)
    return ground_atoms


def create_task_planning_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[_GroundSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _TaskPlanningHeuristic:
    """Create a task planning heuristic that consumes ground atoms and
    estimates the cost-to-go."""
    if heuristic_name in _PYPERPLAN_HEURISTICS:
        return _create_pyperplan_heuristic(heuristic_name, init_atoms, goal,
                                           ground_ops, predicates, objects)
    raise ValueError(f"Unrecognized heuristic name: {heuristic_name}.")


@dataclass(frozen=True)
class _TaskPlanningHeuristic:
    """A task planning heuristic."""
    name: str
    init_atoms: Collection[GroundAtom]
    goal: Set[GroundAtom]
    ground_ops: Collection[_GroundSTRIPSOperator]

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        raise NotImplementedError("Override me!")


############################### Pyperplan Glue ###############################


def _create_pyperplan_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[_GroundSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _PyperplanHeuristicWrapper:
    """Create a pyperplan heuristic that inherits from
    _TaskPlanningHeuristic."""
    assert heuristic_name in _PYPERPLAN_HEURISTICS
    static_atoms = get_static_atoms(ground_ops, init_atoms)
    pyperplan_heuristic_cls = _PYPERPLAN_HEURISTICS[heuristic_name]
    pyperplan_task = _create_pyperplan_task(init_atoms, goal, ground_ops,
                                            predicates, objects, static_atoms)
    pyperplan_heuristic = pyperplan_heuristic_cls(pyperplan_task)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    return _PyperplanHeuristicWrapper(heuristic_name, init_atoms, goal,
                                      ground_ops, static_atoms,
                                      pyperplan_heuristic, pyperplan_goal)


_PyperplanFacts = FrozenSet[str]


@dataclass(frozen=True)
class _PyperplanNode:
    """Container glue for pyperplan heuristics."""
    state: _PyperplanFacts
    goal: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanOperator:
    """Container glue for pyperplan heuristics."""
    name: str
    preconditions: _PyperplanFacts
    add_effects: _PyperplanFacts
    del_effects: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanTask:
    """Container glue for pyperplan heuristics."""
    facts: _PyperplanFacts
    initial_state: _PyperplanFacts
    goals: _PyperplanFacts
    operators: Collection[_PyperplanOperator]


@dataclass(frozen=True)
class _PyperplanHeuristicWrapper(_TaskPlanningHeuristic):
    """A light wrapper around pyperplan's heuristics."""
    _static_atoms: Set[GroundAtom]
    _pyperplan_heuristic: _PyperplanBaseHeuristic
    _pyperplan_goal: _PyperplanFacts

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        # Note: filtering out static atoms.
        pyperplan_facts = _atoms_to_pyperplan_facts(set(atoms) \
                                                    - self._static_atoms)
        return self._evaluate(pyperplan_facts, self._pyperplan_goal,
                              self._pyperplan_heuristic)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _evaluate(pyperplan_facts: _PyperplanFacts,
                  pyperplan_goal: _PyperplanFacts,
                  pyperplan_heuristic: _PyperplanBaseHeuristic) -> float:
        pyperplan_node = _PyperplanNode(pyperplan_facts, pyperplan_goal)
        logging.disable(logging.DEBUG)
        result = pyperplan_heuristic(pyperplan_node)
        logging.disable(logging.NOTSET)
        return result


def _create_pyperplan_task(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[_GroundSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
    static_atoms: Set[GroundAtom],
) -> _PyperplanTask:
    """Helper glue for pyperplan heuristics."""
    all_atoms = set()
    for predicate in predicates:
        all_atoms.update(
            get_all_ground_atoms_for_predicate(predicate, frozenset(objects)))
    # Note: removing static atoms.
    pyperplan_facts = _atoms_to_pyperplan_facts(all_atoms - static_atoms)
    pyperplan_state = _atoms_to_pyperplan_facts(init_atoms - static_atoms)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    pyperplan_operators = set()
    for op in ground_ops:
        # Note: the pyperplan operator must include the objects, because hFF
        # uses the operator name in constructing the relaxed plan, and the
        # relaxed plan is a set. If we instead just used op.name, there would
        # be a very nasty bug where two ground operators in the relaxed plan
        # that have different objects are counted as just one.
        name = op.name + "-".join(o.name for o in op.objects)
        pyperplan_operator = _PyperplanOperator(
            name,
            # Note: removing static atoms from preconditions.
            _atoms_to_pyperplan_facts(op.preconditions - static_atoms),
            _atoms_to_pyperplan_facts(op.add_effects),
            _atoms_to_pyperplan_facts(op.delete_effects))
        pyperplan_operators.add(pyperplan_operator)
    return _PyperplanTask(pyperplan_facts, pyperplan_state, pyperplan_goal,
                          pyperplan_operators)


@functools.lru_cache(maxsize=None)
def _atom_to_pyperplan_fact(atom: GroundAtom) -> str:
    """Convert atom to tuple for interface with pyperplan."""
    arg_str = " ".join(o.name for o in atom.objects)
    return f"({atom.predicate.name} {arg_str})"


def _atoms_to_pyperplan_facts(
        atoms: Collection[GroundAtom]) -> _PyperplanFacts:
    """Light wrapper around _atom_to_pyperplan_fact() that operates on a
    collection of atoms."""
    return frozenset({_atom_to_pyperplan_fact(atom) for atom in atoms})


############################## End Pyperplan Glue ##############################
