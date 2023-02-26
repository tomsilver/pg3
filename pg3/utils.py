"""Utility functions."""
from __future__ import annotations

import functools
import itertools
import logging
import re
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple

from pyperplan.heuristics.heuristic_base import \
    Heuristic as _PyperplanBaseHeuristic
from pyperplan.pddl.parser import TraversePDDLDomain, TraversePDDLProblem, \
    parse_domain_def, parse_lisp_iterator, parse_problem_def
from pyperplan.pddl.pddl import Domain as PyperplanDomain
from pyperplan.pddl.pddl import Type as PyperplanType
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS

from pg3.structs import GroundAtom, LDLRule, LiftedAtom, LiftedDecisionList, \
    Object, ObjectOrVariable, Predicate, STRIPSOperator, Task, Type, \
    Variable, _GroundLDLRule, _GroundSTRIPSOperator


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


def get_all_valid_actions(
    rule: LDLRule,
    atoms: Set[GroundAtom],
    objects: Set[Object],
    goal: Set[GroundAtom],
    static_predicates: Optional[Set[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> Iterator[_GroundSTRIPSOperator]:
    """Get all valid actions for an LRL rule in the given state."""
    for ground_rule in all_ground_ldl_rules(
            rule,
            objects,
            static_predicates=static_predicates,
            init_atoms=init_atoms):
        if ground_rule.pos_state_preconditions.issubset(atoms) and \
            not ground_rule.neg_state_preconditions & atoms and \
            ground_rule.goal_preconditions.issubset(goal):
            yield ground_rule.ground_operator


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
        for ground_op in get_all_valid_actions(
                rule,
                atoms,
                objects,
                goal,
                static_predicates=static_predicates,
                init_atoms=init_atoms):
            return ground_op
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
    """Get all combinations of variables satisfying the given types
    sequence."""
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


def get_all_ground_atoms(predicates: Collection[Predicate], objects: Collection[Object]) -> Set[GroundAtom]:
    """Get all groundings of all the predicates given objects."""
    return _cached_get_all_ground_atoms(frozenset(predicates), frozenset(objects))


@functools.lru_cache(maxsize=None)
def _cached_get_all_ground_atoms(predicates: FrozenSet[Predicate], objects: FrozenSet[Object]) -> Set[GroundAtom]:
    all_atoms = set()
    for predicate in predicates:
        all_atoms.update(
            get_all_ground_atoms_for_predicate(predicate, frozenset(objects)))
    return all_atoms


def get_all_ground_atoms_for_predicate(
        predicate: Predicate, objects: FrozenSet[Object]) -> Set[GroundAtom]:
    """Get all groundings of the predicate given objects."""
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
    all_atoms = get_all_ground_atoms(predicates, objects)
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


def _domain_str_to_pyperplan_domain(domain_str: str) -> PyperplanDomain:
    domain_ast = parse_domain_def(parse_lisp_iterator(domain_str.split("\n")))
    visitor = TraversePDDLDomain()
    domain_ast.accept(visitor)
    domain = visitor.domain
    # Fix edge case where "object" is treated specially by pyperplan.
    for type_name, typ in domain.types.items():
        if type_name == "object":
            typ.parent = None
    return domain


def parse_pddl_domain(
    pddl_domain_str: str
) -> Tuple[Set[Type], Set[Predicate], Set[STRIPSOperator]]:
    """Parse a PDDL domain from a string."""
    # Let pyperplan do most of the heavy lifting.
    pyperplan_domain = _domain_str_to_pyperplan_domain(pddl_domain_str)
    pyperplan_types = pyperplan_domain.types
    pyperplan_predicates = pyperplan_domain.predicates
    pyperplan_operators = pyperplan_domain.actions
    # Convert the pyperplan domain into our structs.
    # Process the type hierarchy. Sort the types such that if X inherits from Y
    # then X is after Y in the list (topological sort).
    type_graph = {
        t: {t.parent}
        for t in pyperplan_types.values() if t.parent is not None
    }
    sorted_types = list(TopologicalSorter(type_graph).static_order())
    pyperplan_type_to_type: Dict[PyperplanType, Type] = {}
    for pyper_type in sorted_types:
        if pyper_type.parent is None:
            assert pyper_type.name == "object"
            parent = None
        else:
            parent = pyperplan_type_to_type[pyper_type.parent]
        new_type = Type(pyper_type.name, parent)
        pyperplan_type_to_type[pyper_type] = new_type
    # Handle case where the domain is untyped.
    # Pyperplan uses the object type by default.
    if not pyperplan_type_to_type:  # pragma: no cover
        pyper_type = next(iter(pyperplan_types.values()))
        new_type = Type(pyper_type.name, parent=None)
        pyperplan_type_to_type[pyper_type] = new_type
    # Convert the predicates.
    predicate_name_to_predicate = {}
    for pyper_pred in pyperplan_predicates.values():
        name = pyper_pred.name
        pred_types = [
            pyperplan_type_to_type[t] for _, (t, ) in pyper_pred.signature
        ]
        pred = Predicate(name, pred_types)
        predicate_name_to_predicate[name] = pred
    # Convert the operators.
    operators = set()
    for pyper_op in pyperplan_operators.values():
        name = pyper_op.name
        parameters = [
            Variable(n, pyperplan_type_to_type[t])
            for n, (t, ) in pyper_op.signature
        ]
        param_name_to_param = {p.name: p for p in parameters}
        preconditions = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [param_name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.precondition
        }
        add_effects = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [param_name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.effect.addlist
        }
        delete_effects = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [param_name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.effect.dellist
        }
        strips_op = STRIPSOperator(name, parameters, preconditions,
                                   add_effects, delete_effects)
        operators.add(strips_op)
    # Collect the final outputs.
    types = set(pyperplan_type_to_type.values())
    predicates = set(predicate_name_to_predicate.values())
    return types, predicates, operators


def pddl_problem_str_to_task(pddl_problem_str: str, pddl_domain_str: str,
                             types: Set[Type],
                             predicates: Set[Predicate]) -> Task:
    """Parse a PDDL problem from a string."""
    # Let pyperplan do most of the heavy lifting.
    # Pyperplan needs the domain to parse the problem. Note that this is
    # cached by lru_cache.
    pyperplan_domain = _domain_str_to_pyperplan_domain(pddl_domain_str)
    # Now that we have the domain, parse the problem.
    lisp_iterator = parse_lisp_iterator(pddl_problem_str.split("\n"))
    problem_ast = parse_problem_def(lisp_iterator)
    visitor = TraversePDDLProblem(pyperplan_domain)
    problem_ast.accept(visitor)
    pyperplan_problem = visitor.get_problem()
    # Create the objects.
    type_name_to_type = {t.name: t for t in types}
    object_name_to_obj = {
        o: Object(o, type_name_to_type[t.name])
        for o, t in pyperplan_problem.objects.items()
    }
    objects = set(object_name_to_obj.values())
    # Create the initial state.
    predicate_name_to_predicate = {p.name: p for p in predicates}
    init = {
        GroundAtom(predicate_name_to_predicate[a.name],
                   [object_name_to_obj[n] for n, _ in a.signature])
        for a in pyperplan_problem.initial_state
    }
    # Create the goal.
    goal = {
        GroundAtom(predicate_name_to_predicate[a.name],
                   [object_name_to_obj[n] for n, _ in a.signature])
        for a in pyperplan_problem.goal
    }
    # Finalize the task.
    task = Task(objects, init, goal)
    return task


def find_balanced_expression(s: str, index: int) -> str:
    """Find balanced expression in string starting from given index."""
    assert s[index] == "("
    start_index = index
    balance = 1
    while balance != 0:
        index += 1
        symbol = s[index]
        if symbol == "(":
            balance += 1
        elif symbol == ")":
            balance -= 1
    return s[start_index:index + 1]


def find_all_balanced_expressions(s: str) -> List[str]:
    """Return a list of all balanced expressions in a string, starting from the
    beginning."""
    assert s[0] == "("
    assert s[-1] == ")"
    exprs = []
    index = 0
    start_index = index
    balance = 1
    while index < len(s) - 1:
        index += 1
        if balance == 0:
            exprs.append(s[start_index:index])
            # Jump to next "(".
            while True:
                if s[index] == "(":
                    break
                index += 1
            start_index = index
            balance = 1
            continue
        symbol = s[index]
        if symbol == "(":
            balance += 1
        elif symbol == ")":
            balance -= 1
    assert balance == 0
    exprs.append(s[start_index:index + 1])
    return exprs


def parse_ldl_from_str(
        ldl_str: str, types: Collection[Type],
        predicates: Collection[Predicate],
        operators: Collection[STRIPSOperator]) -> LiftedDecisionList:
    """Parse a lifted decision list from a string representation of it."""
    parser = _LDLParser(types, predicates, operators)
    return parser.parse(ldl_str)


class _LDLParser:
    """Parser for lifted decision lists from strings."""

    def __init__(self, types: Collection[Type],
                 predicates: Collection[Predicate],
                 operators: Collection[STRIPSOperator]) -> None:
        self._op_name_to_op = {
            operator.name: operator
            for operator in operators
        }
        self._type_name_to_type = {t.name: t for t in types}
        self._pred_name_to_pred = {p.name: p for p in predicates}

    def parse(self, ldl_str: str) -> LiftedDecisionList:
        """Run parsing."""
        rules = []
        rule_matches = re.finditer(r"\(:rule", ldl_str)
        for start in rule_matches:
            rule_str = find_balanced_expression(ldl_str, start.start())
            rule = self._parse_rule(rule_str)
            rules.append(rule)
        return LiftedDecisionList(rules)

    def _parse_rule(self, rule_str: str) -> LDLRule:
        rule_pattern = r"\(:rule(.*):parameters(.*):preconditions(.*)" + \
                       r":goals(.*):action(.*)\)"
        match_result = re.match(rule_pattern, rule_str, re.DOTALL)
        assert match_result is not None
        # Remove white spaces.
        matches = [m.strip().rstrip() for m in match_result.groups()]
        # Unpack the matches.
        rule_name, params_str, preconds_str, goals_str, operator_str = matches
        # Handle the parameters.
        assert "?" in params_str, "Assuming all rules have parameters."
        variable_name_to_variable = {}
        assert params_str.endswith(")")
        for param_str in params_str[:-1].split("?")[1:]:
            param_name, param_type_str = param_str.split("-")
            param_name = param_name.strip()
            param_type_str = param_type_str.strip()
            variable_name = "?" + param_name
            param_type = self._type_name_to_type[param_type_str]
            variable = Variable(variable_name, param_type)
            variable_name_to_variable[variable_name] = variable
        # Handle the preconditions.
        pos_preconds, neg_preconds = self._parse_lifted_atoms(
            preconds_str, variable_name_to_variable)
        # Handle the goals.
        pos_goals, neg_goals = self._parse_lifted_atoms(
            goals_str, variable_name_to_variable)
        assert not neg_goals, "Negative LDL goals not currently supported"
        # Handle the STRIPSOperator.
        operator = self._parse_into_operator(operator_str,
                                             variable_name_to_variable)
        # Finalize the rule.
        params = sorted(variable_name_to_variable.values())
        return LDLRule(rule_name, params, pos_preconds, neg_preconds,
                       pos_goals, operator)

    def _parse_lifted_atoms(
        self, atoms_str: str, variable_name_to_variable: Dict[str, Variable]
    ) -> Tuple[Set[LiftedAtom], Set[LiftedAtom]]:
        """Parse the given string (representing either preconditions or
        effects) into a set of positive lifted atoms and a set of negative
        lifted atoms.

        Check against params to make sure typing is correct.
        """
        assert atoms_str[0] == "("
        assert atoms_str[-1] == ")"

        # Handle conjunctions.
        if atoms_str.startswith("(and") and atoms_str[4] in (" ", "\n", "("):
            clauses = find_all_balanced_expressions(atoms_str[4:-1].strip())
            pos_atoms, neg_atoms = set(), set()
            for clause in clauses:
                clause_pos_atoms, clause_neg_atoms = self._parse_lifted_atoms(
                    clause, variable_name_to_variable)
                pos_atoms |= clause_pos_atoms
                neg_atoms |= clause_neg_atoms
            return pos_atoms, neg_atoms

        # Handle negations.
        if atoms_str.startswith("(not") and atoms_str[4] in (" ", "\n", "("):
            # Only contains a single literal inside not.
            split_strs = atoms_str[4:-1].strip()[1:-1].strip().split()
            pred = self._pred_name_to_pred[split_strs[0]]
            args = [variable_name_to_variable[arg] for arg in split_strs[1:]]
            lifted_atom = LiftedAtom(pred, args)
            return set(), {lifted_atom}

        # Handle single positive atoms.
        split_strs = atoms_str[1:-1].split()
        # Empty conjunction.
        if not split_strs:
            return set(), set()
        pred = self._pred_name_to_pred[split_strs[0]]
        args = [variable_name_to_variable[arg] for arg in split_strs[1:]]
        lifted_atom = LiftedAtom(pred, args)
        return {lifted_atom}, set()

    def _parse_into_operator(
            self, operator_str: str,
            variable_name_to_variable: Dict[str, Variable]) -> STRIPSOperator:
        """Parse the given string into an STRIPSOperator."""
        assert operator_str[0] == "("
        assert operator_str[-1] == ")"
        operator_str = operator_str[1:-1].split()[0]
        operator = self._op_name_to_op[operator_str]
        # Validate parameters.
        variables = variable_name_to_variable.values()
        for v in operator.parameters:
            assert v in variables, \
                "STRIPSOperator parameter {v} missing from LDL rule"
        return operator


def policy_satisfied(policy_str: str, problem_str: str, domain_str: str,
                     action: str) -> bool:
    """Check if an action satisfies this policy in the initial state."""
    types, predicates, operators = parse_pddl_domain(domain_str)
    policy = parse_ldl_from_str(policy_str, types, predicates, operators)
    task = pddl_problem_str_to_task(problem_str, domain_str, types, predicates)
    for rule in policy.rules:
        all_valid_actions = {
            o.pddl_str
            for o in get_all_valid_actions(rule, task.init, task.objects,
                                           task.goal)
        }
        if action in all_valid_actions:
            return True
        # The rule returns some action(s), but not the given one.
        if all_valid_actions:
            return False
    return False
