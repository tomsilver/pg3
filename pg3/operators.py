"""Policy search operators."""
from __future__ import annotations

import abc
import functools
from collections import defaultdict
from typing import ClassVar, FrozenSet, Iterator, List, Optional, Sequence, Set

from pg3 import utils
from pg3.structs import GroundMacro, LDLRule, LiftedAtom, LiftedDecisionList, \
    Predicate, STRIPSOperator, Task, Variable
from pg3.trajectory_gen import _TrajectoryGenerator


class _PG3SearchOperator(abc.ABC):
    """Given an LDL policy, generate zero or more successor LDL policies."""

    def __init__(self,
                 predicates: Set[Predicate],
                 operators: Set[STRIPSOperator],
                 trajectory_gen: _TrajectoryGenerator,
                 train_tasks: Sequence[Task],
                 allow_new_vars: bool = True) -> None:
        self._predicates = predicates
        self._operators = operators
        self._trajectory_gen = trajectory_gen
        self._train_tasks = train_tasks
        self._allow_new_vars = allow_new_vars

    @abc.abstractmethod
    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        """Generate zero or more successor LDL policies."""
        raise NotImplementedError("Override me!")


class _AddRulePG3SearchOperator(_PG3SearchOperator):
    """An operator that adds new rules to an existing LDL policy."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for idx in range(len(ldl.rules) + 1):
            for rule in self._get_candidate_rules():
                new_rules = list(ldl.rules)
                new_rules.insert(idx, rule)
                yield LiftedDecisionList(new_rules)

    @functools.lru_cache(maxsize=None)
    def _get_candidate_rules(self) -> List[LDLRule]:
        return [self._operator_to_rule(op) for op in sorted(self._operators)]

    @staticmethod
    def _operator_to_rule(operator: STRIPSOperator) -> LDLRule:
        """Initialize an LDLRule from an operator."""
        return LDLRule(
            name=operator.name,
            parameters=list(operator.parameters),
            pos_state_preconditions=set(operator.preconditions),
            neg_state_preconditions=set(),
            goal_preconditions=set(),
            operator=operator,
        )


class _AddConditionPG3SearchOperator(_PG3SearchOperator):
    """An operator that adds new preconditions to existing LDL rules."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx, rule in enumerate(ldl.rules):
            rule_vars = frozenset(rule.parameters)
            for condition in self._get_candidate_conditions(rule_vars):
                # Consider adding new condition to positive preconditions,
                # negative preconditions, or goal preconditions.
                for destination in ["pos", "neg", "goal"]:
                    new_pos = set(rule.pos_state_preconditions)
                    new_neg = set(rule.neg_state_preconditions)
                    new_goal = set(rule.goal_preconditions)
                    if destination == "pos":
                        dest_set = new_pos
                    elif destination == "neg":
                        dest_set = new_neg
                    else:
                        assert destination == "goal"
                        dest_set = new_goal
                    # If the condition already exists, skip.
                    if condition in dest_set:
                        continue
                    # Special case: if the condition already exists in the
                    # positive preconditions, don't add to the negative
                    # preconditions, and vice versa.
                    if destination in ("pos", "neg") and condition in \
                        new_pos | new_neg:
                        continue
                    dest_set.add(condition)
                    parameters = sorted({
                        v
                        for c in new_pos | new_neg | new_goal
                        for v in c.variables
                    } | set(rule.operator.parameters))
                    # Create the new rule.
                    new_rule = LDLRule(
                        name=rule.name,
                        parameters=parameters,
                        pos_state_preconditions=new_pos,
                        neg_state_preconditions=new_neg,
                        goal_preconditions=new_goal,
                        operator=rule.operator,
                    )
                    # Create the new LDL.
                    new_rules = list(ldl.rules)
                    new_rules[rule_idx] = new_rule
                    yield LiftedDecisionList(new_rules)

    @functools.lru_cache(maxsize=None)
    def _get_candidate_conditions(
            self, variables: FrozenSet[Variable]) -> List[LiftedAtom]:
        conditions = []
        for pred in sorted(self._predicates):
            # Allow or disallow creation of fresh variables.
            condition_vars = variables
            if self._allow_new_vars:
                new_vars = utils.create_new_variables(pred.types, variables)
                condition_vars |= frozenset(new_vars)
            for condition in utils.get_all_lifted_atoms_for_predicate(
                    pred, condition_vars):
                conditions.append(condition)
        return conditions


class _DeleteConditionPG3SearchOperator(_PG3SearchOperator):
    """An operator that removes conditions from existing LDL rules."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx, rule in enumerate(ldl.rules):
            for condition in sorted(rule.pos_state_preconditions | \
                rule.neg_state_preconditions | rule.goal_preconditions):

                # If the condition to be removed is a
                # precondition of an operator, don't remove it.
                if condition in rule.operator.preconditions:
                    continue

                # Recreate new preconditions.
                # Assumes that a condition can appear only in one set
                new_pos = rule.pos_state_preconditions - {condition}
                new_neg = rule.neg_state_preconditions - {condition}
                new_goal = rule.goal_preconditions - {condition}

                # Reconstruct parameters from the other
                # components of the LDL.
                all_atoms = new_pos | new_neg | new_goal
                new_rule_params_set = \
                    {v for a in all_atoms for v in a.variables}
                new_rule_params_set.update(rule.operator.parameters)
                new_rule_params = sorted(new_rule_params_set)

                # Create the new rule.
                new_rule = LDLRule(
                    name=rule.name,
                    parameters=new_rule_params,
                    pos_state_preconditions=new_pos,
                    neg_state_preconditions=new_neg,
                    goal_preconditions=new_goal,
                    operator=rule.operator,
                )
                # Create the new LDL.
                new_rules = list(ldl.rules)
                new_rules[rule_idx] = new_rule
                yield LiftedDecisionList(new_rules)


class _DeleteRulePG3SearchOperator(_PG3SearchOperator):
    """An operator that removes entire rules from existing LDL rules."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx in range(len(ldl.rules)):
            new_rules = [r for i, r in enumerate(ldl.rules) if i != rule_idx]
            yield LiftedDecisionList(new_rules)


class _BottomUpPG3SearchOperator(_PG3SearchOperator):
    """An operator that uses plans to suggest a single new LDL."""

    _max_macro_length: ClassVar[int] = 1

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:

        # Generate macro examples using this LDL (or using demonstrations).
        examples = []  # (priority, atom set, ground macro, task)
        all_seen_atoms = set()  # used to determine valid negative atoms
        for task in self._train_tasks:
            action_seq, atom_seq, _ = \
            self._trajectory_gen.get_trajectory_for_task(task, ldl)
            assert len(atom_seq) == len(action_seq) + 1
            all_seen_atoms.update({a for atoms in atom_seq for a in atoms})
            for l in range(self._max_macro_length + 1):
                for t in range(len(action_seq) - l):
                    # Priority: higher is better.
                    priority = (t, l)
                    atoms = atom_seq[t]
                    ground_macro = GroundMacro(action_seq[t:(t + l + 1)])
                    examples.append((priority, atoms, ground_macro, task))

        # Collect all uncovered transitions, organized by (lifted) macro.
        macro_to_uncovered = defaultdict(list)
        best_priority = (-float("inf"), -float("inf"))
        selected_macro = None
        for example in examples:
            priority, atoms, ground_macro, task = example
            plan_action = ground_macro.ground_operators[0]
            ldl_action = utils.query_ldl(ldl, atoms, task.objects, task.goal)
            if ldl_action is None or ldl_action != plan_action:
                macro_to_uncovered[ground_macro.parent].append(example)
                if priority > best_priority:
                    best_priority = priority
                    selected_macro = ground_macro.parent

        # No uncovered transitions found, policy is perfect.
        if selected_macro is None:
            return

        # Select a macro to create a rule for, back to front.
        uncovered_macro_examples = macro_to_uncovered[selected_macro]

        # Perform a lifted intersection of positive, negative, and goal
        # preconditions to create a new rule.
        operator = selected_macro.operators[0]
        parameters = selected_macro.parameters
        pos_state_preconditions: Optional[Set[LiftedAtom]] = None
        neg_state_preconditions: Optional[Set[LiftedAtom]] = None
        goal_preconditions: Optional[Set[LiftedAtom]] = None

        for _, atoms, ground_macro, task in uncovered_macro_examples:
            # Create a substitution from objects to parameters.
            sub = ground_macro.get_lift_mapping()
            # Lift positives.
            lifted_pos = {
                a.lift(sub)
                for a in atoms if all(o in sub for o in a.objects)
            }
            # Lift negatives.
            univ = utils.get_all_ground_atoms(self._predicates, task.objects)
            absent_atoms = univ - atoms
            # Only consider negatives that were true at some point.
            absent_atoms &= all_seen_atoms
            lifted_neg = {
                a.lift(sub)
                for a in absent_atoms if all(o in sub for o in a.objects)
            }
            # Lift goal.
            lifted_goal = {
                a.lift(sub)
                for a in task.goal if all(o in sub for o in a.objects)
            }
            # Intersect.
            if pos_state_preconditions is None:
                pos_state_preconditions = lifted_pos
                neg_state_preconditions = lifted_neg
                goal_preconditions = lifted_goal
            else:
                assert neg_state_preconditions is not None
                assert goal_preconditions is not None
                pos_state_preconditions &= lifted_pos
                neg_state_preconditions &= lifted_neg
                goal_preconditions &= lifted_goal

        assert pos_state_preconditions is not None
        assert neg_state_preconditions is not None
        assert goal_preconditions is not None

        # Remap variables to operator parameters to deal with annoying
        # assumption that the operator and LDL parameters are aligned.
        var_sub = {v: k for k, v in selected_macro.parameter_subs[0].items()}
        for missing_param in parameters:
            if missing_param not in var_sub:
                var_sub[missing_param] = missing_param
        parameters = [var_sub[v] for v in parameters]
        pos_state_preconditions = {
            a.substitute(var_sub)
            for a in pos_state_preconditions
        }
        neg_state_preconditions = {
            a.substitute(var_sub)
            for a in neg_state_preconditions
        }
        goal_preconditions = {
            a.substitute(var_sub)
            for a in goal_preconditions
        }

        # Always include the preconditions of the operator.
        pos_state_preconditions |= operator.preconditions

        # Put the new rule at end of the LDL to ensure that previous rules
        # still cover whatever they previously covered.
        new_rule = LDLRule("BottomUpGenerated", parameters,
                           pos_state_preconditions, neg_state_preconditions,
                           goal_preconditions, operator)

        new_rules = list(ldl.rules) + [new_rule]
        yield LiftedDecisionList(new_rules)
