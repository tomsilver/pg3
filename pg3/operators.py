"""Policy search operators."""
from __future__ import annotations

import abc
from collections import defaultdict
import functools
from typing import Callable, FrozenSet, Iterator, List, Set

from pg3 import utils
from pg3.structs import LDLRule, LiftedAtom, LiftedDecisionList, Predicate, \
    STRIPSOperator, Variable, Trajectory, GroundAtom, _GroundSTRIPSOperator


class _PG3SearchOperator(abc.ABC):
    """Given an LDL policy, generate zero or more successor LDL policies."""

    def __init__(self,
                 predicates: Set[Predicate],
                 operators: Set[STRIPSOperator],
                 generate_plan_examples: Callable[[LiftedDecisionList], Iterator[Trajectory]],
                 allow_new_vars: bool = True) -> None:
        self._predicates = predicates
        self._operators = operators
        self._generate_plan_examples = generate_plan_examples
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

    def get_successors(
        self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:

        # Generate plan examples using this LDL (or using demonstrations).
        examples = []
        all_seen_atoms = set()
        for action_seq, atom_seq, task in self._generate_plan_examples(ldl):
            assert len(atom_seq) == len(action_seq) + 1
            for t, (atoms, action) in enumerate(zip(atom_seq, action_seq)):
                # Store step so that we can later go back to front.
                examples.append((t, atoms, action, task))
                all_seen_atoms.update(atoms)

        # Collect all uncovered transitions, organized by (lifted) action.
        op_to_uncovered = defaultdict(list)
        max_uncovered_t = -1
        op_with_max_uncovered_t = None
        for example in examples:
            t, atoms, action, task = example
            ldl_action = utils.query_ldl(ldl, atoms, task.objects, task.goal)
            if ldl_action is None or ldl_action != action:
                op_to_uncovered[action.parent].append(example)
                if t > max_uncovered_t:
                    max_uncovered_t = t
                    op_with_max_uncovered_t = action.parent

        # No uncovered transitions found, policy is perfect.
        if max_uncovered_t == -1:
            return
        assert op_with_max_uncovered_t is not None

        # Select an action to create a rule for, back to front.
        uncovered_op_examples = op_to_uncovered[op_with_max_uncovered_t]
        print(f"Found {len(uncovered_op_examples)} uncovered examples for {op_with_max_uncovered_t.name}")

        # Perform a lifted intersection of positive, negative, and goal
        # preconditions to create a new rule.
        operator = op_with_max_uncovered_t
        parameters = operator.parameters
        pos_state_preconditions = None
        neg_state_preconditions = None
        goal_preconditions = None

        for _, atoms, action, task in uncovered_op_examples:
            # For now, just lift using the action parameters.
            sub = {o: v for o, v in zip(action.objects, parameters)}
            # Lift positives.
            lifted_pos = {a.lift(sub) for a in atoms if all(o in sub for o in a.objects)}
            # Lift negatives.
            univ = utils.get_all_ground_atoms(self._predicates, task.objects)
            absent_atoms = univ - atoms
            # Only consider negatives that were true at some point.
            absent_atoms &= all_seen_atoms
            lifted_neg = {a.lift(sub) for a in absent_atoms if all(o in sub for o in a.objects)}
            # Lift goal.
            lifted_goal = {a.lift(sub) for a in task.goal if all(o in sub for o in a.objects)}
            # Intersect.
            if pos_state_preconditions is None:
                pos_state_preconditions = lifted_pos
                neg_state_preconditions = lifted_neg
                goal_preconditions = lifted_goal
            else:
                pos_state_preconditions &= lifted_pos
                neg_state_preconditions &= lifted_neg
                goal_preconditions &= lifted_goal

        # Always include the preconditions of the operator.
        pos_state_preconditions |= operator.preconditions

        # Put the new rule at end of the LDL to ensure that previous rules
        # still cover whatever they previously covered.
        new_rule = LDLRule("BottomUpGenerated",
            parameters, pos_state_preconditions, neg_state_preconditions, goal_preconditions, operator)

        print("Proposing new rule:")
        print(new_rule)

        new_rules = list(ldl.rules) + [new_rule]
        yield LiftedDecisionList(new_rules)
