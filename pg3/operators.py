"""Policy search operators."""
from __future__ import annotations

import abc
from collections import defaultdict
import functools
from typing import Dict, FrozenSet, Iterator, List, Set, Sequence, Optional, Tuple
from typing_extensions import TypeAlias

from pg3 import utils
from pg3.structs import LDLRule, LiftedAtom, LiftedDecisionList, Predicate, \
    STRIPSOperator, Variable, Task, GroundAtom, _GroundSTRIPSOperator


class _PG3SearchOperator(abc.ABC):
    """Given an LDL policy, generate zero or more successor LDL policies."""

    def __init__(
        self,
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
        train_tasks: Sequence[Task],
        demos: Optional[List[List[str]]] = None,
    ) -> None:
        self._predicates = predicates
        self._operators = operators
        self._train_tasks = train_tasks
        self._user_supplied_demos = demos

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        """Generate zero or more successor LDL policies."""
        for succ in self._get_successors(ldl):
            # TODO
            if any(len(r.parameters) > 4 for r in succ.rules):
                continue
            yield succ

    @abc.abstractmethod
    def _get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        """Generate zero or more successor LDL policies."""
        raise NotImplementedError("Override me!")


class _AddRulePG3SearchOperator(_PG3SearchOperator):
    """An operator that adds new rules to an existing LDL policy."""

    def _get_successors(
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

    def _get_successors(
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
            new_vars = utils.create_new_variables(pred.types, variables)
            condition_vars = variables | frozenset(new_vars)
            # Uncomment to disallow creation of fresh variables.
            # condition_vars = variables
            for condition in utils.get_all_lifted_atoms_for_predicate(
                    pred, condition_vars):
                conditions.append(condition)
        return conditions


class _DeleteConditionPG3SearchOperator(_PG3SearchOperator):
    """An operator that removes conditions from existing LDL rules."""

    def _get_successors(
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

    def _get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx in range(len(ldl.rules)):
            new_rules = [r for i, r in enumerate(ldl.rules) if i != rule_idx]
            yield LiftedDecisionList(new_rules)



class _BottomUpOperatorPG3SearchOperator(_PG3SearchOperator):
    """Use plan examples to propose a change to a candidate policy.
    
    This is a much-simplified version of the operator from the original paper.

    Works as follows:
    1. Find all uncovered transitions, and among these, find the action with
       the most uncovered transitions, and return all those transitions.
    2. Lift all those transitions using the action arguments.
    3. Intersect the positive preconditions and goal.
    4. Put the new rule at end of the policy to ensure that previous rules
       still cover whatever they previously covered
    """
    def __init__(
        self,
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
        train_tasks: Sequence[Task],
        demos: Optional[List[List[str]]] = None,
    ) -> None:
        super().__init__(predicates, operators, train_tasks, demos)
        # Ground the STRIPSOperators once per task and save them.
        # TODO: refactor?
        self._train_task_idx_to_ground_operators = {
            idx: [
                ground_op for op in operators
                for ground_op in utils.all_ground_operators(op, task.objects)
            ]
            for idx, task in enumerate(self._train_tasks)
        }

    def _get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        act_to_uncovered_transitions = self._find_uncovered_transitions(ldl)
        act = max(act_to_uncovered_transitions, key=lambda a: len(act_to_uncovered_transitions[a]))
        # Lift
        lifted_transitions = {self._lift_transition(atoms, act, goal) for atoms, act, goal in act_to_uncovered_transitions[act]}
        if not lifted_transitions:
            raise StopIteration()
        print(f"Found {len(act_to_uncovered_transitions[act])} uncovered transitions")
        print(f"Found {len(lifted_transitions)} distinct lifted transitions")
        # Intersect to get preconditions and goal
        common_preconds = None
        common_goal = None
        for pre, goal in lifted_transitions:
            if common_preconds is None:
                common_preconds = pre
            else:
                common_preconds &= pre
            if common_goal is None:
                common_goal = goal
            else:
                common_goal &= goal
        assert common_preconds is not None
        assert common_goal is not None
        # Add action preconditions
        common_preconds |= act.preconditions
        # Create new rule
        new_rule = LDLRule(
            name=act.name,
            parameters=list(act.parameters),
            pos_state_preconditions=set(common_preconds),
            neg_state_preconditions=set(),
            goal_preconditions=set(common_goal),
            operator=act,
        )
        print("Creating rule:")
        print(new_rule)
        # Finish the new policy
        new_rules = list(ldl.rules)
        # new_rules.append(new_rule)
        new_rules = [new_rule] + new_rules
        yield LiftedDecisionList(new_rules)


    def _find_uncovered_transitions(self, ldl: LiftedDecisionList):
        act_to_uncovered_transitions = defaultdict(set)
        for task_idx in range(len(self._train_tasks)):
            atom_seq, action_seq = self._get_plan_for_task(ldl, task_idx)
            task = self._train_tasks[task_idx]
            objects = task.objects
            goal = task.goal
            for i in range(len(action_seq)):
                demo_atoms = atom_seq[i]
                demo_act = action_seq[i]
                ldl_act = utils.query_ldl(ldl, demo_atoms, objects, goal)
                if ldl_act is None:
                    act_to_uncovered_transitions[demo_act.parent].add(
                        (frozenset(demo_atoms), demo_act, frozenset(goal))
                    )
        return act_to_uncovered_transitions

    def _lift_transition(self, atoms: FrozenSet[GroundAtom], act: _GroundSTRIPSOperator, goal: FrozenSet[GroundAtom]
        ) -> Tuple[FrozenSet[LiftedAtom], FrozenSet[LiftedAtom]]:
        obj_to_var = dict(zip(act.objects, act.parent.parameters))
        lifted_atoms = set()
        for atom in atoms:
            if all(a in obj_to_var for a in atom.objects):
                lifted_atoms.add(atom.lift(obj_to_var))
        lifted_goal = set()
        for atom in goal:
            if all(a in obj_to_var for a in atom.objects):
                lifted_goal.add(atom.lift(obj_to_var))
        return frozenset(lifted_atoms), frozenset(lifted_goal)

    # TODO: refactor below here to reuse code in heuristics
    # TODO: allow LDL-guidance here
    def _get_plan_for_task(self, ldl: LiftedDecisionList,
                                task_idx: int) -> Sequence[Set[GroundAtom]]:
        del ldl  # unused
        return self._get_demo_plan_for_task(task_idx)

    @functools.lru_cache(maxsize=None)
    def _get_demo_plan_for_task(
            self, task_idx: int) -> Sequence[Set[GroundAtom]]:
        # Run planning once per task and cache the result.

        task = self._train_tasks[task_idx]
        ground_operators = self._train_task_idx_to_ground_operators[task_idx]

        assert self._user_supplied_demos is not None
        demo = self._user_supplied_demos[task_idx]
        return self._demo_to_plan(demo, task.init, ground_operators)

    @staticmethod
    def _demo_to_plan(
        demo: List[str], init: Set[GroundAtom],
        ground_operators: List[_GroundSTRIPSOperator]
    ) -> Sequence[Set[GroundAtom]]:
        # Organize ground operators for fast lookup.
        ground_op_map: Dict[Tuple[str, Tuple[str, ...]],
                            _GroundSTRIPSOperator] = {}
        for op in ground_operators:
            ground_op_map[(op.name, tuple(o.name for o in op.objects))] = op

        # Parse demo into ground ops.
        ground_op_demo = []
        for action_str in demo:
            action_str = action_str.strip()
            assert action_str.startswith("(")
            assert action_str.endswith(")")
            action_str = action_str[1:-1]
            op_name, arg_str = action_str.split(" ", 1)
            arg_names = tuple(arg_str.split(" "))
            ground_op = ground_op_map[(op_name, arg_names)]
            ground_op_demo.append(ground_op)

        # Roll the demo forward. If invalid preconditions are encountered,
        # stop the demo there and return the plan prefix.
        atoms_seq = [init]
        atoms = init
        for op in ground_op_demo:
            if not op.preconditions.issubset(atoms):
                break
            atoms = utils.apply_operator(op, atoms)
            atoms_seq.append(atoms)

        return atoms_seq, ground_op_demo
