"""Heuristics for policy search."""
from __future__ import annotations

import abc
import functools
import logging
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, \
    Tuple

from typing_extensions import TypeAlias

from pg3 import utils
from pg3.search import run_astar, run_policy_guided_astar
from pg3.structs import GroundAtom, LiftedDecisionList, Object, Predicate, \
    STRIPSOperator, Task, _GroundSTRIPSOperator


class _PlanningFailure(Exception):
    """Raised when planning for demo generation fails."""


class _PG3Heuristic(abc.ABC):
    """Given an LDL policy, produce a score, with lower better."""

    def __init__(
        self,
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
        train_tasks: Sequence[Task],
        horizon: int,
        demos: Optional[List[List[str]]] = None,
        task_planning_heuristic: str = "lmcut",
        max_policy_guided_rollout: int = 50,
    ) -> None:
        self._predicates = predicates
        self._operators = operators
        self._train_tasks = train_tasks
        self._horizon = horizon
        self._user_supplied_demos = demos
        self._task_planning_heuristic = task_planning_heuristic
        self._max_policy_guided_rollout = max_policy_guided_rollout

    def __call__(self, ldl: LiftedDecisionList) -> float:
        """Compute the heuristic value for the given LDL policy."""
        score = 0.0
        for idx in range(len(self._train_tasks)):
            score += self._get_score_for_task(ldl, idx)
        logging.debug(f"Scoring:\n{ldl}\nScore: {score}")
        return score

    @abc.abstractmethod
    def _get_score_for_task(self, ldl: LiftedDecisionList,
                            task_idx: int) -> float:
        """Produce a score, with lower better."""
        raise NotImplementedError("Override me!")


class _PolicyEvaluationPG3Heuristic(_PG3Heuristic):
    """Score a policy based on the number of train tasks it solves at the
    abstract level."""

    def _get_score_for_task(self, ldl: LiftedDecisionList,
                            task_idx: int) -> float:
        task = self._train_tasks[task_idx]
        if self._ldl_solves_abstract_task(ldl, task.init, task.objects,
                                          task.goal):
            return 0.0
        return 1.0

    def _ldl_solves_abstract_task(self, ldl: LiftedDecisionList,
                                  atoms: Set[GroundAtom], objects: Set[Object],
                                  goal: Set[GroundAtom]) -> bool:
        for _ in range(self._horizon):
            if goal.issubset(atoms):
                return True
            ground_op = utils.query_ldl(ldl, atoms, objects, goal)
            if ground_op is None:
                return False
            atoms = utils.apply_operator(ground_op, atoms)
        return goal.issubset(atoms)


class _PlanComparisonPG3Heuristic(_PG3Heuristic):
    """Score a policy based on agreement with certain plans.

    Which plans are used to compute agreement is defined by subclasses.
    """
    _plan_compare_inapplicable_cost: float = 0.99

    def __init__(
        self,
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
        train_tasks: Sequence[Task],
        horizon: int,
        demos: Optional[List[List[str]]] = None,
        task_planning_heuristic: str = "lmcut",
        max_policy_guided_rollout: int = 50,
    ) -> None:
        super().__init__(predicates, operators, train_tasks, horizon, demos,
                         task_planning_heuristic, max_policy_guided_rollout)
        # Ground the STRIPSOperators once per task and save them.
        self._train_task_idx_to_ground_operators = {
            idx: [
                ground_op for op in operators
                for ground_op in utils.all_ground_operators(op, task.objects)
            ]
            for idx, task in enumerate(self._train_tasks)
        }

    def _get_score_for_task(self, ldl: LiftedDecisionList,
                            task_idx: int) -> float:
        try:
            atom_plan = self._get_atom_plan_for_task(ldl, task_idx)
        except _PlanningFailure:
            return self._horizon  # worst possible score
        # Note: we need the goal because it's an input to the LDL policy.
        task = self._train_tasks[task_idx]
        return self._count_missed_steps(ldl, atom_plan, task.objects,
                                        task.goal)

    @abc.abstractmethod
    def _get_atom_plan_for_task(self, ldl: LiftedDecisionList,
                                task_idx: int) -> Sequence[Set[GroundAtom]]:
        """Given a task, get the plan with which we will compare the policy."""
        raise NotImplementedError("Override me!")

    def _count_missed_steps(self, ldl: LiftedDecisionList,
                            atoms_seq: Sequence[Set[GroundAtom]],
                            objects: Set[Object],
                            goal: Set[GroundAtom]) -> float:
        missed_steps = 0.0
        for t in range(len(atoms_seq) - 1):
            ground_op = utils.query_ldl(ldl, atoms_seq[t], objects, goal)
            if ground_op is None:
                missed_steps += self._plan_compare_inapplicable_cost
            else:
                predicted_atoms = utils.apply_operator(ground_op, atoms_seq[t])
                if predicted_atoms != atoms_seq[t + 1]:
                    missed_steps += 1
        return missed_steps


class _DemoPlanComparisonPG3Heuristic(_PlanComparisonPG3Heuristic):
    """Score a policy based on agreement with demo plans.

    The demos are generated with a planner, once per train task.
    """

    def _get_atom_plan_for_task(self, ldl: LiftedDecisionList,
                                task_idx: int) -> Sequence[Set[GroundAtom]]:
        del ldl  # unused
        return self._get_demo_atom_plan_for_task(task_idx)

    @functools.lru_cache(maxsize=None)
    def _get_demo_atom_plan_for_task(
            self, task_idx: int) -> Sequence[Set[GroundAtom]]:
        # Run planning once per task and cache the result.

        task = self._train_tasks[task_idx]
        objects, init, goal = task.objects, task.init, task.goal
        ground_operators = self._train_task_idx_to_ground_operators[task_idx]

        if self._user_supplied_demos is not None:
            demo = self._user_supplied_demos[task_idx]
            return self._demo_to_atom_plan(demo, init, ground_operators)

        # Set up an A* search.
        _S: TypeAlias = FrozenSet[GroundAtom]
        _A: TypeAlias = _GroundSTRIPSOperator

        def check_goal(atoms: _S) -> bool:
            return goal.issubset(atoms)

        def get_successors(atoms: _S) -> Iterator[Tuple[_A, _S, float]]:
            for op in utils.get_applicable_operators(ground_operators, atoms):
                next_atoms = utils.apply_operator(op, set(atoms))
                yield (op, frozenset(next_atoms), 1.0)

        heuristic = utils.create_task_planning_heuristic(
            heuristic_name=self._task_planning_heuristic,
            init_atoms=init,
            goal=goal,
            ground_ops=ground_operators,
            predicates=self._predicates,
            objects=objects,
        )

        planned_frozen_atoms_seq, _ = run_astar(initial_state=frozenset(init),
                                                check_goal=check_goal,
                                                get_successors=get_successors,
                                                heuristic=heuristic)

        if not check_goal(planned_frozen_atoms_seq[-1]):
            raise _PlanningFailure()

        return [set(atoms) for atoms in planned_frozen_atoms_seq]

    @staticmethod
    def _demo_to_atom_plan(
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

        return atoms_seq


class _PolicyGuidedPG3Heuristic(_PlanComparisonPG3Heuristic):
    """Score a policy based on agreement with policy-guided plans."""

    def _get_atom_plan_for_task(self, ldl: LiftedDecisionList,
                                task_idx: int) -> Sequence[Set[GroundAtom]]:

        task = self._train_tasks[task_idx]
        objects, init, goal = task.objects, task.init, task.goal
        ground_operators = self._train_task_idx_to_ground_operators[task_idx]

        # Set up a policy-guided A* search.
        _S: TypeAlias = FrozenSet[GroundAtom]
        _A: TypeAlias = _GroundSTRIPSOperator

        def check_goal(atoms: _S) -> bool:
            return goal.issubset(atoms)

        def get_valid_actions(atoms: _S) -> Iterator[Tuple[_A, float]]:
            for op in utils.get_applicable_operators(ground_operators, atoms):
                yield (op, 1.0)

        def get_next_state(atoms: _S, ground_op: _A) -> _S:
            return frozenset(utils.apply_operator(ground_op, set(atoms)))

        heuristic = utils.create_task_planning_heuristic(
            heuristic_name=self._task_planning_heuristic,
            init_atoms=init,
            goal=goal,
            ground_ops=ground_operators,
            predicates=self._predicates,
            objects=objects,
        )

        def policy(atoms: _S) -> Optional[_A]:
            return utils.query_ldl(ldl, set(atoms), objects, goal)

        planned_frozen_atoms_seq, _ = run_policy_guided_astar(
            initial_state=frozenset(init),
            check_goal=check_goal,
            get_valid_actions=get_valid_actions,
            get_next_state=get_next_state,
            heuristic=heuristic,
            policy=policy,
            num_rollout_steps=self._max_policy_guided_rollout,
            rollout_step_cost=0)

        if not check_goal(planned_frozen_atoms_seq[-1]):
            raise _PlanningFailure()

        return [set(atoms) for atoms in planned_frozen_atoms_seq]
