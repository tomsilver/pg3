"""Generate trajectories by planning or with demos."""

import abc
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from typing_extensions import TypeAlias

from pg3 import utils
from pg3.search import run_astar, run_policy_guided_astar
from pg3.structs import GroundAtom, LiftedDecisionList, PlanningFailure, \
    Predicate, STRIPSOperator, Task, Trajectory, _GroundSTRIPSOperator


class _TrajectoryGenerator(abc.ABC):
    """Generate trajectories by planning or with demos."""

    def __init__(self,
                 predicates: Set[Predicate],
                 operators: Set[STRIPSOperator],
                 task_planning_heuristic: str = "lmcut",
                 max_policy_guided_rollout: int = 50,
                 user_supplied_demos: Optional[Dict[Task, List[str]]] = None):
        self._predicates = predicates
        self._operators = operators
        self._task_planning_heuristic = task_planning_heuristic
        self._max_policy_guided_rollout = max_policy_guided_rollout
        self._user_supplied_demos = user_supplied_demos

    @abc.abstractmethod
    def get_trajectory_for_task(self, task: Task,
                                ldl: LiftedDecisionList) -> Trajectory:
        """Create a trajectory given a task."""
        raise NotImplementedError("Override me!")


class _StaticTrajectoryGenerator(_TrajectoryGenerator):
    """Generate trajectories once per training task and cache them."""

    def __init__(self,
                 predicates: Set[Predicate],
                 operators: Set[STRIPSOperator],
                 task_planning_heuristic: str = "lmcut",
                 max_policy_guided_rollout: int = 50,
                 user_supplied_demos: Optional[Dict[Task, List[str]]] = None):
        super().__init__(predicates, operators, task_planning_heuristic,
                         max_policy_guided_rollout, user_supplied_demos)
        self._task_to_trajectory: Dict[Task, Trajectory] = {}

    def get_trajectory_for_task(self, task: Task,
                                ldl: LiftedDecisionList) -> Trajectory:
        if task not in self._task_to_trajectory:
            trajectory = self._get_trajectory_for_task(task)
            self._task_to_trajectory[task] = trajectory
        return self._task_to_trajectory[task]

    @abc.abstractmethod
    def _get_trajectory_for_task(self, task: Task) -> Trajectory:
        raise NotImplementedError("Override me!")


class _UserSuppliedDemoTrajectoryGenerator(_StaticTrajectoryGenerator):
    """Generate trajectories by looking up user-supplied demos."""

    def _get_trajectory_for_task(self, task: Task) -> Trajectory:
        assert self._user_supplied_demos is not None
        demo = self._user_supplied_demos[task]
        return self._demo_to_trajectory(demo, task)

    def _demo_to_trajectory(
        self,
        demo: List[str],
        task: Task,
    ) -> Trajectory:
        # Organize ground operators for fast lookup.
        ground_operators = utils.all_ground_operators(self._operators, task)
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
        atoms_seq = [task.init]
        atoms = task.init
        for op in ground_op_demo:
            if not op.preconditions.issubset(atoms):
                break
            atoms = utils.apply_operator(op, atoms)
            atoms_seq.append(atoms)

        return ground_op_demo, atoms_seq, task


class _StaticPlanningTrajectoryGenerator(_StaticTrajectoryGenerator):
    """Generate trajectories by planning in each task."""

    def _get_trajectory_for_task(self, task: Task) -> Trajectory:
        # Run planning once per task and cache the result.
        objects, init, goal = task.objects, task.init, task.goal
        ground_operators = utils.all_ground_operators(self._operators, task)

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

        planned_frozen_atoms_seq, action_seq = run_astar(
            initial_states=[frozenset(init)],
            check_goal=check_goal,
            get_successors=get_successors,
            heuristic=heuristic)

        if not check_goal(planned_frozen_atoms_seq[-1]):
            raise PlanningFailure()

        atom_seq = [set(atoms) for atoms in planned_frozen_atoms_seq]
        return action_seq, atom_seq, task


class _PolicyGuidedPlanningTrajectoryGenerator(_TrajectoryGenerator):
    """Generate trajectories by policy-guided planning."""

    def get_trajectory_for_task(self, task: Task,
                                ldl: LiftedDecisionList) -> Trajectory:
        objects, init, goal = task.objects, task.init, task.goal
        ground_operators = utils.all_ground_operators(self._operators, task)

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

        planned_frozen_atoms_seq, action_seq = run_policy_guided_astar(
            initial_states=[frozenset(init)],
            check_goal=check_goal,
            get_valid_actions=get_valid_actions,
            get_next_state=get_next_state,
            heuristic=heuristic,
            policy=policy,
            num_rollout_steps=self._max_policy_guided_rollout,
            rollout_step_cost=0)

        if not check_goal(planned_frozen_atoms_seq[-1]):
            raise PlanningFailure()

        atom_seq = [set(atoms) for atoms in planned_frozen_atoms_seq]
        return action_seq, atom_seq, task
