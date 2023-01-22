"""PG3 policy search."""

from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

from typing_extensions import TypeAlias

from pg3 import utils
from pg3.heuristics import _DemoPlanComparisonPG3Heuristic, _PG3Heuristic, \
    _PolicyEvaluationPG3Heuristic, _PolicyGuidedPG3Heuristic
from pg3.operators import _AddConditionPG3SearchOperator, \
    _AddRulePG3SearchOperator, _PG3SearchOperator, _BottomUpOperatorPG3SearchOperator
from pg3.search import run_gbfs, run_hill_climbing
from pg3.structs import LiftedDecisionList, Predicate, STRIPSOperator, Task


def learn_policy(domain_str: str,
                 problem_strs: List[str],
                 horizon: int,
                 demos: Optional[List[List[str]]] = None,
                 heuristic_name: str = "policy_guided",
                 search_method: str = "hill_climbing",
                 task_planning_heuristic: str = "lmcut",
                 max_policy_guided_rollout: int = 50,
                 gbfs_max_expansions: int = 100,
                 hc_enforced_depth: int = 0) -> str:
    """Outputs a string representation of a lifted decision list."""
    if demos is not None:
        assert len(demos) == len(problem_strs), "Supply one demo per problem."
        assert heuristic_name == "demo_plan_comparison", \
            ("Only supply demos if using demo_plan_comparison heuristic, and "
             "even then, the demos are optional.")
    types, predicates, operators = utils.parse_pddl_domain(domain_str)
    train_tasks = [
        utils.pddl_problem_str_to_task(problem_str, domain_str, types,
                                       predicates)
        for problem_str in problem_strs
    ]
    ldl = _run_policy_search(predicates, operators, train_tasks, horizon,
                             demos, heuristic_name, search_method,
                             task_planning_heuristic,
                             max_policy_guided_rollout, gbfs_max_expansions,
                             hc_enforced_depth)
    return str(ldl)


def _run_policy_search(predicates: Set[Predicate],
                       operators: Set[STRIPSOperator],
                       train_tasks: Sequence[Task],
                       horizon: int,
                       demos: Optional[List[List[str]]] = None,
                       heuristic_name: str = "policy_guided",
                       search_method: str = "hill_climbing",
                       task_planning_heuristic: str = "lmcut",
                       max_policy_guided_rollout: int = 50,
                       gbfs_max_expansions: int = 100,
                       hc_enforced_depth: int = 0) -> LiftedDecisionList:
    """Search for a lifted decision list policy that solves the training
    tasks."""
    # Set up a search over LDL space.
    _S: TypeAlias = LiftedDecisionList
    # An "action" here is a search operator and an integer representing the
    # count of successors generated by that operator.
    _A: TypeAlias = Tuple[_PG3SearchOperator, int]

    # Create the PG3 search operators.
    search_operators = _create_search_operators(predicates, operators, train_tasks, demos)

    # The heuristic is what distinguishes PG3 from baseline approaches.
    heuristic = _create_heuristic(heuristic_name, predicates, operators,
                                  train_tasks, horizon, demos,
                                  task_planning_heuristic,
                                  max_policy_guided_rollout)

    # Initialize the search with an empty list.
    initial_state = LiftedDecisionList([])

    def get_successors(ldl: _S) -> Iterator[Tuple[_A, _S, float]]:
        for op in search_operators:
            for i, child in enumerate(op.get_successors(ldl)):
                yield (op, i), child, 1.0  # cost always 1

    if search_method == "gbfs":
        # Terminate only after max expansions.
        path, _ = run_gbfs(initial_state=initial_state,
                           check_goal=lambda _: False,
                           get_successors=get_successors,
                           heuristic=heuristic,
                           max_expansions=gbfs_max_expansions,
                           lazy_expansion=True)

    elif search_method == "hill_climbing":
        # Terminate when no improvement is found.
        path, _, _ = run_hill_climbing(initial_state=initial_state,
                                       check_goal=lambda _: False,
                                       get_successors=get_successors,
                                       heuristic=heuristic,
                                       early_termination_heuristic_thresh=0,
                                       enforced_depth=hc_enforced_depth)

    else:
        raise NotImplementedError("Unrecognized search_method "
                                  f"{search_method}.")

    # Return the best seen policy.
    best_ldl = path[-1]
    return best_ldl


def _create_search_operators(
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
            train_tasks: Sequence[Task],
                      demos: Optional[List[List[str]]]) -> List[_PG3SearchOperator]:
    search_operator_classes = [
        _BottomUpOperatorPG3SearchOperator,
        _AddRulePG3SearchOperator,
        _AddConditionPG3SearchOperator,
    ]
    return [cls(predicates, operators, train_tasks, demos) for cls in search_operator_classes]


def _create_heuristic(heuristic_name: str, predicates: Set[Predicate],
                      operators: Set[STRIPSOperator],
                      train_tasks: Sequence[Task], horizon: int,
                      demos: Optional[List[List[str]]],
                      task_planning_heuristic: str,
                      max_policy_guided_rollout: int) -> _PG3Heuristic:
    heuristic_name_to_cls: Dict[str, TypingType[_PG3Heuristic]] = {
        "policy_guided": _PolicyGuidedPG3Heuristic,
        "policy_evaluation": _PolicyEvaluationPG3Heuristic,
        "demo_plan_comparison": _DemoPlanComparisonPG3Heuristic,
    }
    cls = heuristic_name_to_cls[heuristic_name]
    return cls(predicates, operators, train_tasks, horizon, demos,
               task_planning_heuristic, max_policy_guided_rollout)
