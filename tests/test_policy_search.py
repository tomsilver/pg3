"""Tests for policy_search.py."""

import pytest

from pg3 import utils
from pg3.policy_search import learn_policy


def test_learn_policy():
    """Tests for learn_policy()."""

    domain_str = """(define (domain newspapers)
    (:requirements :strips :typing)
    (:types loc paper)
    (:predicates 
        (at ?loc - loc)
        (isHomeBase ?loc - loc)
        (satisfied ?loc - loc)
        (wantsPaper ?loc - loc)
        (unpacked ?paper - paper)
        (carrying ?paper - paper)
        (safe ?loc - loc)
    )
    
    (:action pick-up
        :parameters (?paper - paper ?loc - loc)
        :precondition (and
            (at ?loc)
            (isHomeBase ?loc)
            (unpacked ?paper)
        )
        :effect (and
            (not (unpacked ?paper))
            (carrying ?paper)
        )
    )
    
    (:action move
        :parameters (?from - loc ?to - loc)
        :precondition (and
            (at ?from) 
            (safe ?from)
        )
        :effect (and
            (not (at ?from))
            (at ?to)
        )
    )
    
    (:action deliver
        :parameters (?paper - paper ?loc - loc)
        :precondition (and
            (at ?loc)
            (wantsPaper ?loc)
            (carrying ?paper)
        )
        :effect (and
            (not (carrying ?paper))
            (not (wantsPaper ?loc))
            (satisfied ?loc)
        )
    )
    
)"""

    problem_str1 = """(define (problem newspaper) (:domain newspapers)
  (:objects
	loc1 - loc
	loc2 - loc
	paper1 - paper
  )
  (:init 
	(at loc2)
	(ishomebase loc2)
    (at loc2)
    (safe loc1)
    (safe loc2)
	(unpacked paper1)
	(satisfied loc2)
	(wantspaper loc1)
  )
  (:goal (and
	(satisfied loc1)))
)"""

    problem_str2 = """(define (problem newspaper) (:domain newspapers)
  (:objects
	loc1 - loc
	loc2 - loc
	loc3 - loc
	loc4 - loc
	paper1 - paper
	paper2 - paper
  )
  (:init 
	(at loc1)
	(ishomebase loc1)
    (at loc1)
    (safe loc1)
    (safe loc2)
	(unpacked paper1)
	(unpacked paper2)
	(satisfied loc1)
	(wantspaper loc2)
  )
  (:goal (and
	(satisfied loc2)))
)"""

    problem_str3 = """(define (problem newspaper) (:domain newspapers)
  (:objects
	loc1 - loc
	loc2 - loc
	loc3 - loc
	loc4 - loc
	loc5 - loc
	loc6 - loc
	paper1 - paper
	paper2 - paper
	paper3 - paper
	paper4 - paper
	paper5 - paper
  )
  (:init 
    (isHomeBase loc1)
    (satisfied loc1)
    (at loc1)
    (wantsPaper loc4)
    (wantsPaper loc5)
    (wantsPaper loc6)
    (safe loc1)
    (safe loc4)
    (safe loc5)
    (safe loc6)
    (unpacked paper1)
    (unpacked paper2)
    (unpacked paper3)
    (unpacked paper4)
    (unpacked paper5)
  )
  (:goal (and
	(satisfied loc4) (satisfied loc5) (satisfied loc6)))
)"""

    problem_strs = [problem_str1, problem_str2, problem_str3]

    policy_str = learn_policy(domain_str,
                              problem_strs,
                              horizon=50,
                              heuristic_name="demo_plan_comparison")

    # Test that the policy solves a new task.
    test_problem = """(define (problem newspaper) (:domain newspapers)
  (:objects
	loc1 - loc
	loc2 - loc
	loc3 - loc
	loc4 - loc
	loc5 - loc
	loc6 - loc
    loc7 - loc
	paper1 - paper
	paper2 - paper
	paper3 - paper
	paper4 - paper
	paper5 - paper
    paper6 - paper
  )
  (:init 
    (isHomeBase loc4)
    (satisfied loc4)
    (at loc4)
    (wantsPaper loc5)
    (wantsPaper loc6)
    (wantsPaper loc7)
    (safe loc1)
    (safe loc4)
    (safe loc5)
    (safe loc6)
    (safe loc7)
    (unpacked paper1)
    (unpacked paper2)
    (unpacked paper3)
    (unpacked paper4)
    (unpacked paper5)
    (unpacked paper6)
  )
  (:goal (and
	(satisfied loc5) (satisfied loc6) (satisfied loc7)))
)"""

    types, predicates, operators = utils.parse_pddl_domain(domain_str)
    policy = utils.parse_ldl_from_str(policy_str, types, predicates, operators)
    task = utils.pddl_problem_str_to_task(test_problem, domain_str, types,
                                          predicates)
    state = task.init
    solved = False
    for _ in range(15):
        # Policy solved task.
        if task.goal.issubset(state):
            solved = True
            break
        action = utils.query_ldl(policy, state, task.objects, task.goal)
        assert action is not None
        assert action.preconditions.issubset(state)
        state = utils.apply_operator(action, state)
    assert solved, "Learned policy did not solve task"

    # Test policy learning failure.
    policy_str = learn_policy(domain_str,
                              problem_strs,
                              horizon=50,
                              heuristic_name="demo_plan_comparison",
                              search_method="gbfs",
                              gbfs_max_expansions=0)
    assert policy_str == """(define (policy)\n  \n)"""

    with pytest.raises(NotImplementedError) as e:
        learn_policy(domain_str,
                     problem_strs,
                     horizon=50,
                     search_method="not a real search method")
    assert "Unrecognized search_method" in str(e)

    # Test learning a policy from demonstrations. These demonstrations
    # pick up the newspapers and then stop.
    demo1 = ["(pick-up paper1 loc2)"]

    demo2 = [
        "(pick-up paper1 loc1)",
        "(pick-up paper2 loc1)",
    ]

    demo3 = [
        "(pick-up paper1 loc1)",
        "(pick-up paper2 loc1)",
        "(pick-up paper3 loc1)",
        "(pick-up paper4 loc1)",
        "(pick-up paper5 loc1)",
        # Cover case where an invalid action is included.
        "(pick-up paper1 loc1)"
    ]

    demos = [demo1, demo2, demo3]

    policy_str = learn_policy(domain_str,
                              problem_strs,
                              horizon=50,
                              demos=demos,
                              heuristic_name="demo_plan_comparison")

    policy = utils.parse_ldl_from_str(policy_str, types, predicates, operators)
    assert len(policy.rules) == 1
    assert policy.rules[0].operator.name == "pick-up"

    policy_str = learn_policy(domain_str,
                              problem_strs,
                              horizon=50,
                              max_rule_params=0,
                              heuristic_name="demo_plan_comparison")
    assert policy_str == """(define (policy)\n  \n)"""

    # Test learning from initialization.
    policy_str1 = """(define (policy)\n  \n)"""
    policy_str2 = """(define (policy)
  (:rule pick-up
    :parameters (?loc - loc ?paper - paper)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
  (:rule deliver
    :parameters (?loc - loc ?paper - paper)
    :preconditions (and (at ?loc) (carrying ?paper) (wantspaper ?loc))
    :goals ()
    :action (deliver ?paper ?loc)
  )
  (:rule move
    :parameters (?from - loc ?to - loc)
    :preconditions (and (at ?from) (safe ?from) (wantspaper ?to))
    :goals ()
    :action (move ?from ?to)
  )
)"""

    policy_str = learn_policy(domain_str,
                              problem_strs,
                              horizon=50,
                              search_method="gbfs",
                              gbfs_max_expansions=0,
                              initial_policy_strs=[policy_str1, policy_str2])
    assert policy_str == policy_str2
