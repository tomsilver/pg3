"""Tests for policy_search.py."""

import pytest

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
    assert policy_str == """LiftedDecisionList[
LDLRule-pick-up:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    Operator: pick-up(?paper:paper, ?loc:loc)
LDLRule-deliver:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), carrying(?paper:paper), wantspaper(?loc:loc)]
    Neg State Pre: []
    Goal Pre: []
    Operator: deliver(?paper:paper, ?loc:loc)
LDLRule-move:
    Parameters: [?from:loc, ?to:loc]
    Pos State Pre: [at(?from:loc), safe(?from:loc), wantspaper(?to:loc)]
    Neg State Pre: []
    Goal Pre: []
    Operator: move(?from:loc, ?to:loc)
]"""

    policy_str = learn_policy(domain_str,
                              problem_strs,
                              horizon=50,
                              heuristic_name="demo_plan_comparison",
                              search_method="gbfs",
                              gbfs_max_expansions=0)
    assert policy_str == """LiftedDecisionList[

]"""

    with pytest.raises(NotImplementedError) as e:
        learn_policy(domain_str,
                     problem_strs,
                     horizon=50,
                     search_method="not a real search method")
    assert "Unrecognized search_method" in str(e)
