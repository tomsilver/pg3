"""Tests for policy_search.py."""

import pytest

from pg3.policy_search import run_policy_search
from pg3.structs import GroundAtom, LiftedAtom, Object, Predicate, \
    STRIPSOperator, Task, Type, Variable


def test_run_policy_search():
    """Tests for run_policy_search()."""
    loc_type = Type("loc")
    paper_type = Type("paper")
    at = Predicate("at", [loc_type])
    home_base = Predicate("ishomebase", [loc_type])
    satisfied = Predicate("satisfied", [loc_type])
    wants_paper = Predicate("wantspaper", [loc_type])
    safe = Predicate("safe", [loc_type])
    unpacked = Predicate("unpacked", [paper_type])
    carrying = Predicate("carrying", [paper_type])
    preds = {at, home_base, satisfied, wants_paper, safe, unpacked, carrying}
    paper_var = Variable("?paper", paper_type)
    loc_var = Variable("?loc", loc_type)
    pick_up_operator = STRIPSOperator(name="pick-up",
                                      parameters=[paper_var, loc_var],
                                      preconditions={
                                          LiftedAtom(at, [loc_var]),
                                          LiftedAtom(home_base, [loc_var]),
                                          LiftedAtom(unpacked, [paper_var]),
                                      },
                                      add_effects={
                                          LiftedAtom(carrying, [paper_var]),
                                      },
                                      delete_effects={
                                          LiftedAtom(unpacked, [paper_var]),
                                      })
    deliver_operator = STRIPSOperator(name="deliver",
                                      parameters=[paper_var, loc_var],
                                      preconditions={
                                          LiftedAtom(at, [loc_var]),
                                          LiftedAtom(carrying, [paper_var]),
                                      },
                                      add_effects={
                                          LiftedAtom(satisfied, [loc_var]),
                                      },
                                      delete_effects={
                                          LiftedAtom(carrying, [paper_var]),
                                          LiftedAtom(wants_paper, [loc_var]),
                                      })
    from_var = Variable("?from", loc_type)
    to_var = Variable("?to", loc_type)
    move_operator = STRIPSOperator(name="move",
                                   parameters=[from_var, to_var],
                                   preconditions={
                                       LiftedAtom(at, [from_var]),
                                       LiftedAtom(safe, [from_var]),
                                   },
                                   add_effects={
                                       LiftedAtom(at, [to_var]),
                                   },
                                   delete_effects={
                                       LiftedAtom(at, [from_var]),
                                   })
    operators = {pick_up_operator, deliver_operator, move_operator}

    horizon = 50

    paper1 = Object("paper1", paper_type)
    loc1 = Object("loc1", loc_type)
    loc2 = Object("loc2", loc_type)
    objects = {paper1, loc1, loc2}
    init = {
        GroundAtom(home_base, [loc2]),
        GroundAtom(satisfied, [loc2]),
        GroundAtom(at, [loc2]),
        GroundAtom(wants_paper, [loc1]),
        GroundAtom(safe, [loc1]),
        GroundAtom(safe, [loc2]),
        GroundAtom(unpacked, [paper1]),
    }
    goal = {GroundAtom(satisfied, [loc1])}
    task1 = Task(objects, init, goal)

    paper1 = Object("paper1", paper_type)
    paper2 = Object("paper2", paper_type)
    loc1 = Object("loc1", loc_type)
    loc2 = Object("loc2", loc_type)
    loc3 = Object("loc3", loc_type)
    loc4 = Object("loc4", loc_type)
    objects = {paper1, paper2, loc1, loc2, loc3, loc4}
    init = {
        GroundAtom(home_base, [loc1]),
        GroundAtom(satisfied, [loc1]),
        GroundAtom(at, [loc1]),
        GroundAtom(wants_paper, [loc2]),
        GroundAtom(safe, [loc1]),
        GroundAtom(safe, [loc2]),
        GroundAtom(unpacked, [paper1]),
        GroundAtom(unpacked, [paper2]),
    }
    goal = {GroundAtom(satisfied, [loc2])}
    task2 = Task(objects, init, goal)

    paper1 = Object("paper1", paper_type)
    paper2 = Object("paper2", paper_type)
    paper3 = Object("paper3", paper_type)
    paper4 = Object("paper4", paper_type)
    paper5 = Object("paper5", paper_type)
    loc1 = Object("loc1", loc_type)
    loc2 = Object("loc2", loc_type)
    loc3 = Object("loc3", loc_type)
    loc4 = Object("loc4", loc_type)
    loc5 = Object("loc5", loc_type)
    loc6 = Object("loc6", loc_type)
    objects = {
        paper1, paper2, paper3, paper4, paper5, loc1, loc2, loc3, loc4, loc5,
        loc6
    }
    init = {
        GroundAtom(home_base, [loc1]),
        GroundAtom(satisfied, [loc1]),
        GroundAtom(at, [loc1]),
        GroundAtom(wants_paper, [loc4]),
        GroundAtom(wants_paper, [loc5]),
        GroundAtom(wants_paper, [loc6]),
        GroundAtom(safe, [loc1]),
        GroundAtom(safe, [loc4]),
        GroundAtom(safe, [loc5]),
        GroundAtom(safe, [loc6]),
        GroundAtom(unpacked, [paper1]),
        GroundAtom(unpacked, [paper2]),
        GroundAtom(unpacked, [paper3]),
        GroundAtom(unpacked, [paper4]),
        GroundAtom(unpacked, [paper5]),
    }
    goal = {GroundAtom(satisfied, [loc2])}
    task3 = Task(objects, init, goal)

    train_tasks = [task1, task2, task3]

    policy = run_policy_search(preds,
                               operators,
                               train_tasks,
                               horizon,
                               heuristic_name="demo_plan_comparison")
    assert str(policy) == """LiftedDecisionList[
LDLRule-deliver:
    Parameters: [?loc:loc, ?paper:paper]
    Pos State Pre: [at(?loc:loc), carrying(?paper:paper)]
    Neg State Pre: [ishomebase(?loc:loc)]
    Goal Pre: []
    Operator: deliver(?paper:paper, ?loc:loc)
LDLRule-pick-up:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    Operator: pick-up(?paper:paper, ?loc:loc)
LDLRule-move:
    Parameters: [?from:loc, ?to:loc]
    Pos State Pre: [at(?from:loc), safe(?from:loc)]
    Neg State Pre: []
    Goal Pre: []
    Operator: move(?from:loc, ?to:loc)
]"""

    policy = run_policy_search(preds,
                               operators,
                               train_tasks,
                               horizon,
                               heuristic_name="demo_plan_comparison",
                               search_method="gbfs",
                               gbfs_max_expansions=0)
    assert len(policy.rules) == 0

    with pytest.raises(NotImplementedError) as e:
        run_policy_search(preds,
                          operators,
                          train_tasks,
                          horizon,
                          search_method="not a real search method")
    assert "Unrecognized search_method" in str(e)
