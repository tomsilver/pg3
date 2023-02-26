"""Tests for heuristics.py."""

from pg3.heuristics import _PlanComparisonPG3Heuristic, \
    _PolicyEvaluationPG3Heuristic
from pg3.structs import GroundAtom, LDLRule, LiftedAtom, LiftedDecisionList, \
    Object, Predicate, STRIPSOperator, Task, Type, Variable
from pg3.trajectory_gen import _PolicyGuidedPlanningTrajectoryGenerator, \
    _StaticPlanningTrajectoryGenerator


def test_pg3_heuristics():
    """Tests for PG3 heuristic classes."""
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

    pick_up_rule = LDLRule(name="PickUp",
                           parameters=pick_up_operator.parameters,
                           pos_state_preconditions=set(
                               pick_up_operator.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           operator=pick_up_operator)

    paper, loc = deliver_operator.parameters
    assert "paper" in str(paper)
    assert "loc" in str(loc)

    deliver_rule1 = LDLRule(name="Deliver",
                            parameters=[loc, paper],
                            pos_state_preconditions=set(
                                deliver_operator.preconditions),
                            neg_state_preconditions=set(),
                            goal_preconditions=set(),
                            operator=deliver_operator)

    deliver_rule2 = LDLRule(
        name="Deliver",
        parameters=[loc, paper],
        pos_state_preconditions=set(deliver_operator.preconditions),
        neg_state_preconditions={LiftedAtom(satisfied, [loc])},  # different
        goal_preconditions=set(),
        operator=deliver_operator)

    from_loc, to_loc = move_operator.parameters
    assert "from" in str(from_loc)
    assert "to" in str(to_loc)

    move_rule1 = LDLRule(name="Move",
                         parameters=[from_loc, to_loc],
                         pos_state_preconditions=set(
                             move_operator.preconditions),
                         neg_state_preconditions=set(),
                         goal_preconditions=set(),
                         operator=move_operator)

    move_rule2 = LDLRule(
        name="Move",
        parameters=[from_loc, to_loc],
        pos_state_preconditions=set(move_operator.preconditions) | \
                                {LiftedAtom(wants_paper, [to_loc])},
        neg_state_preconditions=set(),
        goal_preconditions=set(),
        operator=move_operator
    )

    # Scores should monotonically decrease.
    policy_sequence = [
        LiftedDecisionList([]),
        LiftedDecisionList([pick_up_rule]),
        LiftedDecisionList([pick_up_rule, deliver_rule1]),
        LiftedDecisionList([pick_up_rule, deliver_rule2]),
        LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule1]),
        LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule2]),
    ]

    # The policy-guided heuristic should strictly decrease.
    traj_gen = _PolicyGuidedPlanningTrajectoryGenerator(preds, operators)
    heuristic = _PlanComparisonPG3Heuristic(traj_gen, train_tasks, horizon)
    score_sequence = [heuristic(ldl) for ldl in policy_sequence]
    for i in range(len(score_sequence) - 1):
        assert score_sequence[i] > score_sequence[i + 1]

    # Make sure doesn't crash when planning fails.
    ldl = policy_sequence[0]
    traj_gen = _PolicyGuidedPlanningTrajectoryGenerator(preds, set())
    heuristic = _PlanComparisonPG3Heuristic(traj_gen, train_tasks, horizon)
    assert heuristic(ldl) > 0

    # The baseline score functions should decrease (not strictly).
    traj_gen = _StaticPlanningTrajectoryGenerator(preds, operators)
    heuristic = _PlanComparisonPG3Heuristic(traj_gen, train_tasks, horizon)
    score_sequence = [heuristic(ldl) for ldl in policy_sequence]
    for i in range(len(score_sequence) - 1):
        assert score_sequence[i] >= score_sequence[i + 1]

    # Make sure doesn't crash when planning fails.
    ldl = policy_sequence[0]
    traj_gen = _StaticPlanningTrajectoryGenerator(preds, set())
    heuristic = _PlanComparisonPG3Heuristic(traj_gen, train_tasks, horizon)
    assert heuristic(ldl) > 0

    heuristic = _PolicyEvaluationPG3Heuristic(traj_gen, train_tasks, horizon)
    score_sequence = [heuristic(ldl) for ldl in policy_sequence]
    for i in range(len(score_sequence) - 1):
        assert score_sequence[i] >= score_sequence[i + 1]
