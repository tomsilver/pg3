"""Tests for operators.py."""

from pg3.operators import _AddConditionPG3SearchOperator, \
    _AddRulePG3SearchOperator, _DeleteConditionPG3SearchOperator, \
    _DeleteRulePG3SearchOperator
from pg3.structs import LDLRule, LiftedAtom, LiftedDecisionList, Predicate, \
    STRIPSOperator, Type, Variable


def test_pg3_search_operators():
    """Tests for PG3 search operator classes."""
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

    pick_up_rule = LDLRule(name="MyPickUp",
                           parameters=pick_up_operator.parameters,
                           pos_state_preconditions=set(
                               pick_up_operator.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           operator=pick_up_operator)

    ldl1 = LiftedDecisionList([])
    ldl2 = LiftedDecisionList([pick_up_rule])

    # _AddRulePG3SearchOperator
    op = _AddRulePG3SearchOperator(preds, operators)

    succ1 = list(op.get_successors(ldl1))
    assert len(succ1) == 3
    ldl1_1, ldl1_2, ldl1_3 = sorted(succ1, key=str)
    assert str(ldl1_1) == """(define (policy)
  (:rule deliver
    :parameters (?paper - paper ?loc - loc)
    :preconditions (and (at ?loc) (carrying ?paper))
    :goals ()
    :action (deliver ?paper ?loc)
  )
)"""

    assert str(ldl1_2) == """(define (policy)
  (:rule move
    :parameters (?from - loc ?to - loc)
    :preconditions (and (at ?from) (safe ?from))
    :goals ()
    :action (move ?from ?to)
  )
)"""

    assert str(ldl1_3) == """(define (policy)
  (:rule pick-up
    :parameters (?paper - paper ?loc - loc)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
)"""

    succ2 = list(op.get_successors(ldl2))
    assert len(succ2) == 3
    ldl2_1 = min(succ2, key=str)
    assert str(ldl2_1) == """(define (policy)
  (:rule MyPickUp
    :parameters (?paper - paper ?loc - loc)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
  (:rule deliver
    :parameters (?paper - paper ?loc - loc)
    :preconditions (and (at ?loc) (carrying ?paper))
    :goals ()
    :action (deliver ?paper ?loc)
  )
)"""

    # _AddConditionPG3SearchOperator
    op = _AddConditionPG3SearchOperator(preds, operators)

    succ1 = list(op.get_successors(ldl1))
    assert len(succ1) == 0

    succ2 = list(op.get_successors(ldl2))
    assert len(succ2) == 36
    ldl2_1 = min(succ2, key=str)
    assert str(ldl2_1) == """(define (policy)
  (:rule MyPickUp
    :parameters (?loc - loc ?paper - paper ?x0 - loc)
    :preconditions (and (at ?loc) (at ?x0) (ishomebase ?loc) (unpacked ?paper))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
)"""

    # _DeleteConditionPG3SearchOperator
    op = _DeleteConditionPG3SearchOperator(preds, operators)

    # Empty rule should have no successors
    succ1 = list(op.get_successors(ldl1))
    assert len(succ1) == 0

    # Should return zero because we don't remove preconditions
    #   that are also preconditions of the operator
    succ2 = list(op.get_successors(ldl2))
    assert len(succ2) == 0

    # Removing only one condition that is not in operator
    succ3 = list(op.get_successors(ldl2_1))
    assert len(succ3) == 1

    assert str(succ3[0]) == """(define (policy)
  (:rule MyPickUp
    :parameters (?loc - loc ?paper - paper)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
)"""

    dummy_1 = Predicate("Dummy", [])  # zero arity
    atom_1 = LiftedAtom(dummy_1, [])

    dummy_2 = Predicate("OtherDummy", [])
    atom_2 = LiftedAtom(dummy_2, [])

    dummy_type = Type("dummytype", ["a", "b"])
    dummy_var = Variable("?dv", dummy_type)
    other_dummy_var = sorted(pick_up_operator.preconditions)[0].variables[0]

    dummy_3 = Predicate("OneMoreDummy", \
        [dummy_type, other_dummy_var.type])
    atom_3 = LiftedAtom(dummy_3, [dummy_var, other_dummy_var])

    pos_preconds = set(pick_up_operator.preconditions).union([atom_1])
    another_pick_up_rule = LDLRule(name="MyOtherPickUp",
                                   parameters=pick_up_operator.parameters +
                                   [dummy_var],
                                   pos_state_preconditions=pos_preconds,
                                   neg_state_preconditions=set([atom_3]),
                                   goal_preconditions=set([atom_2]),
                                   operator=pick_up_operator)

    ldl3 = LiftedDecisionList([another_pick_up_rule])

    succ4 = list(op.get_successors(ldl3))
    assert len(succ4) == 3

    assert str(succ4[0]) == """(define (policy)
  (:rule MyOtherPickUp
    :parameters (?dv - dummytype ?loc - loc ?paper - paper)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper) (not (OneMoreDummy ?dv ?loc)))
    :goals (OtherDummy )
    :action (pick-up ?paper ?loc)
  )
)"""

    assert str(succ4[1]) == """(define (policy)
  (:rule MyOtherPickUp
    :parameters (?loc - loc ?paper - paper)
    :preconditions (and (Dummy ) (at ?loc) (ishomebase ?loc) (unpacked ?paper))
    :goals (OtherDummy )
    :action (pick-up ?paper ?loc)
  )
)"""

    # _DeleteRulePG3SearchOperator
    op = _DeleteRulePG3SearchOperator(preds, operators)

    # Empty list should have no successors
    succ1 = list(op.get_successors(ldl1))
    assert len(succ1) == 0

    # Removing from list with one rule should have 1 empty successor
    succ2 = list(op.get_successors(ldl2))
    assert len(succ2) == 1
    ldl2_succ = next(iter(succ2))
    assert len(ldl2_succ.rules) == 0
