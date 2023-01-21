"""Tests for structs.py."""

import pytest

from pg3 import utils
from pg3.structs import GroundAtom, LDLRule, LiftedAtom, LiftedDecisionList, \
    Object, Predicate, STRIPSOperator, Type, Variable, _Atom, \
    _GroundSTRIPSOperator


def test_object_type():
    """Tests for Type class."""
    name = "test"
    my_type = Type(name)
    assert my_type.name == name
    assert isinstance(hash(my_type), int)
    name = "test2"
    my_type2 = Type(name, parent=my_type)
    assert my_type2.name == name
    assert isinstance(hash(my_type2), int)
    assert my_type2.parent == my_type
    name = "test2"
    my_type3 = Type(name, parent=my_type)  # same as my_type2
    obj = Object("obj1", my_type)
    assert obj.is_instance(my_type)
    assert not obj.is_instance(my_type2)
    assert not obj.is_instance(my_type3)
    obj = Object("obj2", my_type2)
    assert obj.is_instance(my_type)
    assert obj.is_instance(my_type2)
    assert obj.is_instance(my_type3)


def test_object():
    """Tests for Object class."""
    my_name = "obj"
    my_type = Type("type")
    obj = Object(my_name, my_type)
    assert isinstance(obj, Object)
    assert obj.name == my_name
    assert obj.type == my_type
    assert str(obj) == repr(obj) == "obj:type"
    assert isinstance(hash(obj), int)
    with pytest.raises(AssertionError):
        Object("?obj", my_type)  # name cannot start with ?


def test_variable():
    """Tests for Variable class."""
    my_name = "?var"
    my_type = Type("type")
    var = Variable(my_name, my_type)
    assert isinstance(var, Variable)
    assert var.name == my_name
    assert var.type == my_type
    assert str(var) == repr(var) == "?var:type"
    assert isinstance(hash(var), int)
    with pytest.raises(AssertionError):
        Variable("var", my_type)  # name must start with ?


def test_predicate_and_atom():
    """Tests for Predicate, LiftedAtom, GroundAtom classes."""
    # Predicates
    cup_type = Type("cup_type")
    plate_type = Type("plate_type")
    pred = Predicate("On", [cup_type, plate_type])
    other_pred = Predicate("On", [cup_type, plate_type])
    assert pred == other_pred
    assert len({pred, other_pred}) == 1
    assert str(pred) == repr(pred) == "On"
    pred2 = Predicate("On2", [cup_type, plate_type])
    assert pred != pred2
    assert pred < pred2
    cup1 = Object("cup1", cup_type)
    plate = Object("plate", plate_type)
    cup_var = Variable("?cup", cup_type)
    plate_var = Variable("?plate", plate_type)
    # Lifted atoms
    lifted_atom = LiftedAtom(pred, [cup_var, plate_var])
    lifted_atom2 = LiftedAtom(pred, [cup_var, plate_var])
    lifted_atom3 = LiftedAtom(pred2, [cup_var, plate_var])
    with pytest.raises(AssertionError):
        LiftedAtom(pred2, [cup_var])  # bad arity
    with pytest.raises(AssertionError):
        LiftedAtom(pred2, [plate_var, cup_var])  # bad types
    assert lifted_atom.predicate == pred
    assert lifted_atom.variables == [cup_var, plate_var]
    assert {lifted_atom, lifted_atom2} == {lifted_atom}
    assert lifted_atom == lifted_atom2
    assert lifted_atom < lifted_atom3
    assert sorted([lifted_atom3, lifted_atom]) == [lifted_atom, lifted_atom3]
    assert isinstance(lifted_atom, LiftedAtom)
    assert (str(lifted_atom) == repr(lifted_atom) ==
            "On(?cup:cup_type, ?plate:plate_type)")
    # Ground atoms
    ground_atom = GroundAtom(pred, [cup1, plate])
    assert ground_atom.predicate == pred
    assert ground_atom.objects == [cup1, plate]
    assert {ground_atom} == {ground_atom}
    assert (str(ground_atom) == repr(ground_atom) ==
            "On(cup1:cup_type, plate:plate_type)")
    assert isinstance(ground_atom, GroundAtom)
    lifted_atom3 = ground_atom.lift({cup1: cup_var, plate: plate_var})
    assert lifted_atom3 == lifted_atom
    atom = _Atom(pred, [cup1, plate])
    with pytest.raises(NotImplementedError):
        str(atom)  # abstract class
    unary_predicate = Predicate("Unary", [cup_type])
    with pytest.raises(ValueError) as e:
        GroundAtom(unary_predicate, cup1)  # expecting a sequence of atoms
    assert "Atoms expect a sequence of entities" in str(e)
    with pytest.raises(ValueError) as e:
        LiftedAtom(unary_predicate, cup_var)  # expecting a sequence of atoms
    assert "Atoms expect a sequence of entities" in str(e)


def test_operators():
    """Tests for STRIPSOperator and _GroundSTRIPSOperator."""
    cup_type = Type("cup_type")
    plate_type = Type("plate_type")
    on = Predicate("On", [cup_type, plate_type])
    not_on = Predicate("NotOn", [cup_type, plate_type])
    cup_var = Variable("?cup", cup_type)
    plate_var = Variable("?plate", plate_type)
    parameters = [cup_var, plate_var]
    preconditions = {LiftedAtom(not_on, [cup_var, plate_var])}
    add_effects = {LiftedAtom(on, [cup_var, plate_var])}
    delete_effects = {LiftedAtom(not_on, [cup_var, plate_var])}

    # STRIPSOperator
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects)
    assert str(strips_operator) == repr(strips_operator) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]"""
    assert isinstance(hash(strips_operator), int)
    strips_operator2 = STRIPSOperator("Pick", parameters, preconditions,
                                      add_effects, delete_effects)
    assert strips_operator == strips_operator2
    strips_operator3 = STRIPSOperator("PickDuplicate", parameters,
                                      preconditions, add_effects,
                                      delete_effects)
    assert strips_operator < strips_operator3
    assert strips_operator3 > strips_operator
    assert str(strips_operator) == repr(strips_operator) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]"""

    # _GroundSTRIPSOperator
    cup = Object("cup", cup_type)
    plate = Object("plate", plate_type)
    ground_op = strips_operator.ground((cup, plate))
    assert isinstance(ground_op, _GroundSTRIPSOperator)
    assert ground_op.parent is strips_operator
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-Pick:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: [NotOn(cup:cup_type, plate:plate_type)]
    Add Effects: [On(cup:cup_type, plate:plate_type)]
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]"""
    ground_op2 = strips_operator2.ground((cup, plate))
    ground_op3 = strips_operator3.ground((cup, plate))
    assert ground_op == ground_op2
    assert ground_op < ground_op3
    assert ground_op3 > ground_op
    assert hash(ground_op) == hash(ground_op2)
    assert ground_op2.pddl_str == "(Pick cup plate)"
    noop_strips_operator = STRIPSOperator("Noop", [], set(), set(), set())
    ground_noop = noop_strips_operator.ground(tuple())
    assert ground_noop.pddl_str == "(Noop)"


def test_lifted_decision_lists():
    """Tests for LDLRule, _GroundLDLRule, LiftedDecisionList."""
    cup_type = Type("cup_type")
    plate_type = Type("plate_type")
    robot_type = Type("robot_type")
    on = Predicate("On", [cup_type, plate_type])
    not_on = Predicate("NotOn", [cup_type, plate_type])
    on_table = Predicate("OnTable", [cup_type])
    holding = Predicate("Holding", [cup_type])
    hand_empty = Predicate("HandEmpty", [robot_type])
    cup_var = Variable("?cup", cup_type)
    plate_var = Variable("?plate", plate_type)
    robot_var = Variable("?robot", robot_type)
    pick_op = STRIPSOperator("Pick",
                             parameters=[cup_var],
                             preconditions={LiftedAtom(on_table, [cup_var])},
                             add_effects={LiftedAtom(holding, [cup_var])},
                             delete_effects={LiftedAtom(on_table, [cup_var])})

    place_op = STRIPSOperator(
        "Place",
        parameters=[cup_var, plate_var],
        preconditions={LiftedAtom(holding, [cup_var])},
        add_effects={LiftedAtom(on, [cup_var, plate_var])},
        delete_effects={LiftedAtom(not_on, [cup_var, plate_var])})

    # LDLRule
    pick_rule = LDLRule(
        "MyPickRule",
        parameters=[cup_var, plate_var, robot_var],
        pos_state_preconditions={
            LiftedAtom(on_table, [cup_var]),
            LiftedAtom(hand_empty, [robot_var])
        },
        neg_state_preconditions={LiftedAtom(holding, [cup_var])},
        goal_preconditions={LiftedAtom(on, [cup_var, plate_var])},
        operator=pick_op)

    assert str(pick_rule) == repr(pick_rule) == """(:rule MyPickRule
    :parameters (?cup - cup_type ?plate - plate_type ?robot - robot_type)
    :preconditions (and (HandEmpty ?robot) (OnTable ?cup) (not (Holding ?cup)))
    :goals (On ?cup ?plate)
    :action (Pick ?cup)
  )"""

    place_rule = LDLRule(
        "MyPlaceRule",
        parameters=[cup_var, plate_var],
        pos_state_preconditions={LiftedAtom(holding, [cup_var])},
        neg_state_preconditions=set(),
        goal_preconditions={LiftedAtom(on, [cup_var, plate_var])},
        operator=place_op)

    assert str(place_rule) == repr(place_rule) == """(:rule MyPlaceRule
    :parameters (?cup - cup_type ?plate - plate_type)
    :preconditions (Holding ?cup)
    :goals (On ?cup ?plate)
    :action (Place ?cup ?plate)
  )"""

    assert pick_rule != place_rule

    pick_rule2 = LDLRule(
        "MyPickRule",
        parameters=[cup_var, plate_var, robot_var],
        pos_state_preconditions={
            LiftedAtom(on_table, [cup_var]),
            LiftedAtom(hand_empty, [robot_var])
        },
        neg_state_preconditions={LiftedAtom(holding, [cup_var])},
        goal_preconditions={LiftedAtom(on, [cup_var, plate_var])},
        operator=pick_op)

    assert pick_rule == pick_rule2
    assert pick_rule < place_rule
    assert place_rule > pick_rule

    # Make sure rules are hashable.
    rules = {pick_rule, place_rule}
    assert rules == {pick_rule, place_rule}

    # Test that errors are raised if rules are malformed.
    with pytest.raises(AssertionError):
        _ = LDLRule("MissingStatePreconditionsRule",
                    parameters=[cup_var, plate_var, robot_var],
                    pos_state_preconditions=set(),
                    neg_state_preconditions=set(),
                    goal_preconditions={LiftedAtom(on, [cup_var, plate_var])},
                    operator=pick_op)
    with pytest.raises(AssertionError):
        _ = LDLRule("MissingParametersRule",
                    parameters=[plate_var, robot_var],
                    pos_state_preconditions={
                        LiftedAtom(on_table, [cup_var]),
                        LiftedAtom(hand_empty, [robot_var])
                    },
                    neg_state_preconditions=set(),
                    goal_preconditions={LiftedAtom(on, [cup_var, plate_var])},
                    operator=pick_op)

    # _GroundLDLRule
    cup1 = Object("cup1", cup_type)
    plate1 = Object("plate1", plate_type)
    robot = Object("robot", robot_type)
    ground_pick_rule = pick_rule.ground((cup1, plate1, robot))

    assert str(ground_pick_rule) == repr(
        ground_pick_rule) == """GroundLDLRule-MyPickRule:
    Parameters: [cup1:cup_type, plate1:plate_type, robot:robot_type]
    Pos State Pre: [HandEmpty(robot:robot_type), OnTable(cup1:cup_type)]
    Neg State Pre: [Holding(cup1:cup_type)]
    Goal Pre: [On(cup1:cup_type, plate1:plate_type)]
    Operator: Pick(cup1:cup_type)"""

    ground_place_rule = place_rule.ground((cup1, plate1))

    assert ground_pick_rule != ground_place_rule
    ground_pick_rule2 = pick_rule.ground((cup1, plate1, robot))
    assert ground_pick_rule == ground_pick_rule2
    assert ground_pick_rule < ground_place_rule
    assert ground_place_rule > ground_pick_rule

    # Make sure ground rules are hashable.
    rule_set = {ground_pick_rule, ground_place_rule}
    assert rule_set == {ground_pick_rule, ground_place_rule}

    # LiftedDecisionList
    rules = [place_rule, pick_rule]
    ldl = LiftedDecisionList(rules)
    assert ldl.rules == rules

    assert str(ldl) == """(define (policy)
  (:rule MyPlaceRule
    :parameters (?cup - cup_type ?plate - plate_type)
    :preconditions (Holding ?cup)
    :goals (On ?cup ?plate)
    :action (Place ?cup ?plate)
  )
  (:rule MyPickRule
    :parameters (?cup - cup_type ?plate - plate_type ?robot - robot_type)
    :preconditions (and (HandEmpty ?robot) (OnTable ?cup) (not (Holding ?cup)))
    :goals (On ?cup ?plate)
    :action (Pick ?cup)
  )
)"""

    atoms = {GroundAtom(on_table, [cup1]), GroundAtom(hand_empty, [robot])}
    goal = {GroundAtom(on, [cup1, plate1])}
    objects = {cup1, plate1, robot}

    expected_op = pick_op.ground((cup1, ))
    assert utils.query_ldl(ldl, atoms, objects, goal) == expected_op

    # Test for missing positive static preconditions.
    static_predicates = {hand_empty}  # pretend static for this test
    init_atoms = set()
    assert utils.query_ldl(ldl, atoms, objects, goal, static_predicates,
                           init_atoms) is None

    # Test for present negative static preconditions.
    static_predicates = {holding}  # pretend static for this test
    init_atoms = {GroundAtom(holding, [cup1])}
    assert utils.query_ldl(ldl, atoms, objects, goal, static_predicates,
                           init_atoms) is None

    atoms = {GroundAtom(holding, [cup1])}

    expected_op = place_op.ground((cup1, plate1))
    assert utils.query_ldl(ldl, atoms, objects, goal) == expected_op

    atoms = set()
    assert utils.query_ldl(ldl, atoms, objects, goal) is None

    ldl2 = LiftedDecisionList(rules)
    assert ldl == ldl2

    ldl3 = LiftedDecisionList(rules[::-1])
    assert ldl != ldl3

    ldl4 = LiftedDecisionList([place_rule])
    assert ldl != ldl4

    ldl5 = LiftedDecisionList(rules[:])
    assert ldl == ldl5

    # Make sure lifted decision lists are hashable.
    assert len({ldl, ldl2}) == 1

    # Special cases for strings.
    noop = STRIPSOperator("Noop",
                          parameters=[],
                          preconditions=set(),
                          add_effects=set(),
                          delete_effects=set())
    ldl_rule_no_preconds = LDLRule("MyUniversalRule",
                                   parameters=[cup_var, plate_var, robot_var],
                                   pos_state_preconditions=set(),
                                   neg_state_preconditions=set(),
                                   goal_preconditions={
                                       LiftedAtom(on, [cup_var, plate_var]),
                                       LiftedAtom(hand_empty, [robot_var])
                                   },
                                   operator=noop)
    assert str(ldl_rule_no_preconds) == """(:rule MyUniversalRule
    :parameters (?cup - cup_type ?plate - plate_type ?robot - robot_type)
    :preconditions ()
    :goals (and (HandEmpty ?robot) (On ?cup ?plate))
    :action (Noop )
  )"""
