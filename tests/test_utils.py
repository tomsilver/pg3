"""Tests for utils.py."""

import pytest

from pg3 import utils
from pg3.structs import Type, Variable
from pg3.utils import _PyperplanHeuristicWrapper, _TaskPlanningHeuristic


def test_create_new_variables():
    """Tests for create_new_variables()."""
    cup_type = Type("cup", ["feat1"])
    plate_type = Type("plate", ["feat1"])
    vs = utils.create_new_variables([cup_type, cup_type, plate_type])
    assert vs == [
        Variable("?x0", cup_type),
        Variable("?x1", cup_type),
        Variable("?x2", plate_type)
    ]
    existing_vars = {Variable("?x0", cup_type), Variable("?x5", cup_type)}
    vs = utils.create_new_variables([plate_type], existing_vars=existing_vars)
    assert vs == [Variable("?x6", plate_type)]
    existing_vars = {Variable("?x", cup_type), Variable("?xerox", cup_type)}
    vs = utils.create_new_variables([plate_type], existing_vars=existing_vars)
    assert vs == [Variable("?x0", plate_type)]
    existing_vars = {Variable("?x5", cup_type)}
    vs = utils.create_new_variables([plate_type],
                                    existing_vars=existing_vars,
                                    var_prefix="?llama")
    assert vs == [Variable("?llama0", plate_type)]


@pytest.mark.parametrize("heuristic_name, expected_heuristic_cls", [
    ("hadd", _PyperplanHeuristicWrapper),
    ("hmax", _PyperplanHeuristicWrapper),
    ("hff", _PyperplanHeuristicWrapper),
    ("hsa", _PyperplanHeuristicWrapper),
    ("lmcut", _PyperplanHeuristicWrapper),
])
def test_create_task_planning_heuristic(heuristic_name,
                                        expected_heuristic_cls):
    """Tests for create_task_planning_heuristic()."""
    heuristic = utils.create_task_planning_heuristic(heuristic_name, set(),
                                                     set(), set(), set(),
                                                     set())
    assert isinstance(heuristic, expected_heuristic_cls)


def test_create_task_planning_heuristic_raises_error_for_unknown_heuristic():
    """Test creating unknown heuristic raises a ValueError."""
    with pytest.raises(ValueError):
        utils.create_task_planning_heuristic("not a real heuristic", set(),
                                             set(), set(), set(), set())


def test_create_task_planning_heuristic_base_class():
    """Test to cover _TaskPlanningHeuristic base class."""
    base_heuristic = _TaskPlanningHeuristic("base", set(), set(), set())
    with pytest.raises(NotImplementedError):
        base_heuristic(set())


def test_parse_ldl_from_str():
    """Tests for parse_ldl_from_str()."""
    domain_str = """(define (domain gripper-strips)
   (:predicates (room ?r)
        (ball ?b)
        (gripper ?g)
        (at-robby ?r)
        (at ?b ?r)
        (free ?g)
        (carry ?o ?g))

   (:action move
       :parameters  (?from ?to)
       :precondition (and  (room ?from) (room ?to) (at-robby ?from))
       :effect (and  (at-robby ?to)
             (not (at-robby ?from))))



   (:action pick
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                (at ?obj ?room) (at-robby ?room) (free ?gripper))
       :effect (and (carry ?obj ?gripper)
            (not (at ?obj ?room)) 
            (not (free ?gripper))))


   (:action drop
       :parameters  (?obj  ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                (carry ?obj ?gripper) (at-robby ?room))
       :effect (and (at ?obj ?room)
            (free ?gripper)
            (not (carry ?obj ?gripper)))))"""
    types, predicates, operators = utils.parse_pddl_domain(domain_str)

    # pylint: disable=line-too-long
    ldl_str = """(define (policy)
	(:rule rule1 
		:parameters (?gripper - object ?obj - object ?room - object)		
        :preconditions (and (ball ?obj) (room ?room) (gripper ?gripper) (not (at ?obj ?room)) (carry ?obj ?gripper) (at-robby ?room))
        :goals (at ?obj ?room)
		:action (drop ?obj ?room ?gripper)
	)
	(:rule rule2 
		:parameters (?goalroom - object ?gripper - object ?obj - object ?room - object)		
        :preconditions (and (ball ?obj) (room ?room) (gripper ?gripper) (at ?obj ?room) (at-robby ?room) (free ?gripper) (not (at ?obj ?goalroom)))
        :goals ()
		:action (pick ?obj ?room ?gripper)
	)
	(:rule rule3
		:parameters (?from - object ?gripper - object ?obj - object ?to - object)
		:preconditions (and (room ?from) (room ?to) (at-robby ?from) (carry ?obj ?gripper)) 
		:goals (at ?obj ?to)
		:action (move ?from ?to)
	)
	(:rule rule4
		:parameters (?from - object ?goalroom- object ?gripper - object ?obj - object ?to - object)
		:preconditions (and (room ?from) (room ?to) (at-robby ?from) (free ?gripper) (at ?obj ?to)) 
		:goals (at ?obj ?goalroom)
		:action (move ?from ?to)
	)
)"""

    ldl = utils.parse_ldl_from_str(ldl_str, types, predicates, operators)
    assert str(ldl) == """(define (policy)
  (:rule rule1
    :parameters (?gripper - object ?obj - object ?room - object)
    :preconditions (and (at-robby ?room) (ball ?obj) (carry ?obj ?gripper) (gripper ?gripper) (room ?room) (not (at ?obj ?room)))
    :goals (at ?obj ?room)
    :action (drop ?obj ?room ?gripper)
  )
  (:rule rule2
    :parameters (?goalroom - object ?gripper - object ?obj - object ?room - object)
    :preconditions (and (at ?obj ?room) (at-robby ?room) (ball ?obj) (free ?gripper) (gripper ?gripper) (room ?room) (not (at ?obj ?goalroom)))
    :goals ()
    :action (pick ?obj ?room ?gripper)
  )
  (:rule rule3
    :parameters (?from - object ?gripper - object ?obj - object ?to - object)
    :preconditions (and (at-robby ?from) (carry ?obj ?gripper) (room ?from) (room ?to))
    :goals (at ?obj ?to)
    :action (move ?from ?to)
  )
  (:rule rule4
    :parameters (?from - object ?goalroom - object ?gripper - object ?obj - object ?to - object)
    :preconditions (and (at ?obj ?to) (at-robby ?from) (free ?gripper) (room ?from) (room ?to))
    :goals (at ?obj ?goalroom)
    :action (move ?from ?to)
  )
)"""
