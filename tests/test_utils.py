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
