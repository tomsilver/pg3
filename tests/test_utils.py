"""Tests for utils.py."""

from pg3 import utils
from pg3.structs import Type, Variable


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
