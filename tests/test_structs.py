"""Tests for structs.py."""

import pytest

from pg3.structs import GroundAtom, LiftedAtom, Object, Predicate, Type, \
    Variable, _Atom


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
    cup2 = Object("cup2", cup_type)
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

