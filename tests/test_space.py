import pytest
from rlenergy_gym.utility import space




def test_continuous_space_init():
    with pytest.raises(ValueError) as execinfo:
        test_space_1 = space.ContinuousSpace(12, 10)
    assert str(execinfo.value) == 'The lower bounds is not smaller than the higher bound.'



def test_contains_function():
    test_space = space.ContinuousSpace(-10, 10)
    assert test_space.contains(10) == False
    assert test_space.contains(9.99) == True
    assert test_space.contains(-10) == False
    assert test_space.contains(-9.99)  == True

