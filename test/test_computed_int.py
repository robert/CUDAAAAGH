from CUDAAAAGH import ComputedInt


def test_addition():
    a = ComputedInt(5) + ComputedInt(10)
    assert a == ComputedInt(15)


def test_xor():
    a = ComputedInt(0b11001010) ^ ComputedInt(0b11110001)
    assert a == ComputedInt(0b00111011)


def test_gt():
    assert ComputedInt(123456) > ComputedInt(-1)
