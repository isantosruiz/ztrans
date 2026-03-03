import sympy as sp

from z_transform import KroneckerDelta, inverse_z_transform, z_transform


def test_inverse_geometric_symbolic_parameter():
    z, n, a = sp.symbols("z n a", integer=True, nonnegative=True)
    got = inverse_z_transform(z / (z - a), z=z, n=n)
    assert sp.simplify(got - a**n) == 0


def test_inverse_constant_is_delta():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    got = inverse_z_transform(1, z=z, n=n)
    assert got == KroneckerDelta(n)


def test_round_trip_power_sequence():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    F = z_transform(lambda k: 2**k, n=n, z=z)
    got = inverse_z_transform(F, z=z, n=n)
    assert sp.simplify(got - 2**n) == 0


def test_noconds_default_and_condition_output():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    xz = z_transform([1, 2, 3], n=n, z=z)
    with_cond = z_transform([1, 2, 3], n=n, z=z, noconds=False)

    assert not isinstance(xz, tuple)
    assert isinstance(with_cond, tuple)
    assert len(with_cond) == 2
    assert sp.simplify(with_cond[0] - xz) == 0


def test_sin_transform_uses_closed_form():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    F = z_transform(sp.sin(n), n=n, z=z)
    assert not F.has(sp.Sum)
    assert sp.simplify(inverse_z_transform(F, z=z, n=n) - sp.sin(n)) == 0


def test_cos_shifted_transform_uses_closed_form():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    x = sp.cos(3 * n + 2)
    F = z_transform(x, n=n, z=z)
    assert not F.has(sp.Sum)
    assert sp.simplify(inverse_z_transform(F, z=z, n=n) - x) == 0


def test_undef_function_shift_property():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")
    x = y(n) + y(n - 2)

    got = z_transform(x, n=n, z=z)
    expected = Y(z) + z ** (-2) * Y(z)
    assert sp.simplify(got - expected) == 0


def test_undef_function_shift_property_with_coefficient():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")

    got = z_transform(3 * y(n - 1), n=n, z=z)
    expected = 3 * z ** (-1) * Y(z)
    assert sp.simplify(got - expected) == 0


def test_undef_function_forward_shift_once():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")

    got = z_transform(y(n + 1), n=n, z=z)
    expected = z * Y(z) - z * y(0)
    assert sp.simplify(got - expected) == 0


def test_undef_function_forward_shift_twice():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")

    got = z_transform(y(n + 2), n=n, z=z)
    expected = z ** 2 * Y(z) - z ** 2 * y(0) - z * y(1)
    assert sp.simplify(got - expected) == 0
