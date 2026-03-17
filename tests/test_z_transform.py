import sympy as sp

from z_transform import (
    InverseZTransform,
    KroneckerDelta,
    ZTransform,
    inverse_z_transform,
    z_correspondence,
    z_initial_conds,
    z_transform,
)


def test_inverse_geometric_symbolic_parameter():
    z, n, a = sp.symbols("z n a", integer=True, nonnegative=True)
    got = inverse_z_transform(z / (z - a), z=z, n=n)
    assert sp.simplify(got - a**n) == 0


def test_inverse_constant_is_delta():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    got = inverse_z_transform(1, z=z, n=n)
    assert got == KroneckerDelta(n)


def test_inverse_preserves_delta_plus_constant():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    got = inverse_z_transform(1 + z / (z - 1), z=z, n=n)
    expected = 1 + KroneckerDelta(n)
    assert sp.simplify(got - expected) == 0


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
    x = y(n) + y(n - 2)

    got = z_transform(x, n=n, z=z)
    expected = ZTransform(y(n), n, z) + z ** (-2) * ZTransform(y(n), n, z)
    assert sp.simplify(got - expected) == 0


def test_undef_function_shift_property_with_coefficient():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")

    got = z_transform(3 * y(n - 1), n=n, z=z)
    expected = 3 * z ** (-1) * ZTransform(y(n), n, z)
    assert sp.simplify(got - expected) == 0


def test_undef_function_forward_shift_once():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")

    got = z_transform(y(n + 1), n=n, z=z)
    expected = z * ZTransform(y(n), n, z) - z * y(0)
    assert sp.simplify(got - expected) == 0


def test_undef_function_forward_shift_twice():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")

    got = z_transform(y(n + 2), n=n, z=z)
    expected = z ** 2 * ZTransform(y(n), n, z) - z ** 2 * y(0) - z * y(1)
    assert sp.simplify(got - expected) == 0


def test_z_transform_undef_without_correspondence_uses_placeholder():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    got = z_transform(y(n), n=n, z=z)
    assert got == ZTransform(y(n), n, z)


def test_z_correspondence_for_symbolic_placeholders():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")
    x = sp.Function("x")
    X = sp.Function("X")

    expr = ZTransform(y(n), n, z) + InverseZTransform(X(z), z, n)
    got = z_correspondence(expr, {y: Y, x: X})
    expected = Y(z) + x(n)
    assert sp.simplify(got - expected) == 0


def test_z_correspondence_for_unevaluated_sum_form():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")

    expr = z_transform(y(n), n=n, z=z, evaluate=False)
    got = z_correspondence(expr, {y: Y})
    assert got == Y(z)


def test_z_initial_conds_for_forward_shift_terms():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    Y = sp.Function("Y")

    expr = z_transform(y(n + 2), n=n, z=z)
    expr = z_correspondence(expr, {y: Y})
    with_corr = z_initial_conds(expr, n, {y: [2, 4]})
    expected = z ** 2 * Y(z) - 2 * z ** 2 - 4 * z
    assert sp.simplify(with_corr - expected) == 0


def test_z_initial_conds_replaces_only_declared_functions():
    z, n = sp.symbols("z n", integer=True, nonnegative=True)
    y = sp.Function("y")
    x = sp.Function("x")
    Y = sp.Function("Y")

    expr = z_transform(y(n + 1), n=n, z=z) + x(0)
    expr = z_correspondence(expr, {y: Y})
    with_corr = z_initial_conds(expr, n, {y: [3]})
    expected = z * Y(z) - 3 * z + x(0)
    assert sp.simplify(with_corr - expected) == 0
