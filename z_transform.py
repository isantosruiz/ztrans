import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.functions.special.tensor_functions import KroneckerDelta as _SympyKroneckerDelta

__all__ = [
    "KroneckerDelta",
    "ZTransform",
    "InverseZTransform",
    "z_correspondence",
    "z_initial_conds",
    "z_transform",
    "inverse_z_transform",
]


def KroneckerDelta(a, b=0):
    """Return Kronecker delta with default second argument b=0."""
    return _SympyKroneckerDelta(a, b)


class ZTransform(sp.Function):
    """Symbolic placeholder for unevaluated unilateral Z-transform."""

    nargs = (3,)


class InverseZTransform(sp.Function):
    """Symbolic placeholder for unevaluated unilateral inverse Z-transform."""

    nargs = (3,)


def z_correspondence(f, fdict, /):
    """Replace formal Z-transform placeholders using function correspondences.

    Similar to SymPy's ``laplace_correspondence``:
    - ``ZTransform(y(n), n, z)`` -> ``Y(z)``
    - ``InverseZTransform(Y(z), z, n)`` -> ``y(n)``
    - ``Sum(y(n)*z**(-n), (n, 0, oo))`` -> ``Y(z)``
    """
    p = sp.Wild("p")
    z = sp.Wild("z")
    n = sp.Wild("n")
    a = sp.Wild("a")
    if not isinstance(f, sp.Expr):
        return f
    if not (f.has(ZTransform) or f.has(InverseZTransform) or f.has(sp.Sum)):
        return f

    for y, Y in fdict.items():
        if (m := f.match(ZTransform(y(a), n, z))) is not None and m[a] == m[n]:
            return Y(m[z])
        if (m := f.match(InverseZTransform(Y(a), z, n))) is not None and m[a] == m[z]:
            return y(m[n])
        if (m := f.match(sp.Sum(y(a) * p ** (-a), (a, 0, sp.oo)))) is not None:
            return Y(m[p])

    func = f.func
    args = [z_correspondence(arg, fdict) for arg in f.args]
    return func(*args)


def z_initial_conds(f, n, fdict, /):
    """Replace discrete initial conditions in Z-domain expressions.

    Similar to SymPy's ``laplace_initial_conds`` but for sequences:
    given ``{y: [y0, y1, y2, ...]}`` this replaces ``y(0)``, ``y(1)``,
    ``y(2)``, ... in ``f``.

    Parameters
    ----------
    f : sympy expression
        Expression containing discrete initial condition terms.
    n : sympy expression
        Discrete index symbol (kept for API symmetry).
    fdict : dict
        Dictionary mapping functions to ordered initial values.
        Example: ``{y: [2, 4, 8]}`` means ``y(0)=2, y(1)=4, y(2)=8``.
    """
    _ = n  # API symmetry with laplace_initial_conds-style helpers.
    for y, ic in fdict.items():
        for k in range(len(ic)):
            f = f.replace(y(sp.Integer(k)), ic[k])
    return f


def _select_piecewise_branch(piecewise_expr):
    """Select the convergent and evaluated branch from a Piecewise expression."""
    for expr, cond in piecewise_expr.args:
        if cond is True or cond == sp.S.true:
            continue
        if not expr.has(sp.Sum):
            return expr, cond

    for expr, cond in piecewise_expr.args:
        if not (cond is True or cond == sp.S.true):
            return expr, cond

    expr, _ = piecewise_expr.args[0]
    return expr, sp.S.true


def _strip_trivial_piecewise(expr):
    """Remove unevaluated fallback branches and collect their conditions."""
    conditions = []

    def _replace(piecewise_expr):
        chosen_expr, cond = _select_piecewise_branch(piecewise_expr)
        if cond != sp.S.true:
            conditions.append(cond)
        return chosen_expr

    cleaned = expr.replace(lambda e: isinstance(e, sp.Piecewise), _replace)

    if not conditions:
        return cleaned, sp.S.true

    condition = sp.simplify(sp.And(*conditions)) if len(conditions) > 1 else conditions[0]
    return cleaned, condition


def _normalize_short_kronecker(expr):
    """Convert short KroneckerDelta(arg) forms into KroneckerDelta(arg, 0)."""

    def _is_short_or_generic_delta(e):
        return isinstance(e, AppliedUndef) and e.func.__name__ == "KroneckerDelta"

    def _replace_delta(e):
        if len(e.args) == 1:
            return _SympyKroneckerDelta(e.args[0], 0)
        if len(e.args) == 2:
            return _SympyKroneckerDelta(e.args[0], e.args[1])
        return e

    return expr.replace(_is_short_or_generic_delta, _replace_delta)


def _to_sympy_expr(x, n, z):
    """Convert input x to a SymPy expression with delta short-form support."""
    if callable(x):
        try:
            raw = x(n)
        except TypeError as exc:
            if "KroneckerDelta takes at least 2 arguments" not in str(exc):
                raise

            # Compatibility path for callables using sp.KroneckerDelta(arg).
            original = sp.KroneckerDelta
            try:
                sp.KroneckerDelta = KroneckerDelta
                raw = x(n)
            finally:
                sp.KroneckerDelta = original

        expr = sp.sympify(raw)

    elif isinstance(x, str):
        expr = sp.sympify(
            x,
            locals={
                "KroneckerDelta": KroneckerDelta,
                "Heaviside": sp.Heaviside,
                str(n): n,
                str(z): z,
            },
        )
    else:
        expr = sp.sympify(x)

    return _normalize_short_kronecker(expr)


def _solve_delta_index(delta_expr, n):
    """Return n=index for KroneckerDelta when there is a unique solution."""
    try:
        a, b = delta_expr.args
        sols = sp.solve(sp.Eq(a, b), n)
    except Exception:
        return None

    if len(sols) != 1:
        return None

    idx = sp.simplify(sols[0])
    if idx.has(n):
        return None

    return idx


def _extract_heaviside_bounds(term, n):
    """Extract unilateral support bounds from Heaviside factors.

    Supported forms:
    - Heaviside(n - k): n >= k
    - Heaviside(k - n): n <= k
    """
    lower = sp.Integer(0)
    upper = sp.oo
    clean = term

    for h in list(clean.atoms(sp.Heaviside)):
        if not h.has(n):
            clean = sp.simplify(clean.subs(h, sp.simplify(h.doit())))
            continue

        arg = sp.expand(h.args[0])
        coeff = sp.simplify(arg.coeff(n))
        rest = sp.simplify(arg - coeff * n)

        if coeff == 1:
            lower = sp.Max(lower, sp.ceiling(-rest))
            clean = clean.subs(h, 1)
        elif coeff == -1:
            upper = sp.Min(upper, sp.floor(rest))
            clean = clean.subs(h, 1)
        else:
            return term, sp.Integer(0), sp.oo, False

    return sp.simplify(clean), lower, upper, True


def _build_delta_closed_form(clean_term, n, z, lower, upper):
    """Collapse n-dependent KroneckerDelta factors to closed form when possible."""
    deltas = [d for d in clean_term.atoms(_SympyKroneckerDelta) if d.has(n)]
    if not deltas:
        return None

    idx_list = []
    out = clean_term

    for d in deltas:
        idx = _solve_delta_index(d, n)
        if idx is None:
            return None
        idx_list.append(idx)
        out = out.subs(d, 1)

    idx0 = idx_list[0]
    consistency = [sp.Eq(idx0, idx) for idx in idx_list[1:]]
    expr = sp.simplify(out.subs(n, idx0) * z ** (-idx0))

    support = [sp.Ge(idx0, lower)]
    if upper != sp.oo:
        support.append(sp.Le(idx0, upper))

    cond = sp.And(*(consistency + support)) if (consistency or support) else sp.S.true
    return sp.Piecewise((expr, cond), (0, True))


def _has_n_trig(expr, n):
    """Return True if expr contains sin/cos terms depending on n."""
    for trig in expr.atoms(sp.sin, sp.cos):
        if trig.has(n):
            return True
    return False


def _sum_linear_exponential_term(term, n, z, lower, upper):
    """Sum term*z**(-n) when term is coeff*exp(alpha*n) with coeff independent of n."""
    coeff = sp.Integer(1)
    alpha = sp.Integer(0)

    for factor in sp.Mul.make_args(term):
        if factor.func == sp.exp:
            arg = sp.expand(factor.args[0])
            c = sp.simplify(arg.coeff(n))
            rest = sp.simplify(arg - c * n)
            if rest.has(n):
                return None

            if c == 0:
                coeff *= factor
            else:
                alpha += c
                coeff *= sp.exp(rest)
        else:
            coeff *= factor

    coeff = sp.simplify(coeff)
    if coeff.has(n):
        return None

    q = sp.simplify(sp.exp(alpha) / z)
    start = sp.simplify(coeff * sp.exp(alpha * lower) * z ** (-lower))

    if upper == sp.oo:
        closed = start / (1 - q)
        return sp.Piecewise((closed, sp.Abs(q) < 1), (sp.S.NaN, True))

    count = sp.simplify(upper - lower + 1)
    closed_ne = sp.simplify(start * (1 - q ** count) / (1 - q))
    closed_eq = sp.simplify(start * count)
    return sp.Piecewise((closed_ne, sp.Ne(q, 1)), (closed_eq, True))


def _build_trig_euler_closed_form(term, n, z, lower, upper):
    """Convert sin/cos(n) terms with Euler identities and sum geometrically."""
    if not _has_n_trig(term, n):
        return None

    rewritten = sp.expand(term.rewrite(sp.exp))
    if rewritten.has(sp.sin, sp.cos):
        return None

    partials = []
    for add_term in rewritten.as_ordered_terms():
        closed = _sum_linear_exponential_term(add_term, n, z, lower, upper)
        if closed is None:
            return None
        partials.append(closed)

    return sp.simplify(sp.Add(*partials))


def _build_undef_function_closed_form(term, n, z, lower, upper):
    """Build formal transform for undefined functions y(n+k)."""
    if lower != 0 or upper != sp.oo:
        return None

    factors = list(sp.Mul.make_args(term))
    undef = [f for f in factors if isinstance(f, AppliedUndef) and f.has(n)]
    if len(undef) != 1:
        return None

    fn = undef[0]
    if len(fn.args) != 1:
        return None

    arg = sp.expand(fn.args[0])
    coeff_n = sp.simplify(arg.coeff(n))
    shift = sp.simplify(arg - coeff_n * n)
    if coeff_n != 1 or shift.has(n):
        return None

    coeff = sp.simplify(sp.Mul(*[f for f in factors if f != fn]))
    if coeff.has(n):
        return None

    formal_name = fn.func.__name__.upper()
    formal = sp.Function(formal_name)(z)

    if shift == 0:
        return sp.simplify(coeff * formal)

    if shift.is_integer is not True:
        return None

    if shift.is_negative is True:
        # Delay: y(n-k) -> z**(-k) Y(z), k > 0.
        return sp.simplify(coeff * z ** shift * formal)

    if shift.is_nonnegative is True:
        # Advance: y(n+k) -> z**k Y(z) - sum_{m=0}^{k-1} z**(k-m) y(m), k > 0.
        m = sp.symbols("m", integer=True, nonnegative=True)
        correction = sp.summation(z ** (shift - m) * fn.func(m), (m, 0, shift - 1))
        return sp.simplify(coeff * (z ** shift * formal - correction))

    return None


def z_transform(x, n=None, z=None, evaluate=True, noconds=True):
    """Return the unilateral Z-transform of a discrete-time sequence.

    The unilateral transform is:
    ``X(z) = sum(x[n] * z**(-n), (n, 0, oo))``.

    :param x:
        Input sequence representation. Supported values are a SymPy expression,
        callable, string expression, or a finite ``list``/``tuple``.
    :type x: sympy.Expr | callable | str | list | tuple
    :param n:
        Discrete-time index symbol. If omitted, a nonnegative integer symbol
        named ``n`` is created.
    :type n: sympy.Symbol | None
    :param z:
        Transform variable. If omitted, a symbol named ``z`` is created.
    :type z: sympy.Symbol | None
    :param evaluate:
        If ``True``, evaluate/simplify the transform. If ``False``, return an
        unevaluated summation.
    :type evaluate: bool
    :param noconds:
        If ``True`` (default), return only ``X(z)``. If ``False``, return
        ``(X(z), condition)``.
    :type noconds: bool
    :returns:
        The Z-transform expression, optionally with convergence/support
        condition.
    :rtype: sympy.Expr | tuple[sympy.Expr, sympy.Expr]
    """
    n = n or sp.symbols("n", integer=True, nonnegative=True)
    z = z or sp.symbols("z")

    def _pack(xz, cond):
        return xz if noconds else (xz, cond)

    if isinstance(x, (list, tuple)):
        xz = sum(sp.sympify(value) * z ** (-k) for k, value in enumerate(x))
        xz = sp.simplify(xz) if evaluate else xz
        return _pack(xz, sp.Ne(z, 0))

    xn = _to_sympy_expr(x, n, z)

    if not evaluate:
        xz = sp.Sum(xn * z ** (-n), (n, 0, sp.oo))
        return _pack(xz, sp.S.true)

    terms = sp.expand(xn).as_ordered_terms()
    transformed_terms = []
    conditions = []

    for term in terms:
        clean, lower, upper, ok = _extract_heaviside_bounds(term, n)

        if ok:
            # Evaluate KroneckerDelta terms independent of n.
            for d in list(clean.atoms(_SympyKroneckerDelta)):
                if not d.has(n):
                    clean = sp.simplify(clean.subs(d, sp.simplify(d.doit())))

            partial = _build_delta_closed_form(clean, n, z, lower, upper)
            if partial is None:
                partial = _build_trig_euler_closed_form(clean, n, z, lower, upper)
            if partial is None:
                partial = _build_undef_function_closed_form(clean, n, z, lower, upper)

            if partial is None:
                if upper == sp.oo:
                    partial = sp.Sum(clean * z ** (-n), (n, lower, sp.oo)).doit()
                else:
                    finite = sp.Sum(clean * z ** (-n), (n, lower, upper)).doit()
                    partial = sp.Piecewise((finite, sp.Le(lower, upper)), (0, True))
        else:
            partial = sp.Sum(term * z ** (-n), (n, 0, sp.oo)).doit()

        partial = sp.simplify(partial)
        partial, cond = _strip_trivial_piecewise(partial)
        transformed_terms.append(partial)

        if cond != sp.S.true:
            conditions.append(cond)

    xz = sp.simplify(sp.Add(*transformed_terms))
    cond = sp.simplify(sp.And(*conditions)) if conditions else sp.S.true

    return _pack(xz, cond)


def inverse_z_transform(F, z=None, n=None):
    """Compute the unilateral inverse Z-transform by residue summation.

    Parameters
    ----------
    F : sympy.Expr, callable, or str
        Z-domain expression.
    z : sympy.Symbol, optional
        Transform variable. Defaults to z.
    n : sympy.Symbol, optional
        Discrete-time index symbol. Defaults to integer nonnegative n.

    Returns
    -------
    sympy.Expr
        Time-domain sequence x[n].
    """
    z = z or sp.symbols("z")
    n = n or sp.symbols("n", integer=True, nonnegative=True)

    if callable(F):
        Fz = sp.sympify(F(z))
    elif isinstance(F, str):
        Fz = sp.sympify(
            F,
            locals={
                "KroneckerDelta": KroneckerDelta,
                "Heaviside": sp.Heaviside,
                str(n): n,
                str(z): z,
            },
        )
    else:
        Fz = sp.sympify(F)

    Fz = sp.simplify(sp.expand(Fz))
    den = sp.denom(sp.together(Fz))
    poles = list(sp.roots(den, z).keys())
    if not poles and not Fz.has(z):
        return sp.simplify(Fz * KroneckerDelta(n))

    residues = [sp.residue(Fz * z ** (n - 1), z, p) for p in poles]
    return sp.simplify(sp.Add(*residues))
