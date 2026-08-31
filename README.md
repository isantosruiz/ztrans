# ztrans

[![Tests](https://github.com/isantosruiz/ztrans/actions/workflows/tests.yml/badge.svg)](https://github.com/isantosruiz/ztrans/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`ztrans` provides symbolic utilities for unilateral Z-transforms in
[SymPy](https://www.sympy.org/). It is designed for exact algebraic derivations
in discrete-time signal processing, control systems, recurrence equations, and
teaching workflows.

The repository currently includes:

- `z_transform(...)` with `noconds=True` by default.
- `inverse_z_transform(...)` for rational Z-domain expressions by residue summation.
- `z_correspondence(...)` for formal replacement mappings such as `y(n) <-> Y(z)`.
- `z_initial_conds(...)` for substituting discrete initial conditions.
- Input support for `sympy.Expr`, `str`, `callable`, and finite sequences.
- Compatibility imports through both `from z_transform import ...` and `import ztrans`.

## Why this package?

SymPy provides a rich symbolic mathematics system, but it does not currently expose
a top-level Z-transform interface analogous to its Laplace transform utilities.
`ztrans` fills that focused gap with a small, inspectable implementation that keeps
all results as ordinary SymPy expressions.

## Installation

Install the development version directly from GitHub:

```bash
python -m pip install "git+https://github.com/isantosruiz/ztrans.git"
```

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/isantosruiz/ztrans.git
cd ztrans
python -m pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- SymPy 1.12 or later, below SymPy 2

## Quick Start

```python
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

z = sp.symbols("z")
a = sp.symbols("a", real=True)
n, k = sp.symbols("n k", integer=True, nonnegative=True)

F = z_transform(lambda k: a**k, n=n, z=z)
print(F)  # z/(z - a)

xn = inverse_z_transform(F, z=z, n=n)
print(xn)  # a**n

print(inverse_z_transform(1, z=z, n=n))  # KroneckerDelta(0, n)

y = sp.Function("y")
Y = sp.Function("Y")
formal = ZTransform(y(n), n, z) + InverseZTransform(Y(z), z, n)
print(z_correspondence(formal, {y: Y}))  # Y(z) + y(n)

expr = z_transform(y(n + 2), n=n, z=z)   # z**2*ZTransform(y(n), n, z) - z**2*y(0) - z*y(1)
expr = z_correspondence(expr, {y: Y})     # z**2*Y(z) - z**2*y(0) - z*y(1)
print(z_initial_conds(expr, n, {y: [2, 4]}))  # z**2*Y(z) - 2*z**2 - 4*z
```

## Mathematical Scope

The package implements the unilateral Z-transform convention

```text
X(z) = sum(x[n] * z**(-n), n = 0..oo).
```

Forward transforms are evaluated by linearity, finite sequence expansion,
Kronecker-delta collapse, Heaviside-derived support bounds, Euler rewriting for
trigonometric sequences, and formal shift rules for undefined functions such as
`y(n + 1)`.

Inverse transforms currently focus on rational expressions. The implementation
uses Cauchy's coefficient formula, residue summation over nonzero poles, a
separate Laurent principal-part treatment at `z = 0` for delayed impulse terms,
and a high-frequency correction for the `n = 0` sample when needed.

This is intentionally a compact symbolic package, not a complete transform table
or a full discrete-systems framework.

## Documentation

Documentation source files are available in [`docs/`](docs/). They include a
quickstart tutorial, API overview, development notes, and a short review-readiness
checklist. The documentation can be built locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build
```

## Tests

```bash
python -m pytest -q
```

Continuous integration runs the test suite on supported Python versions.

## Project Structure

```text
ztrans/__init__.py
z_transform.py
tests/test_z_transform.py
docs/
requirements.txt
pyproject.toml
```

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for
development setup, testing, documentation, and pull request guidelines. All
participants are expected to follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Citation

If you use `ztrans` in research or teaching materials, please cite the repository
metadata in [`CITATION.cff`](CITATION.cff). A DOI should be added after archiving a
tagged release on Zenodo or a similar long-term preservation service.

## License

`ztrans` is distributed under the MIT License. See [`LICENSE`](LICENSE).
