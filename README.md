# ztrans

Utilidades simbólicas para transformada Z unilateral en SymPy.

Incluye:
- `z_transform(...)` con opción `noconds=True` por omisión.
- `inverse_z_transform(...)` por suma de residuos en el plano-Z.
- `z_correspondence(...)` para reemplazo formal tipo `y(n) <-> Y(z)`.
- `z_initial_conds(...)` para sustituir condiciones iniciales discretas.
- soporte de entrada como `sympy.Expr`, `str`, `callable` y secuencias finitas.

## Requisitos

- Python 3.10+
- Dependencias en `requirements.txt`

## Instalación local

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Uso rápido

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

n, z, a = sp.symbols("n z a", integer=True, nonnegative=True)

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

## Tests

```bash
python -m pytest -q
```

## Estructura

```text
z_transform.py
tests/test_z_transform.py
requirements.txt
```
