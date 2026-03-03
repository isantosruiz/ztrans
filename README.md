# ztrans

Utilidades simbólicas para transformada Z unilateral en SymPy.

Incluye:
- `z_transform(...)` con opción `noconds=True` por omisión.
- `inverse_z_transform(...)` por suma de residuos en el plano-Z.
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
from z_transform import z_transform, inverse_z_transform, KroneckerDelta

n, z, a = sp.symbols("n z a", integer=True, nonnegative=True)

F = z_transform(lambda k: a**k, n=n, z=z)
print(F)  # z/(z - a)

xn = inverse_z_transform(F, z=z, n=n)
print(xn)  # a**n

print(inverse_z_transform(1, z=z, n=n))  # KroneckerDelta(0, n)
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
