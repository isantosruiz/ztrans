# Quickstart

This tutorial shows the core workflow for computing symbolic unilateral
Z-transforms with `ztrans`.

## A Geometric Sequence

```python
import sympy as sp
import ztrans

z = sp.symbols("z")
a = sp.symbols("a", real=True)
n = sp.symbols("n", integer=True, nonnegative=True)

F = ztrans.z_transform(lambda k: a**k, n=n, z=z)
print(F)
```

The output is:

```text
z/(z - a)
```

The inverse transform recovers the original sequence:

```python
xn = ztrans.inverse_z_transform(F, z=z, n=n)
print(xn)
```

```text
a**n
```

## Finite Sequences

```python
ztrans.z_transform([1, 2, 3], n=n, z=z)
```

```text
1 + 2/z + 3/z**2
```

## Initial Conditions

```python
y = sp.Function("y")
Y = sp.Function("Y")

expr = ztrans.z_transform(y(n + 2), n=n, z=z)
expr = ztrans.z_correspondence(expr, {y: Y})
ztrans.z_initial_conds(expr, n, {y: [2, 4]})
```

```text
z**2*Y(z) - 2*z**2 - 4*z
```

## Conditions

By default, `z_transform` returns only the transformed expression. Use
`noconds=False` to request the symbolic convergence or support condition:

```python
ztrans.z_transform(lambda k: a**k, n=n, z=z, noconds=False)
```
