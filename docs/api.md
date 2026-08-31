# API Overview

The public API is available from both `ztrans` and `z_transform`.

## z_transform

Signature: `z_transform(x, n=None, z=None, evaluate=True, noconds=True)`

Return the unilateral Z-transform of a discrete-time sequence:

```text
X(z) = sum(x[n] * z**(-n), n = 0..oo)
```

Supported inputs include:

- SymPy expressions.
- String expressions.
- Callables that accept the sequence index.
- Finite lists or tuples.

Set `noconds=False` to return `(transform, condition)`.

## inverse_z_transform

Signature: `inverse_z_transform(F, z=None, n=None)`

Return the unilateral inverse Z-transform of a Z-domain expression. The current
implementation focuses on rational expressions and uses symbolic residues in the
Z-plane.

## z_correspondence

Signature: `z_correspondence(f, fdict)`

Replace formal transform placeholders using a function correspondence dictionary.
For example, `{y: Y}` maps `ZTransform(y(n), n, z)` to `Y(z)` and
`InverseZTransform(Y(z), z, n)` to `y(n)`.

## z_initial_conds

Signature: `z_initial_conds(f, n, fdict)`

Substitute declared discrete initial conditions. For example, `{y: [2, 4]}` maps
`y(0)` to `2` and `y(1)` to `4`.

## ZTransform

SymPy function placeholder for unevaluated unilateral Z-transforms.

## InverseZTransform

SymPy function placeholder for unevaluated unilateral inverse Z-transforms.

## KroneckerDelta

Signature: `KroneckerDelta(a, b=0)`

Convenience wrapper around SymPy's two-argument Kronecker delta with default
second argument `b=0`.
