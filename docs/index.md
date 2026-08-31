# ztrans

`ztrans` provides symbolic utilities for unilateral Z-transforms in SymPy.

It is intended for exact derivations in discrete-time signal processing, control
systems, recurrence equations, and teaching material. The package focuses on a
small, inspectable set of algebraic operations rather than a complete transform
table.

## Main Features

- Forward unilateral Z-transforms for SymPy expressions, callables, strings, and
  finite sequences.
- Inverse Z-transforms for rational expressions using residue summation.
- Formal placeholders for unknown time-domain and Z-domain functions.
- Helpers for Z-domain correspondences and discrete initial conditions.
- Pure Python implementation built on SymPy.

## Installation

```bash
python -m pip install "git+https://github.com/isantosruiz/ztrans.git"
```

For development:

```bash
git clone https://github.com/isantosruiz/ztrans.git
cd ztrans
python -m pip install -e ".[dev]"
```

## Import

The recommended package import is:

```python
import ztrans
```

The original module import remains available:

```python
from z_transform import z_transform, inverse_z_transform
```
