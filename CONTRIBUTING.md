# Contributing to ztrans

Thank you for your interest in improving `ztrans`. This project aims to provide a
small, readable, and well-tested symbolic Z-transform utility for SymPy users.

## Development Setup

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/isantosruiz/ztrans.git
cd ztrans
python -m pip install -e ".[dev]"
```

## Running Tests

Run the test suite before opening a pull request:

```bash
python -m pytest -q
```

Run lint checks with:

```bash
ruff check .
```

## Documentation

Documentation source files live in `docs/`. Build them locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build
```

When adding public functions, please update:

- Docstrings in `z_transform.py`.
- The API overview in `docs/api.md`.
- Tests that cover the expected behavior and edge cases.
- README examples if the feature changes the user-facing workflow.

## Pull Request Guidelines

Please keep changes focused and include:

- A short description of the problem and solution.
- Tests for new behavior or bug fixes.
- Documentation updates for user-facing changes.
- Notes about any mathematical assumptions or limitations.

The project favors exact symbolic behavior and interoperability with ordinary
SymPy expressions over package-specific object models.

## Reporting Issues

When reporting a bug, include:

- Your Python and SymPy versions.
- The expression or sequence you transformed.
- The expected result and the actual result.
- A minimal reproducible example.

## Generative AI Disclosure

pyOpenSci asks authors to disclose the use of generative AI tools in the
development or maintenance of submitted packages. If you contribute code,
documentation, tests, or issue responses with material assistance from such tools,
please disclose that assistance in the pull request or issue discussion.
