# Development

## Local Setup

```bash
git clone https://github.com/isantosruiz/ztrans.git
cd ztrans
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest -q
```

## Linting

```bash
ruff check .
```

## Documentation

```bash
python -m pip install -e ".[docs]"
mkdocs build
```

## Release Checklist

Before publishing a release:

1. Update `CHANGELOG.md`.
2. Update the version in `pyproject.toml` and `CITATION.cff`.
3. Create a GitHub release with a matching tag.
4. Archive the release with Zenodo or an equivalent preservation service.
5. Add the DOI to `CITATION.cff`, the README, and any software paper metadata.
