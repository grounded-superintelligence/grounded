# Releasing the Python SDK

The package distribution is `grounded-sdk`; Python imports remain under
`grounded`.

## One-time setup

1. Create `testpypi` and `pypi` GitHub environments and require approval for
   the `pypi` environment.
2. Configure a Trusted Publisher for this repository and
   `.github/workflows/publish.yml` on TestPyPI and PyPI.
3. Confirm that the `grounded-sdk` project name is still available before the
   first release.

No long-lived PyPI token is required.

## Release

1. Update the version in `pyproject.toml`.
2. Run:

   ```bash
   python -m pip install ".[dev]"
   python -m pytest
   python -m ruff check src tests demo.py
   python -m build
   python -m twine check dist/*
   ```

3. Run the `Publish Python package` workflow against TestPyPI and verify a
   clean installation.
4. Publish a GitHub release for the same version. The protected `pypi`
   environment gates the production upload.

Wheel contents include the SDK package, not the repository's `tests/` folder.
