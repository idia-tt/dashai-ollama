## What and why

<!-- What does this change, and why? -->

## Checklist

- [ ] The **PR title** follows [Conventional Commits](https://www.conventionalcommits.org/)
      (`feat(model): ...`, `fix: ...`). It becomes the commit message on a squash merge.
- [ ] `uv run pre-commit run --all-files` passes (ruff, ty, gitleaks, ...).
- [ ] Tests cover the change, and `uv run pytest` passes.
- [ ] If a parameter or default changed, the README tables and the descriptions in
      all five languages are updated.
