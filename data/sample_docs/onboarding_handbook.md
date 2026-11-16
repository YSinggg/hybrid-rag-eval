# Engineering Onboarding & Deployment Policy

## 1. First-Week Setup

New engineers receive repository access on Day 1 and complete their local environment setup using `uv` and Docker Compose. Every service repository includes a `Makefile` with standard targets: `make test`, `make lint`, and `make run`.

During Week 1, new hires pair with an onboarding buddy to ship at least one small production pull request (typically a bug fix, metric, or documentation improvement).

## 2. Pull Request & CI Gates

Direct pushes to `main` are blocked for all engineers. Every pull request requires:

- Passing unit and integration tests (`pytest`) with at least **80% line coverage** on modified modules.
- Clean linting and formatting checks (`ruff check` and `ruff format --check`).
- At least **one approving review** from a code owner listed in `.github/CODEOWNERS`.

## 3. Progressive Rollout & On-Call Readiness

Merged changes deploy automatically to `staging`, then progress to `production` via canary rollout: **5% traffic for 15 minutes**, followed by **50% for 15 minutes**, and finally **100%** if error rate remains below `0.2%`.

Engineers do not join the primary on-call rotation until they have completed **30 days** at the company and shadowed at least **two live incident shifts** (or simulated game-day drills).
