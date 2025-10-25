# Merge Conflict Resolution Guide

This guide walks through resolving conflicts when rebasing or merging the
`codex/improve-signal-strategy-for-paper-trading` branch with `main`.
It explains what to keep in each conflicted file and why.

## 1. codex_strategy.py

1. Start from the version on the strategy branch. It already contains the
   timeframe auto-switching helpers (`_initialize_timeframes`, `_set_interval`,
   `_pnl_for_interval`, `maybe_switch_timeframe`), adaptive strictness, and the
   enhanced risk/diagnostics logic. These must be preserved.
2. Inspect the `main`-branch changes and manually fold in any critical hotfixes
   (for example new bug guards, renamed constants, or dependency updates).
   Apply them in the corresponding sections of the strategy file while keeping
   the new helper functions intact.
3. Verify the resulting file still:
   - Reads Telegram credentials from environment variables.
   - Uses `datetime.now(timezone.utc)` for timestamps.
   - Calls `maybe_switch_timeframe()` once at the end of each candle cycle.
   - Writes `interval` to every trade and diagnostic record.

## 2. README.md

1. Keep all documentation for adaptive strictness and timeframe auto-switching.
2. Merge in any new sections from `main` (such as setup or troubleshooting
   additions) without deleting the new flag descriptions
   (`--diag_rate_limit_candles`, `--enable_session_guard`,
   `--cooldown_after_sl_mult`, `--atr_regime_shift`, auto-switch env vars).
3. Ensure the examples mention automatic `.env` loading and CSV rotation.

## 3. .env.example

1. Retain every key needed by the strategy: Telegram credentials, relax-mode
   toggles, and the timeframe auto-switch variables.
2. If `main` added new settings, append them while keeping the comments that
   explain defaults.

## 4. tests/smoke_strategy.py

1. Keep the auto-switch escalation/step-down tests, relax-mode assertions, and
   interval-tagging checks.
2. Reapply any new fixtures or helpers from `main` if they exist. Make sure
   `send_telegram` remains stubbed during tests.

## 5. .github/workflows/ci.yml

1. Preserve the workflow that runs `python -m compileall .` and
   `python tests/smoke_strategy.py`.
2. Merge any additional CI jobs from `main` by adding steps, not replacing the
   existing compile/test coverage.

## 6. requirements.txt

1. Take the union of dependencies. If both sides specify a package, keep the
   higher compatible version.

## 7. delivery helpers

1. Keep `delivery/create_update_archive.py`, `delivery/extract_update_archive.py`,
   and `delivery/USAGE.md`. If `main` does not have these files, simply retain
   the branch versions.

## 8. General workflow

1. After resolving conflicts, run:
   ```bash
   python -m compileall .
   python tests/smoke_strategy.py
   ```
2. Review `git diff` for leftover conflict markers.
3. Commit with a message such as
   `Resolve merge conflicts: keep strategy upgrades, integrate main hotfixes`.
4. Push and rerun CI before merging the PR.

## 9. Troubleshooting

- If a conflict reappears, use `git merge --abort` and restart with a clean
  working tree.
- Use `git checkout --ours <file>` or `git checkout --theirs <file>` only when
  certain which side to favor; otherwise resolve manually.

This guide should provide the missing context so you can resolve each file
confidently without losing the upgraded strategy features.
