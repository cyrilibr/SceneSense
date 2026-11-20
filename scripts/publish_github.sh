#!/usr/bin/env bash
# Publish this existing local Git repository after GitHub CLI login.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required: https://cli.github.com/" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Sign in once using: gh auth login" >&2
  exit 1
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo "An origin remote is already configured. No repository was created." >&2
  exit 1
fi
gh repo create SceneSense-A4 --private --source=. --remote=origin --push
printf '\nPublished successfully. GitHub repository URL:\n'
gh repo view --json url --jq .url
