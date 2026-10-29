#!/usr/bin/env bash
set -e

REPO_URL="${1:-}"
BRANCH="${2:-main}"

if [ ! -d ".git" ]; then
    git init
    git branch -M "$BRANCH"
fi

git add .
git commit -m "feat: CSMSS Kanchanwadi Civic Monitor v2.0 (Tasks 1-5 Complete)" || true

if [ -n "$REPO_URL" ]; then
    if git remote | grep -q "^origin$"; then
        git remote set-url origin "$REPO_URL"
    else
        git remote add origin "$REPO_URL"
    fi
    git push -u origin "$BRANCH"
    echo "✅ Successfully pushed to $REPO_URL ($BRANCH)"
else
    echo "ℹ️ Local git commit ready. Run: ./push_to_github.sh https://github.com/<username>/<repo>.git"
fi
