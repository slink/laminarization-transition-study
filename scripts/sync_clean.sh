#!/usr/bin/env bash
#
# sync_clean.sh — Mirror fluid-flow/ → laminarization-transition-study/
# as a single-commit clean snapshot (no dev artifacts).
#
# Usage:
#   ./scripts/sync_clean.sh          # dry-run (shows what would happen)
#   ./scripts/sync_clean.sh --go     # execute for real

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$(cd "$SRC/../laminarization-transition-study" 2>/dev/null && pwd)" || {
    echo "ERROR: target repo not found at ../laminarization-transition-study" >&2
    exit 1
}

DRY_RUN=true
if [[ "${1:-}" == "--go" ]]; then
    DRY_RUN=false
fi

# ── Rsync exclusions ──────────────────────────────────────────────
EXCLUDES=(
    # Dotfiles/dirs (covers .claude/, .venv/, .pytest_cache/, .DS_Store, .worktrees/)
    '.*'
    # Claude content
    'CLAUDE.md'
    # Docs/plans
    'docs/'
    'RESEARCH_README.md'
    # Python cache
    '__pycache__/'
    '*.pyc'
    '*.egg-info'
    # LaTeX artifacts
    '*.aux'
    '*.bbl'
    '*.blg'
    '*.log'
    '*.out'
    '*.fls'
    '*.fdb_latexmk'
    '*.synctex.gz'
    '*Notes.bib'
    'aipsamp.tex'
)

RSYNC_ARGS=(-av --delete)
for pattern in "${EXCLUDES[@]}"; do
    RSYNC_ARGS+=(--exclude="$pattern")
done
# Always protect target's .git/
RSYNC_ARGS+=(--exclude='.git/')

# ── Dry-run or execute ────────────────────────────────────────────
if $DRY_RUN; then
    echo "=== DRY RUN (pass --go to execute) ==="
    echo ""
    echo "Source: $SRC/"
    echo "Target: $TARGET/"
    echo ""
    rsync "${RSYNC_ARGS[@]}" --dry-run "$SRC/" "$TARGET/"
    exit 0
fi

echo "=== Syncing $SRC/ → $TARGET/ ==="
rsync "${RSYNC_ARGS[@]}" "$SRC/" "$TARGET/"

# ── Single-commit reset ──────────────────────────────────────────
echo ""
echo "=== Resetting target to single commit ==="
cd "$TARGET"
git checkout --orphan temp_sync
git add -A
git commit -m "Laminarization transition in sediment-laden oscillatory boundary layers"
git branch -M main

echo ""
echo "=== Done ==="
echo "Target repo has $(git rev-list --count HEAD) commit(s)."
echo ""
echo "Remote:"
git remote -v
echo ""
echo "To publish: cd $TARGET && git push --force origin main"
