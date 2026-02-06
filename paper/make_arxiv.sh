#!/usr/bin/env bash
#
# Build a flat tarball of the paper for arXiv submission.
# Usage: cd paper && bash make_arxiv.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

echo "==> Staging files in $STAGING"

# --- Copy section files (flat) ---
for f in introduction background model numerics results scaling; do
    cp "sections/${f}.tex" "$STAGING/${f}.tex"
done

# --- Copy figures (flat) ---
cp figures/*.pdf "$STAGING/"

# --- Copy pre-compiled bibliography ---
cp main.bbl "$STAGING/"

# --- Copy and patch main.tex ---
sed \
    -e 's|\\input{sections/|\\input{|g' \
    -e '/^\\bibliographystyle{/d' \
    -e 's|^\\bibliography{references}|\\input{main.bbl}|' \
    main.tex > "$STAGING/main.tex"

# Add the arXiv 4-pass hint after \end{document}
printf '\\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}\n' \
    >> "$STAGING/main.tex"

# --- Patch results.tex: flatten figure paths ---
sed -i '' 's|\\includegraphics\(\[[^]]*\]\){figures/|\\includegraphics\1{|g' \
    "$STAGING/results.tex"

# --- Verify compilation ---
echo "==> Running pdflatex to verify compilation …"
(cd "$STAGING" && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1) || {
    echo "!! pdflatex failed.  Staging directory preserved at $STAGING"
    trap - EXIT  # keep the dir for debugging
    exit 1
}
echo "==> pdflatex succeeded"

# --- Remove pdflatex build artifacts before packaging ---
rm -f "$STAGING"/main.{pdf,aux,log,out} "$STAGING"/missfont.log

# --- Build tarball (flat, only the submission files) ---
OUTFILE="$SCRIPT_DIR/arxiv.tar.gz"
TAR_FILES=(
    main.tex main.bbl
    introduction.tex background.tex model.tex
    numerics.tex results.tex scaling.tex
)
# Add all PDFs
for pdf in "$STAGING"/*.pdf; do
    TAR_FILES+=("$(basename "$pdf")")
done

(cd "$STAGING" && tar czf "$OUTFILE" "${TAR_FILES[@]}")

echo "==> Created $OUTFILE"
echo "==> Contents:"
tar tzvf "$OUTFILE"
