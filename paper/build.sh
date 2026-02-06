#!/usr/bin/env bash

# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

set -euo pipefail

cd "$(dirname "$0")"

build_doc() {
    local name="$1"
    echo "Building ${name}.tex..."
    pdflatex -interaction=nonstopmode "${name}.tex"
    bibtex "${name}"
    pdflatex -interaction=nonstopmode "${name}.tex"
    pdflatex -interaction=nonstopmode "${name}.tex"
    echo "Complete: paper/${name}.pdf"
}

case "${1:-all}" in
    article)
        build_doc main
        ;;
    aip)
        build_doc main-aip
        ;;
    all)
        build_doc main
        echo ""
        build_doc main-aip
        ;;
    *)
        echo "Usage: $0 [article|aip|all]"
        echo "  article - Build standard article format (main.pdf)"
        echo "  aip     - Build AIP/revtex format (main-aip.pdf)"
        echo "  all     - Build both versions (default)"
        exit 1
        ;;
esac
