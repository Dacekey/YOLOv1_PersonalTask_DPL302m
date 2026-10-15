#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

python scripts/generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/yolo-report-build-1.log
bibtex main > /tmp/yolo-report-bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/yolo-report-build-2.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/yolo-report-build-3.log
if rg -n 'LaTeX Warning: (Citation|Reference)|undefined references|Overfull|Fatal error' main.log; then
    echo 'Build requires review; see main.log.'
    exit 1
fi
python scripts/check_report.py
echo 'Built main.pdf; citations and figures are up to date; architecture and implementation spaces remain reserved.'
