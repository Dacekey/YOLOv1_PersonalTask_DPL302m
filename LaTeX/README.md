# YOLO Personal Task Report

The submission PDF is `main.pdf`. Edit `main.tex` and `references.bib`; the report retains the original English language and author information.

YOLOv1 is the primary subject. The current main discussion retains an approximately 80% / 20% prose balance between YOLOv1-focused content and subsequent-version context. The appendix and bibliography are excluded from that balance. This is a prose-word estimate, not a claim that exactly 80% of pages or figures describe v1.

## Course requirement coverage

| Requirement in `../Subject_Requirement.txt` | Report location |
| --- | --- |
| Read the original paper | Original-paper citations throughout Sections 1–7 |
| Problem motivation | Section 2: Background and Motivation |
| Main idea of the model | Section 3.1–3.2: grid responsibility and prediction tensor |
| Architecture diagram | Section 3.3: explanation and table retained; Figure 3 is reserved for a later illustration |
| Key innovations compared to previous models | Section 4: R-CNN / Fast R-CNN / YOLOv1 comparison |
| Advantages and limitations | Section 5, including paper results and localization analysis |
| Typical applications | Section 6, scoped to original-model capabilities |
| Simple demo / implementation example | Section 7 is intentionally pending; space for real-image inference results is reserved |

Section 8 contains concise evolution context; Appendix A catalogues v1–v12, first appearance, authors/organizations, critical improvements, and task emphasis. Historical preprint and conference years are distinguished. Sources for v5/v8/YOLO11 are official software references.

## Rebuild

From this directory, run:

```bash
bash scripts/build_report.sh
```

The script generates the five active schematics, compiles with PDFLaTeX/BibTeX, and checks references, assets, the two reserved figures, pending Section 7, and the approximate scope ratio. It does not run a demo. It requires Python with NumPy/Matplotlib, `rg`, and a TeX Live installation containing the packages used in `main.tex`.

To compile without regenerating the assets:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Figure status and provenance

The revision follows `../notes/report-notes.txt`: the architecture illustration and inference-image example remain reserved, and Section 7 is left for implementation work when supporting materials are available. The report does not contain a synthetic demonstration, experiment setup, measured demo outputs, or claims that implementation is complete.

`scripts/generate_figures.py` creates five active explanatory schematics as vector PDFs with PNG previews. The architecture explanation and high-level table remain in Section 3.3, but its generated architecture image is not included in the report.

| Figure | Current status |
| --- | --- |
| Pipeline comparison | Included: explanatory schematic |
| Grid responsibility | Included: explanatory schematic |
| YOLOv1 architecture | Reserved space; illustration to be supplied later |
| Loss assignment | Included: explanatory schematic |
| Inference pipeline | Included: algorithmic schematic, not an inferred photograph |
| Real-image inference example | Reserved space in pending Section 7 |
| Evolution overview | Included: supporting schematic |

Previous synthetic-demo files and the generated architecture asset are retained on disk but excluded from the report and build workflow. They do not represent a completed implementation for the current report.

The approximate prose count can be reviewed with `python scripts/check_report.py`. The counting procedure excludes reserved figure captions, the appendix, and references. The report remains a work in progress: architecture imagery and the implementation requirement await later materials.
