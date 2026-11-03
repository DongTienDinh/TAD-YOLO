# TAD-YOLO IEEE paper

`main.tex` is an IEEE conference-format preliminary report written in academic
English. The paper reports only component-wise TAD-YOLO results that can be
attributed to the stated architecture.

## Build

Use a TeX distribution containing `IEEEtran`, `graphicx`, `booktabs`,
`amsmath`, `svg`, and BibTeX:

```bash
pdflatex --shell-escape main.tex
bibtex main
pdflatex --shell-escape main.tex
pdflatex --shell-escape main.tex
```

Run these commands from the `paper/` directory. The source is sized for at
least four IEEE conference pages, but the exact page count must be confirmed
with the target IEEE template/TeX distribution.

The architecture figure is loaded directly from
`assets/tad_yolo_architecture.svg`. Keep this relative path unchanged when
uploading the ZIP to Overleaf. The compact dataset chart is stored alongside
it as `assets/visdrone_dataset_distribution.svg`.

## Before submission

1. Verify the author order, romanized names, and FPT University affiliation.
2. Verify all six ablation rows against their exact validation artifacts; the
   repository does not currently contain matching artifacts for every row.
3. Repeat TAD-YOLO training with seeds 42 and 123 and report mean and standard
   deviation together with the existing seed-0 result.
4. Add per-class and object-scale AP if available.
5. Recompile and check page count, figure legibility, overfull boxes, and IEEE
   PDF compliance.
