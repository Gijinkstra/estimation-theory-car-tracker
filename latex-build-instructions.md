## The following info was produced using Claude to collaborate on LaTeX reports

- May have to install a markdown previewer for VSCode to view this document properly. Usually previewed using `Ctrl + Shift + V`.
- Perl will likely need to be installed to run latexmk in the command line for Windows. I can do this, but if you want it, it can be done with `wingit install StrawberryPerl.StrawberryPerl` in Powershell.
	- Check this by running `perl --version` followed by `latexmk --version`.
- When compiling in the terminal, run `latexmk -pdf report.tex`.

## LaTeX Project Layout for Git Collaboration

### Recommended Directory Structure

```
project/
│
├── .gitignore
├── README.md
├── Makefile                  # or latexmkrc
│
├── main.tex                  # root document
├── preamble.sty              # shared custom styles/packages
│
├── chapters/                 # or sections/
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_methodology.tex
│   └── 04_conclusion.tex
│
├── figures/
│   ├── raw/                  # original source files (svg, py, R scripts)
│   └── output/               # generated figures (tracked .pdf/.png)
│
├── tables/
│   └── results.tex
│
├── bibliography/
│   └── references.bib
│
└── build/                    # gitignored compiled output
```

---

### `main.tex` Structure

```latex
\documentclass[12pt, a4paper]{article}
\usepackage{preamble}

\title{Project Title}
\author{Author One \and Author Two}
\date{\today}

\begin{document}

\maketitle
\tableofcontents

\input{chapters/01_introduction}
\input{chapters/02_background}
\input{chapters/03_methodology}
\input{chapters/04_conclusion}

\bibliographystyle{plain}
\bibliography{bibliography/references}

\end{document}
```

---

### `Makefile` for Easy Building

```makefile
MAIN = main
LATEX = pdflatex
BIBTEX = biber

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.log *.bbl *.bcf *.blg *.toc *.out *.run.xml
	rm -f chapters/*.aux tables/*.aux

cleanall: clean
	rm -f $(MAIN).pdf

.PHONY: all clean cleanall
```

Or use **latexmk** (recommended):
```bash
# .latexmkrc
$pdf_mode = 1;
$out_dir = 'build';
$clean_ext = 'aux bbl bcf blg idx ilg ind lof lot out run.xml toc acn acr alg glg glo gls ist fls fdb_latexmk synctex.gz';
```

```bash
latexmk -pdf main.tex       # build
latexmk -C                  # clean all
```

---

### `.gitignore` for This Layout

```gitignore
# Build output
build/
*.pdf
*.dvi

# Aux files
*.aux
*.log
*.bbl
*.bcf
*.blg
*.toc
*.out
*.fls
*.gz
*.run.xml
*.fdb_latexmk

# Keep these PDFs if figures are pre-compiled
!figures/output/*.pdf
!figures/output/*.png
```

---

### Git Collaboration Tips

**Branch strategy:**
```bash
main          # stable, compiled version
dev           # integration branch
chapter/intro # per-chapter branches (one author per branch)
```

**Reducing merge conflicts:**
```latex
% Write one sentence per line — Git diffs line by line
This is the first sentence.
This is the second sentence, which is long and wraps
but should still be on one line for clean diffs.
```

**Useful `.gitattributes`** to improve diffs:
```gitattributes
*.tex   diff=tex
*.bib   diff=bibtex
*.sty   diff=tex
```

Enable in git config:
```bash
git config diff.tex.xfuncname "^(\\\\(sub)*section\\*?\\{.*)$"
```

---

### Quick Init Script

```bash
#!/bin/bash
mkdir -p project/{chapters,figures/{raw,output},tables,bibliography,build}
cd project
touch main.tex preamble.sty Makefile README.md .gitignore
touch chapters/{01_introduction,02_background,03_methodology,04_conclusion}.tex
touch bibliography/references.bib
git init
git add .
git commit -m "Initial LaTeX project structure"
```

Run with:
```bash
bash setup.sh
```