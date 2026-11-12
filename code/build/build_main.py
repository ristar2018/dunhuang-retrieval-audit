# -*- coding: utf-8 -*-
"""Build the paper_sivp manuscript into a sn-jnl two-column LaTeX manuscript
and compile it, so that the page count is measured rather than estimated.

Pipeline: tokenise figures and pipe tables -> pandoc (prose only) -> swap the
tokens back in as real LaTeX floats -> wrap in the sn-jnl preamble used by the
previous submission -> pdflatex x2 + bibtex.
"""
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = r"<PROJECT_ROOT>"
MD = os.path.join(ROOT, "paper_sivp", "正文_v1_英文.md")
PAPER = os.path.join(ROOT, "paper_sivp")
OUT = os.path.join(ROOT, "08_阶段0复核", "tex")
FIGSRC = os.path.join(ROOT, "08_阶段0复核", "figs")
FIGS = os.path.join(os.environ.get("TEMP", OUT), "msfigs")
PY = sys.executable

UNI = [
    ("\u2212", "\\ensuremath{-}"),
    ("\u2013", "--"),
    ("\u2014", "---"),
    ("\u2032", "\\ensuremath{{}^{\\prime}}"),
    ("\u2192", "\\ensuremath{\\rightarrow}"),
    ("\u03c1", "\\ensuremath{\\rho}"),
    ("\u0394", "\\ensuremath{\\Delta}"),
    ("\u00d7", "\\ensuremath{\\times}"),
    ("\u2248", "\\ensuremath{\\approx}"),
    ("\u00b0", "\\ensuremath{^{\\circ}}"),
    ("\u00b1", "\\ensuremath{\\pm}"),
    ("\u222a", "\\ensuremath{\\cup}"),
    ("\u00a7", "\\S{}"),
]
MATH = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$")


def esc_out(s, escape=True):
    """Convert unicode to LaTeX and escape specials, leaving math spans intact."""
    out, pos = [], 0
    for m in MATH.finditer(s):
        out.append(_plain(s[pos:m.start()], escape))
        out.append(m.group(0))
        pos = m.end()
    out.append(_plain(s[pos:], escape))
    return "".join(out)


def _plain(s, escape):
    for a, b in UNI:
        s = s.replace(a, b)
    if escape:
        s = s.replace("\\", "\x00")
        for ch in "%&#_":
            s = s.replace(ch, "\\" + ch)
        s = s.replace("\x00", "\\")
        s = s.replace("~", "\\textasciitilde{}")
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return s


MATHMAP = [
    ("\u2212", "-"),
    ("\u2013", "-"),
    ("\u2032", "{}^{\\prime}"),
    ("\u2192", "\\rightarrow "),
    ("\u03c1", "\\rho "),
    ("\u0394", "\\Delta "),
    ("\u00d7", "\\times "),
    ("\u2248", "\\approx "),
    ("\u00b0", "{}^{\\circ}"),
    ("\u00b1", "\\pm "),
    ("\u222a", "\\cup "),
]


def uni_pass(s):
    """Replace non-ASCII symbols with LaTeX, leaving existing escapes alone."""
    out, pos = [], 0
    for m in MATH.finditer(s):
        out.append(_plain(s[pos:m.start()], False))
        inner = m.group(0)
        for a, b in MATHMAP:
            inner = inner.replace(a, b)
        out.append(inner)
        pos = m.end()
    out.append(_plain(s[pos:], False))
    return "".join(out)


def unbold(s):
    s = s.replace("**", "")
    return s.strip()


def strip_prefix(s, kind):
    """'**Table 1. Caption title.** remainder' -> 'Caption title. remainder'."""
    s = s.strip()
    s = re.sub(r"^\*\*%s \d+\.\s*" % kind, "", s)
    s = re.sub(r"^\*\*%s \d+\.\*\*\s*" % kind, "", s)
    s = re.sub(r"^\*\*%s \d+\.\s*" % kind, "", s)
    s = re.sub(r"^\*\*\s*", "", s, count=1)
    s = s.strip()
    if "**" in s:
        s = s.replace("**", "", 1).strip()
    if s.endswith("**"):
        s = s[:-2].rstrip()
    return s


# ---------------------------------------------------------------- read source
with io.open(MD, encoding="utf-8") as f:
    raw = f.read()
raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
lines = raw.split("\n")

title = ""
start = 0
for i, l in enumerate(lines):
    if l.startswith("# "):
        title = l[2:].strip()
        start = i + 1
        break
if not title:
    print("FAIL: no title")
    sys.exit(2)

# abstract block
abs_start = None
for i in range(start, len(lines)):
    if lines[i].strip().lower() in ("## abstract", "# abstract"):
        abs_start = i + 1
        break
if abs_start is None:
    print("FAIL: no abstract")
    sys.exit(2)
abs_end = abs_start
while abs_end < len(lines) and not lines[abs_end].startswith("---"):
    abs_end += 1
abstract = " ".join(x.strip() for x in lines[abs_start:abs_end] if x.strip())
body = lines[abs_end + 1:]

# ------------------------------------------------- tokenise floats and tables
figs = []      # dicts: caption, path, width, wide
tables = []    # dicts: caption, header, align, rows
out = []
i = 0
while i < len(body):
    l = body[i]
    m = re.match(r"^!\[(.*)\]\((.*?)\)\{(.*?width=([0-9.]+)mm.*?)\}\s*$", l)
    if m:
        alt, path, _, width = m.group(1), m.group(2), m.group(3), float(m.group(4))
        figs.append(dict(caption=unbold(strip_prefix(alt, "Figure")),
                         path=os.path.basename(path), width=width))
        out.append("@@FIG%d@@" % (len(figs) - 1))
        i += 1
        continue
    if re.match(r"^\*\*Table \d+\.", l.strip()):
        cap = l.strip()
        j = i + 1
        while j < len(body) and not body[j].strip():
            j += 1
        if j < len(body) and body[j].lstrip().startswith("|"):
            rows = []
            while j < len(body) and body[j].lstrip().startswith("|"):
                rows.append(body[j].strip())
                j += 1
            tables.append(dict(caption=strip_prefix(cap, "Table"), rows=rows))
            out.append("@@TAB%d@@" % (len(tables) - 1))
            i = j
            continue
    # strip the manual section numbers; LaTeX numbers automatically
    if l.strip() == "---":
        out.append("")
        i += 1
        continue
    l = re.sub(r"^(## )\d+\.\s+", r"\1", l)
    l = re.sub(r"^(### )\d+\.\d+\s+", r"\1", l)
    out.append(l)
    i += 1

print("figures: %d  tables: %d" % (len(figs), len(tables)))
if len(figs) != 5 or len(tables) != 6:
    print("FAIL: expected 5 figures and 6 tables")
    sys.exit(2)


def split_row(row):
    row = row.strip()
    row = row[1:] if row.startswith("|") else row
    row = row[:-1] if row.endswith("|") else row
    return [c.strip() for c in row.split("|")]


def make_table(idx, t):
    rows = t["rows"]
    header = split_row(rows[0])
    sep = split_row(rows[1])
    data = [split_row(r) for r in rows[2:]]
    spec = ""
    for c in sep:
        c = c.strip()
        if c.endswith(":") and c.startswith(":"):
            spec += "c"
        elif c.endswith(":"):
            spec += "r"
        else:
            spec += "l"
    ncol = len(header)
    size = "\\scriptsize" if ncol >= 8 else "\\footnotesize"
    cap = esc_out(t["caption"])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{%s}" % cap,
        "\\label{tab:%d}" % (idx + 1),
        size,
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}%s@{}}" % spec,
        "\\toprule",
        " & ".join(esc_out(c) for c in header) + " \\\\",
        "\\midrule",
    ]
    for r in data:
        r = (r + [""] * ncol)[:ncol]
        lines.append(" & ".join(esc_out(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def make_figure(idx, f):
    wide = f["width"] > 120
    env = "figure*" if wide else "figure"
    width = "\\textwidth" if wide else "\\columnwidth"
    return "\n".join([
        "\\begin{%s}[t]" % env,
        "\\centering",
        "\\includegraphics[width=%s]{%s}" % (width, os.path.join(FIGS, f["path"]).replace("\\", "/")),
        "\\caption{%s}" % esc_out(f["caption"]),
        "\\label{fig:%d}" % (idx + 1),
        "\\end{%s}" % env,
    ])


# ------------------------------------------------------------- pandoc the body
if not os.path.isdir(OUT):
    os.makedirs(OUT)
if not os.path.isdir(FIGS):
    os.makedirs(FIGS)
for f in figs:
    src = os.path.join(FIGSRC, f["path"])
    if not os.path.exists(src):
        print("FAIL: missing figure %s" % src)
        sys.exit(2)
    shutil.copy2(src, os.path.join(FIGS, f["path"]))

tmp_md = os.path.join(OUT, "_body.md")
with io.open(tmp_md, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out))
tmp_tex = os.path.join(OUT, "_body.tex")
rc = subprocess.call(["pandoc", tmp_md, "-f", "markdown+tex_math_single_backslash", "-t", "latex",
                      "--shift-heading-level-by=-1", "--syntax-highlighting=none",
                      "-o", tmp_tex])
if rc != 0 or not os.path.exists(tmp_tex):
    print("FAIL: pandoc rc=%s" % rc)
    sys.exit(2)
with io.open(tmp_tex, encoding="utf-8") as f:
    bt = f.read()

for n, t in enumerate(tables):
    tok = "@@TAB%d@@" % n
    if tok not in bt:
        print("FAIL: token %s missing in pandoc output" % tok)
        sys.exit(2)
    bt = bt.replace(tok, make_table(n, t))
for n, f in enumerate(figs):
    tok = "@@FIG%d@@" % n
    if tok not in bt:
        print("FAIL: token %s missing in pandoc output" % tok)
        sys.exit(2)
    bt = bt.replace(tok, make_figure(n, f))
bt = uni_pass(bt)

PRE = r"""\documentclass[pdflatex,sn-mathphys-num,iicol]{sn-jnl}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{tabularx}
\usepackage{adjustbox}
\graphicspath{{./}}
\begin{document}
\title[Answer-Key Provenance and Margin Units]{%s}
\author*[1]{\fnm{Mengjie} \sur{Hu}}\email{hmj@hist.edu.cn}
\author[1]{\fnm{Yingtao} \sur{Ji}}
\author[2]{\fnm{Yuan} \sur{Zhao}}
\affil[1]{\orgdiv{School of Music and Dance}, \orgname{Henan Institute of Science and Technology}, \orgaddress{\city{Xinxiang}, \postcode{453003}, \country{China}}}
\affil[2]{\orgdiv{School of Foreign Languages}, \orgname{Henan Institute of Science and Technology}, \orgaddress{\city{Xinxiang}, \postcode{453003}, \country{China}}}
\abstract{%s}
\keywords{Metric learning, Answer-key provenance, Image retrieval, Cultural heritage, Dunhuang Mogao Caves, Angular margin, Benchmark audit}
\maketitle
""" % (esc_out(title), esc_out(abstract))

POST = r"""
\paragraph*{Acknowledgements}
The authors thank the maintainers of publicly accessible cultural heritage image repositories, including the Digital Dunhuang resource library and Wikimedia Commons. This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

\paragraph*{Ethical approval}
This study does not involve human participants or animals.

\paragraph*{Competing interests}
The authors declare no competing interests.

\paragraph*{Data availability}
The Dunhuang dataset is derived from publicly accessible cultural heritage resources. The curated label set, provenance records, and training scripts are available from the corresponding author on reasonable request.

\bibliography{references}
\end{document}
"""

main_tex = os.path.join(OUT, "main.tex")
with io.open(main_tex, "w", encoding="utf-8", newline="\n") as f:
    f.write(PRE + "\n" + bt + "\n" + POST)
print("wrote %s (%d chars)" % (main_tex, len(PRE) + len(bt) + len(POST)))