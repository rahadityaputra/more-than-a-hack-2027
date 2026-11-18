"""Compile the documentation listed in README.md's "Read these, in this order" into docs.pdf.

The reading order is parsed out of README.md (the only place that owns it), so re-ordering or
adding documents there is enough -- rerun this afterwards:

    uv run --no-project --with markdown --with xhtml2pdf python make_docs_pdf.py

docs.pdf is generated output and stays out of git (*.pdf in .gitignore). The dependencies come
from --with rather than pyproject.toml on purpose: they are a documentation-build concern, not
something the scorer or a submission needs.
"""

import io
import os
import re
import sys
from datetime import datetime, timezone

import markdown
from xhtml2pdf import pisa

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")
OUTPUT = os.path.join(HERE, "docs.pdf")

SECTION_START = "## Read these"
SECTION_END = "\n---"
ORDER_RE = re.compile(r"\*\*\[([A-Z][A-Z_.-]*\.md)\]\(")

GLYPH_FALLBACKS = (
    ("├─", "|- "),
    ("└─", "`- "),
    ("│", "|"),
    ("─", "-"),
    ("←", "<-"),
)


def fallback_glyphs(text):
    for glyph, ascii_form in GLYPH_FALLBACKS:
        text = text.replace(glyph, ascii_form)
    return text

CSS = """
@page { size: a4 portrait; margin: 2cm 1.8cm; }
body { font-family: Helvetica; font-size: 9.5pt; line-height: 1.5; color: #111111; }
h1 { font-size: 16pt; border-bottom: 1px solid #888888; padding-bottom: 4px; }
h2 { font-size: 12.5pt; margin-top: 14px; }
h3 { font-size: 10.5pt; margin-top: 12px; }
p  { margin: 5px 0; }
pre { font-family: Courier; font-size: 8pt; background-color: #f4f4f4;
      border: 0.5px solid #bbbbbb; padding: 6px; }
code { font-family: Courier; font-size: 8.5pt; }
table { width: 100%; }
th, td { border: 0.5px solid #999999; padding: 3px 5px; font-size: 8.5pt; text-align: left; }
.doc { page-break-before: always; }
.cover { font-size: 12pt; }
"""


def reading_order():
    """The .md files README.md says to read, in its own order."""
    with open(README, encoding="utf-8") as handle:
        text = handle.read()
    start = text.index(SECTION_START)
    section = text[start:text.index(SECTION_END, start)]
    matches = ORDER_RE.findall(section)
    if len(matches) < 2:
        sys.exit("make_docs_pdf: could not parse the reading order out of README.md")
    return matches


def html_for(docs):
    today = datetime.now(timezone.utc).date().isoformat()
    links = "".join('<a href="#doc{0}">[{0}] &nbsp; {1}</a><br/>'.format(i, name)
                    for i, name in enumerate(docs))
    cover = (
        '<div class="cover"><h1>More than a Hack 2027 &mdash; Phase 1: Ball Possession</h1>'
        "<p>Compiled documentation, in README.md's reading order.</p>"
        "<p>Generated {} &mdash; regenerate with "
        "<code>uv run --no-project --with markdown --with xhtml2pdf "
        "python make_docs_pdf.py</code></p>"
        "<p>{}</p></div>".format(today, links)
    )
    parts = [cover]
    for i, name in enumerate(docs):
        with open(os.path.join(HERE, name), encoding="utf-8") as handle:
            body = markdown.markdown(
                fallback_glyphs(handle.read()),
                extensions=["tables", "fenced_code", "sane_lists"])
        parts.append('<div class="doc" id="doc{0}"><a name="doc{0}"/>{1}</div>'.format(i, body))
    return ("<html><head><meta charset=\"utf-8\"/><style>{}</style></head><body>{}</body></html>"
            .format(CSS, "".join(parts)))


def main():
    docs = reading_order()
    print("reading order: " + " -> ".join(docs))
    html = html_for(docs)

    with open(OUTPUT, "wb") as handle:
        status = pisa.CreatePDF(io.BytesIO(html.encode("utf-8")), dest=handle)
    if status.err:
        sys.exit("make_docs_pdf: {} error(s) while rendering".format(status.err))
    size = os.path.getsize(OUTPUT)
    print("wrote {} ({:.0f} KiB)".format(os.path.relpath(OUTPUT, HERE), size / 1024))


if __name__ == "__main__":
    main()
