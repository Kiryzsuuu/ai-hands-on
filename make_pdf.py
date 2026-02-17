"""Generate PDF book from Markdown/text using Matplotlib (no extra deps).

Why this approach:
- Works on Windows/Mac/Linux as long as Matplotlib is installed.
- Avoids requiring Pandoc/LaTeX or HTML-to-PDF native dependencies.

Usage (Windows PowerShell):
  & "./.venv/Scripts/python.exe" make_pdf.py

Output:
  BUKU_PANDUAN_AI.pdf
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent

# Source files are appended in this order (so the PDF feels like a cohesive module).
SOURCES = [
    ROOT / "BUKU_PANDUAN_AI.md",
    ROOT / "README.md",
    ROOT / "TUTORIAL.md",
    ROOT / "PANDUAN.md",
    ROOT / "GLOSSARY.md",
]

OUT_PDF = ROOT / "BUKU_PANDUAN_AI.pdf"


# --- text cleanup helpers ---

# Remove emojis / symbols that often don't exist in default PDF fonts.
# (Keeps Indonesian letters and punctuation.)
#
# Notes:
# - Some emoji-like glyphs are in BMP (e.g. ⭐ U+2B50), so we include 2B00-2BFF.
# - Keycap sequences use combining enclosing keycap U+20E3 (in 20D0-20FF).
# - Variation selectors and ZWJ can cause missing-glyph warnings too.
_EMOJI_RANGES = re.compile(
    "["
    "\U00010000-\U0010FFFF"  # non-BMP (most emojis)
    "\u2600-\u27BF"          # misc symbols
    "\u2B00-\u2BFF"          # misc symbols and arrows (includes ⭐)
    "\u20D0-\u20FF"          # combining diacritical marks for symbols (includes U+20E3)
    "\uFE00-\uFE0F"          # variation selectors
    "\u200D"                 # zero-width joiner
    "]",
    flags=re.UNICODE,
)


def _sanitize_line(line: str) -> str:
    # Normalize line endings and strip trailing whitespace
    line = line.rstrip("\n\r").rstrip()

    # Convert Windows path quotes in docs to something readable in PDF
    line = line.replace("\\t", "    ")

    # Remove emojis/symbols that often render as empty boxes
    line = _EMOJI_RANGES.sub("", line)

    # Replace fancy quotes that can render poorly in some fonts
    line = line.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Avoid markdown-only artifacts
    line = line.replace("**", "")
    line = line.replace("__", "")
    line = line.replace("`", "")

    return line


def _markdown_to_plain_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_code_block = False

    for raw in text.splitlines():
        line = raw

        # Handle fenced code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                lines.append("")
                lines.append("[CODE]")
            else:
                lines.append("[/CODE]")
                lines.append("")
            continue

        # Strip markdown headings
        if not in_code_block:
            line = re.sub(r"^#{1,6}\s+", "", line)

        # Convert markdown links [text](url) => text (url)
        if not in_code_block:
            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)

        # Keep bullets readable
        if not in_code_block:
            line = re.sub(r"^\s*[-*]\s+", "- ", line)

        # Keep blockquotes readable
        if not in_code_block and line.lstrip().startswith(">"):
            line = "  " + line.lstrip().lstrip(">").lstrip()

        # Slightly indent code block lines
        if in_code_block:
            line = "    " + line

        line = _sanitize_line(line)
        lines.append(line)

    return lines


def _wrap_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue

        # Preserve explicit indentation (esp. code blocks)
        indent_match = re.match(r"^(\s+)", line)
        indent = indent_match.group(1) if indent_match else ""
        content = line[len(indent) :]

        # Don't wrap short lines
        if len(line) <= width:
            wrapped.append(line)
            continue

        # Wrap while keeping indentation
        for i, part in enumerate(
            textwrap.wrap(
                content,
                width=max(10, width - len(indent)),
                break_long_words=False,
                break_on_hyphens=False,
            )
        ):
            wrapped.append((indent if i == 0 else indent) + part)

    return wrapped


def _read_sources() -> str:
    parts: list[str] = []
    for path in SOURCES:
        if not path.exists():
            continue
        parts.append(f"\n\n===== {path.name} =====\n")
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts).strip() + "\n"


def build_pdf() -> None:
    content = _read_sources()
    lines = _markdown_to_plain_lines(content)

    # Heuristic page layout
    # A4 portrait in inches: 8.27 x 11.69
    page_size = (8.27, 11.69)
    font_name = "DejaVu Sans"  # bundled with Matplotlib and supports Indonesian well
    font_size = 10

    # These are rough character/line limits tuned for font size 10.
    max_chars_per_line = 95
    max_lines_per_page = 55

    lines = _wrap_lines(lines, width=max_chars_per_line)

    with PdfPages(OUT_PDF) as pdf:
        page_num = 1
        idx = 0
        total_lines = len(lines)

        while idx < total_lines:
            fig = plt.figure(figsize=page_size)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            # Take a slice for this page
            page_lines = lines[idx : idx + max_lines_per_page]
            idx += max_lines_per_page

            # Render page text
            page_text = "\n".join(page_lines)
            fig.text(
                0.06,
                0.94,
                page_text,
                va="top",
                ha="left",
                family=font_name,
                fontsize=font_size,
            )

            # Footer
            fig.text(
                0.5,
                0.03,
                f"BUKU_PANDUAN_AI  |  Halaman {page_num}",
                ha="center",
                va="bottom",
                family=font_name,
                fontsize=9,
                color="#444444",
            )

            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1


def main() -> None:
    build_pdf()
    print(f"✅ PDF created: {OUT_PDF}")


if __name__ == "__main__":
    main()
