#!/usr/bin/env python3
"""
Build LogAnomalyML_Project_Overview.pdf from project_overview_pdf_source.txt
Run from repo root: python3 docs/generate_project_overview_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "project_overview_pdf_source.txt"
OUTPUT = ROOT / "LogAnomalyML_Project_Overview.pdf"


class DocPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Log Anomaly Detection - Project Overview", align="C")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def ascii_safe(text: str) -> str:
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2192", "->")
        .replace("\u2248", "~")
    )


def main() -> int:
    if not SOURCE.is_file():
        print(f"Missing source: {SOURCE}", file=sys.stderr)
        return 1

    pdf = DocPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 40, 80)
    title = "Log Anomaly Detection System"
    pdf.multi_cell(
        0, 10, ascii_safe(title), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(
        0,
        6,
        ascii_safe(
            "Technical overview for dissertation repository - BGL / LogHub, "
            "ML pipeline, Rails dashboard, and REST integration."
        ),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(8)

    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(2)
            continue
        line = ascii_safe(line)
        if line.startswith("===1==="):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(25, 55, 110)
            pdf.multi_cell(
                0, 8, line[7:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("===2==="):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(45, 85, 130)
            pdf.multi_cell(
                0, 7, line[7:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("===0==="):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(
                0, 5.5, line[7:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(OUTPUT))
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
