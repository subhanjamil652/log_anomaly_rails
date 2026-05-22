#!/usr/bin/env python3
"""
Build Project_Five_Questions_Answers.pdf from project_questions_pdf_source.txt

Run from repo root:
  pip install fpdf2
  python3 docs/generate_project_questions_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    print("Install: pip install fpdf2", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "project_questions_pdf_source.txt"
OUTPUT = ROOT / "Project_Five_Questions_Answers.pdf"

STUDENT_NAME = "Subhan Jameel"
STUDENT_ID = "U2926092"
INSTITUTION = "University of East London"
MODULE = "CN7000 - MSc Project Dissertation"
PROGRAMME = "MSc Artificial Intelligence"


class DocPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "BGL Log Anomaly Detection - Project Q&A", align="C")
        self.ln(10)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no() - 1}", align="C")


def ascii_safe(text: str) -> str:
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2192", "->")
        .replace("\u2248", "~")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def draw_cover(pdf: FPDF) -> None:
    pdf.set_y(55)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(
        0,
        7,
        ascii_safe(INSTITUTION),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 40, 80)
    pdf.multi_cell(
        0,
        9,
        ascii_safe(
            "Log anomaly detection on BGL\n"
            "Five project questions (detailed answers)"
        ),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(18)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(
        0,
        7,
        ascii_safe(f"Student name: {STUDENT_NAME}"),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.multi_cell(
        0,
        7,
        ascii_safe(f"Student ID: {STUDENT_ID}"),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(45, 45, 45)
    pdf.multi_cell(
        0,
        6.5,
        ascii_safe(PROGRAMME),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.multi_cell(
        0,
        6.5,
        ascii_safe(MODULE),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_y(-40)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0,
        5,
        ascii_safe("Repository: ml_pipeline + log_anomaly_rails"),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )


def main() -> int:
    if not SOURCE.is_file():
        print(f"Missing source: {SOURCE}", file=sys.stderr)
        return 1

    pdf = DocPDF()
    pdf.add_page()
    draw_cover(pdf)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(
        0,
        5.5,
        ascii_safe(
            "The following sections answer five questions about the dissertation project, "
            "based on the implemented codebase (training pipeline, Flask API, Rails dashboard)."
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(4)

    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(2)
            continue
        line = ascii_safe(line)
        if line.startswith("===1==="):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(25, 55, 110)
            pdf.multi_cell(
                0, 8, line[7:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("===2==="):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(45, 85, 130)
            pdf.multi_cell(
                0, 6.5, line[7:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT
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
