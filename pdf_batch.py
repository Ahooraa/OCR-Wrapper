"""Small helpers for applying the existing PDF pipeline to several files."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PER_PDF = "per_pdf"
BY_TYPE = "by_type"
BATCH_LAYOUTS = (PER_PDF, BY_TYPE)

# Some Windows applications still fail near the legacy 260-character limit.
MAX_OUTPUT_PATH = 240


@dataclass(frozen=True)
class PDFJob:
    input_path: Path
    output_base: Path
    pagemap_path: Path


def validate_pdfs(paths: Iterable[Path]) -> list[Path]:
    """Validate and de-duplicate explicitly selected PDFs."""
    pdfs = []
    seen = set()
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        identity = path.resolve()
        if identity not in seen:
            seen.add(identity)
            pdfs.append(path)
    if not pdfs:
        raise ValueError("Select at least one PDF file")
    return pdfs


def create_jobs(paths: Iterable[Path], output_dir: Path, layout: str) -> list[PDFJob]:
    """Build short, collision-free paths for one of the two batch layouts."""
    if layout not in BATCH_LAYOUTS:
        raise ValueError(f"Unknown batch output layout: {layout}")

    output_dir = Path(output_dir).resolve()
    jobs = []
    used_names = set()
    for pdf in validate_pdfs(paths):
        name = _document_name(pdf, output_dir, layout, used_names)
        used_names.add(name.casefold())

        if layout == PER_PDF:
            output_base = output_dir / name / "transcript"
            pagemap_path = output_base.with_suffix(".pagemap.json")
        else:
            output_base = output_dir / "markdown" / name
            pagemap_path = output_dir / "pagemaps" / f"{name}.pagemap.json"
        jobs.append(PDFJob(pdf, output_base, pagemap_path))
    return jobs


def place_pagemap(job: PDFJob) -> None:
    """Move the original sidecar when the selected layout stores JSON separately."""
    generated = job.output_base.with_suffix(".pagemap.json")
    if generated == job.pagemap_path:
        return
    job.pagemap_path.parent.mkdir(parents=True, exist_ok=True)
    generated.replace(job.pagemap_path)


def _document_name(pdf: Path, output_dir: Path, layout: str, used: set[str]) -> str:
    suffix = "\\transcript.pagemap.json" if layout == PER_PDF else "\\pagemaps\\.pagemap.json"
    maximum = min(100, MAX_OUTPUT_PATH - len(str(output_dir)) - len(suffix) - 1)
    if maximum < 20:
        raise ValueError("Output folder path is too long; choose a shorter output folder")

    original = pdf.stem.rstrip(" .") or "document"
    candidate = original
    needs_hash = len(candidate) > maximum or candidate.casefold() in used
    if needs_hash:
        digest = hashlib.sha1(str(pdf.resolve()).encode("utf-8")).hexdigest()[:8]
        candidate = f"{original[:maximum - 9].rstrip()}-{digest}"
    return candidate
