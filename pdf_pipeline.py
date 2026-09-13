"""Run the existing single-PDF pipeline with caller-provided settings."""

import itertools
import tempfile
from pathlib import Path

from model import write_inspector_transcript
from ocr import run_ocr_pages
from pages import get_pdf_images


def process_pdf(
    pdf_path,
    output_base,
    formats,
    direction,
    engine,
    *,
    transcribe=None,
    dpi=300,
    limit=None,
    workers=1,
    log=print,
    progress=None,
    should_stop=lambda: False,
):
    """Process one PDF exactly as the original CLI/GUI pipeline does."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if engine == "inspector":
        write_inspector_transcript(pdf_path, output_base, formats, direction, log)
        return
    if transcribe is None:
        raise ValueError("An OCR transcriber is required for this engine")

    with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as temp_dir:
        pages, total = get_pdf_images(pdf_path, Path(temp_dir), dpi)
        if limit:
            pages = itertools.islice(pages, limit)
            total = min(total, limit)
        log(f"[INFO] Found {total} pages to process.")

        if engine == "bina" and workers > 1:
            log("[INFO] bina engine is single-device - ignoring workers")
            workers = 1
        run_ocr_pages(
            transcribe,
            pages,
            output_base,
            formats,
            direction=direction,
            total=total,
            workers=workers,
            parallel_mode="process" if engine == "chrome" else "thread",
            parallel_engine=engine,
            log=log,
            progress=progress,
            should_stop=should_stop,
        )
