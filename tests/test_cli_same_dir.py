"""CLI tests for --same_dir (write outputs next to the input)."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import fitz

    import book_ocr_batch
except ImportError as error:  # pragma: no cover - depends on the environment
    book_ocr_batch = None
    SKIP_REASON = f"CLI test dependencies unavailable: {error}"
else:
    SKIP_REASON = ""


@unittest.skipIf(book_ocr_batch is None, SKIP_REASON)
class SameDirCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _text_pdf(self, name="paper.pdf"):
        """A small PDF with a real text layer (pdf-inspector rejects empty ones)."""
        path = self.root / name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "OCR wrapper test page.")
        doc.save(path)
        doc.close()
        return path

    def _run_cli(self, *args):
        argv = ["book_ocr_batch.py", *args]
        # The pipeline logs to stdout; keep the test output readable.
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return book_ocr_batch.main()

    def test_same_dir_writes_beside_the_pdf(self):
        pdf = self._text_pdf()

        code = self._run_cli(
            "--pdf", str(pdf), "--engine", "inspector", "--direction", "ltr",
            "--output_file", "transcript", "--same_dir",
        )

        self.assertEqual(code, 0)
        self.assertTrue((self.root / "transcript.md").is_file())
        # Without the flag a bare --output_file would land in the working dir.
        self.assertFalse((Path.cwd() / "transcript.md").exists())

    def test_same_dir_points_the_pagemap_at_the_same_directory(self):
        pdf = self._text_pdf()

        self._run_cli(
            "--pdf", str(pdf), "--engine", "inspector", "--direction", "ltr",
            "--output_file", "transcript", "--same_dir",
        )

        self.assertTrue((self.root / "transcript.pagemap.json").is_file())

    def test_same_dir_is_rejected_for_multiple_pdfs(self):
        pdf = self._text_pdf()

        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                self._run_cli("--pdfs", str(pdf), "--same_dir")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--same_dir is not supported with --pdfs", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
