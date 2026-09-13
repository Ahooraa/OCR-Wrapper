"""Headless smoke tests for the PySide6 GUI.

The GUI is driven offscreen through a real Qt event loop: the OCR worker still
runs on its own thread and reports back through queued signals, so these tests
exercise the same signal round-trip the desktop app relies on.

Only the pdf-inspector engine is covered - it needs no model download and no
Windows-only DLL. Text direction is LTR because the RTL path reshapes
Arabic-script runs with arabic-reshaper, which is an optional extra.
"""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Must be set before Qt is imported so no display/plugin is needed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import fitz
    import pdf_inspector  # noqa: F401  (pdf-inspector engine, exercised below)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    import gui
except ImportError as error:  # pragma: no cover - depends on the environment
    QApplication = None
    QMessageBox = None
    gui = None
    SKIP_REASON = f"GUI test dependencies unavailable: {error}"
else:
    SKIP_REASON = ""


@unittest.skipIf(gui is None, SKIP_REASON)
class QtGuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One QApplication per process; widgets need it to exist first.
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.window = gui.OCRApp()
        # LTR keeps arabic-reshaper out of the picture (optional extra).
        self.window.direction_buttons["ltr"].setChecked(True)
        self.window.engine_buttons["inspector"].setChecked(True)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    # --- helpers ---

    def _text_pdf(self, name="paper.pdf", directory=None):
        """A small PDF with a real text layer."""
        path = Path(directory or self.root) / name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "OCR wrapper test page.")
        doc.save(path)
        doc.close()
        return path

    def _textless_pdf(self, name="scanned.pdf"):
        """A PDF with no text layer - pdf-inspector rejects it."""
        path = self.root / name
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(72, 72, 300, 200), color=(0, 0, 0), width=2)
        doc.save(path)
        doc.close()
        return path

    def _select_single_pdf(self, pdf, output_base, in_folder=False):
        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(pdf))
        self.window.output_file.setText(str(output_base))
        self.window.folder_check.setChecked(in_folder)
        for fmt, box in self.window.format_boxes.items():
            box.setChecked(fmt == "md")

    def _select_batch(self, pdfs, output_dir, layout=gui.BY_TYPE):
        self.window.input_type_buttons["pdfs"].setChecked(True)
        self.window._input_type_changed()
        self.window.pdf_paths = list(pdfs)
        self.window.input_path.setText(f"{len(pdfs)} PDF files selected")
        self.window.output_file.setText(str(output_dir))
        self.window.layout_buttons[layout].setChecked(True)

    def _start_and_wait(self, timeout=120):
        self.window._start()
        self.assertIsNotNone(self.window.thread, "the worker thread was never started")
        deadline = time.time() + timeout
        while self.window.thread.is_alive():
            self.app.processEvents()
            time.sleep(0.01)
            if time.time() > deadline:
                self.fail("the OCR worker did not finish in time")
        # Let the queued progress/finished signals reach the GUI thread.
        for _ in range(50):
            self.app.processEvents()
            time.sleep(0.005)

    # --- tests ---

    def test_launch_gui_enters_and_leaves_the_event_loop(self):
        """main.py launches the app through this entry point."""
        QTimer.singleShot(0, self.app.quit)
        self.assertEqual(gui.launch_gui(), 0)

    def test_radio_groups_are_independent(self):
        """Engine, device, and direction radios must not uncheck each other."""
        self.window.engine_buttons["bina"].setChecked(True)
        self.assertFalse(self.window.engine_buttons["chrome"].isChecked())
        self.assertTrue(
            self.window.device_buttons["cuda"].isChecked()
            or self.window.device_buttons["cpu"].isChecked()
        )
        self.assertTrue(self.window.direction_buttons["ltr"].isChecked())

        self.window.device_buttons["cpu"].setChecked(True)
        self.assertTrue(self.window.engine_buttons["bina"].isChecked())

    def test_batch_input_toggles_the_batch_widgets(self):
        self.window.input_type_buttons["pdfs"].setChecked(True)
        self.window._input_type_changed()
        self.assertTrue(self.window.layout_row.isVisibleTo(self.window))
        self.assertFalse(self.window.folder_check.isVisibleTo(self.window))
        self.assertEqual(self.window.output_label.text(), "Output folder:")

        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.assertFalse(self.window.layout_row.isVisibleTo(self.window))
        self.assertTrue(self.window.folder_check.isVisibleTo(self.window))
        self.assertEqual(self.window.output_label.text(), "Transcript:")

    def test_single_pdf_writes_markdown_and_finishes_clean(self):
        pdf = self._text_pdf()
        self._select_single_pdf(pdf, self.root / "single")

        self._start_and_wait()

        self.assertTrue((self.root / "single.md").is_file(), "no markdown was written")
        self.assertEqual(self.window.status_label.text(), "Done")
        self.assertEqual(self.window.progress.value(), self.window.progress.maximum())
        self.assertIn("Wrote single.md", self.window.log.toPlainText())
        # run control returns to the idle state
        self.assertTrue(self.window.start_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())
        self.assertFalse(self.window.timer.isActive())  # elapsed timer stopped
        self.assertFalse(self.window.running)

    def test_output_folder_option_nests_the_transcript(self):
        pdf = self._text_pdf()
        self._select_single_pdf(pdf, self.root / "book", in_folder=True)

        self._start_and_wait()

        self.assertTrue((self.root / "book" / "book.md").is_file())

    def test_save_next_to_input_is_off_by_default(self):
        self.assertFalse(self.window.same_dir_check.isChecked())

    def test_save_next_to_input_writes_beside_the_pdf(self):
        books = self.root / "books"
        books.mkdir()
        pdf = self._text_pdf("paper.pdf", books)
        # A bare name: without the option this would land in the CWD.
        self._select_single_pdf(pdf, "transcript")
        self.window.same_dir_check.setChecked(True)

        self._start_and_wait()

        self.assertTrue((books / "transcript.md").is_file())
        self.assertFalse((Path.cwd() / "transcript.md").exists())

    def test_save_next_to_input_creates_the_folder_beside_the_pdf(self):
        books = self.root / "books"
        books.mkdir()
        pdf = self._text_pdf("paper.pdf", books)
        self._select_single_pdf(pdf, "my_transcript", in_folder=True)
        self.window.same_dir_check.setChecked(True)

        self._start_and_wait()

        self.assertTrue((books / "my_transcript" / "my_transcript.md").is_file())

    def test_save_next_to_input_targets_the_image_folder_itself(self):
        pages = self.root / "pages"
        pages.mkdir()
        self.window.input_type_buttons["dir"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(pages))
        self.window.output_file.setText("pages_transcript")
        self.window.folder_check.setChecked(False)
        self.window.same_dir_check.setChecked(True)

        self.assertEqual(self.window._output_base(), pages / "pages_transcript")

        self.window.folder_check.setChecked(True)
        self.assertEqual(
            self.window._output_base(), pages / "pages_transcript" / "pages_transcript"
        )

    def test_save_next_to_input_is_hidden_for_batch(self):
        self.window.input_type_buttons["pdfs"].setChecked(True)
        self.window._input_type_changed()
        self.assertFalse(self.window.same_dir_check.isVisibleTo(self.window))

        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.assertTrue(self.window.same_dir_check.isVisibleTo(self.window))

    def test_batch_by_type_layout_places_markdown_and_pagemap(self):
        pdfs = [self._text_pdf("first.pdf"), self._text_pdf("second.pdf")]
        self._select_batch(pdfs, self.root / "out", layout=gui.BY_TYPE)

        self._start_and_wait()

        self.assertEqual(self.window.status_label.text(), "Done")
        self.assertTrue((self.root / "out" / "markdown" / "first.md").is_file())
        self.assertTrue((self.root / "out" / "markdown" / "second.md").is_file())
        self.assertTrue(
            (self.root / "out" / "pagemaps" / "first.pagemap.json").is_file()
        )
        self.assertTrue(
            (self.root / "out" / "pagemaps" / "second.pagemap.json").is_file()
        )
        self.assertIn("PDFs succeeded: 2/2", self.window.log.toPlainText())

    def test_batch_per_pdf_layout_uses_one_folder_per_document(self):
        pdfs = [self._text_pdf("paper.pdf")]
        self._select_batch(pdfs, self.root / "out", layout=gui.PER_PDF)

        self._start_and_wait()

        self.assertTrue((self.root / "out" / "paper" / "transcript.md").is_file())
        self.assertTrue(
            (self.root / "out" / "paper" / "transcript.pagemap.json").is_file()
        )

    def test_batch_keeps_going_when_one_document_fails(self):
        pdfs = [self._textless_pdf(), self._text_pdf()]
        self._select_batch(pdfs, self.root / "out", layout=gui.BY_TYPE)

        with patch.object(QMessageBox, "critical") as critical:
            self._start_and_wait()

        self.assertEqual(self.window.status_label.text(), "Done with errors")
        log = self.window.log.toPlainText()
        self.assertIn("[ERROR] scanned.pdf", log)
        self.assertIn("PDFs succeeded: 1/2", log)
        # A per-document failure is logged, not fatal.
        critical.assert_not_called()
        self.assertTrue((self.root / "out" / "markdown" / "paper.md").is_file())
        self.assertFalse((self.root / "out" / "markdown" / "scanned.md").exists())

    def test_start_without_input_warns_and_does_not_start(self):
        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)
        self.assertFalse(self.window.running)

    def test_start_without_formats_warns_and_does_not_start(self):
        self._select_single_pdf(self._text_pdf(), self.root / "single")
        for box in self.window.format_boxes.values():
            box.setChecked(False)

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)

    def test_inspector_with_image_folder_is_rejected(self):
        image_dir = self.root / "pages"
        image_dir.mkdir()
        self.window.input_type_buttons["dir"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(image_dir))
        self.window.engine_buttons["inspector"].setChecked(True)

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)

    def test_batch_refuses_to_start_with_no_pdfs_selected(self):
        self._select_batch([], self.root / "out")

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)


if __name__ == "__main__":
    unittest.main()
