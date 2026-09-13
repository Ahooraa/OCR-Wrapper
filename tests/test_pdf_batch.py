import tempfile
import unittest
from pathlib import Path

from pdf_batch import (
    BY_TYPE,
    MAX_OUTPUT_PATH,
    PER_PDF,
    beside_input,
    create_jobs,
    place_pagemap,
)


class BesideInputTests(unittest.TestCase):
    def test_pdf_transcript_lands_in_the_pdf_directory(self):
        self.assertEqual(
            beside_input(Path("/books/out/transcript"), Path("/books/paper.pdf")),
            Path("/books/transcript"),
        )

    def test_image_folder_transcript_lands_inside_that_folder(self):
        self.assertEqual(
            beside_input(
                Path("/books/transcript"), Path("/scans/pages"), source_is_dir=True
            ),
            Path("/scans/pages/transcript"),
        )

    def test_an_absolute_output_path_keeps_only_its_name(self):
        self.assertEqual(
            beside_input(Path("/elsewhere/a/b/transcript"), Path("/books/paper.pdf")),
            Path("/books/transcript"),
        )


class PDFBatchTests(unittest.TestCase):
    def test_per_pdf_layout_uses_short_fixed_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / ("long-paper-title-" * 12 + ".pdf")
            pdf.touch()

            [job] = create_jobs([pdf], root / "output", PER_PDF)

            self.assertEqual(job.output_base.name, "transcript")
            self.assertLessEqual(len(str(job.pagemap_path)), MAX_OUTPUT_PATH)

    def test_by_type_layout_separates_markdown_and_pagemaps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "paper.pdf"
            pdf.touch()

            [job] = create_jobs([pdf], root / "output", BY_TYPE)

            self.assertEqual(job.output_base, root / "output" / "markdown" / "paper")
            self.assertEqual(
                job.pagemap_path,
                root / "output" / "pagemaps" / "paper.pagemap.json",
            )

    def test_place_pagemap_moves_sidecar_to_json_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "paper.pdf"
            pdf.touch()
            [job] = create_jobs([pdf], root / "output", BY_TYPE)
            generated = job.output_base.with_suffix(".pagemap.json")
            generated.parent.mkdir(parents=True)
            generated.write_text("[0]", encoding="utf-8")

            place_pagemap(job)

            self.assertFalse(generated.exists())
            self.assertEqual(job.pagemap_path.read_text(encoding="utf-8"), "[0]")


if __name__ == "__main__":
    unittest.main()
