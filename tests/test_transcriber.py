"""Tests for the shared engine/transcriber factory.

Every engine is stubbed out, so these run without a model download, without the
Windows/Chrome OCR DLLs, and without touching the real Hugging Face cache.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import transcriber


class FakeNormalizer:
    def normalize(self, text):
        return text.upper()


class UsesPageTranscriberTests(unittest.TestCase):
    def test_pdf_inspector_has_no_page_transcriber(self):
        self.assertFalse(transcriber.uses_page_transcriber("inspector"))

    def test_page_engines_need_a_transcriber(self):
        for engine in ("bina", "oneocr", "chrome"):
            self.assertTrue(transcriber.uses_page_transcriber(engine))


class BuildTranscriberTests(unittest.TestCase):
    def test_whole_document_engine_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            transcriber.build_transcriber("inspector")
        self.assertIn("no page transcriber", str(raised.exception))

    def test_oneocr_binds_its_engine_and_passes_the_page(self):
        page = Path("page_0001.png")
        with patch.object(transcriber, "get_ocr_engine", return_value="ENGINE") as build, \
                patch.object(transcriber, "oneocr_transcribe_page", return_value="text") as page_fn:
            transcribe = transcriber.build_transcriber("oneocr")
            self.assertEqual(transcribe(page), "text")
        build.assert_called_once_with()
        page_fn.assert_called_once_with("ENGINE", page)

    def test_chrome_binds_its_engine_and_passes_the_page(self):
        page = Path("page_0001.png")
        with patch.object(transcriber, "get_screenai_engine", return_value="SCREENAI"), \
                patch.object(transcriber, "chrome_transcribe_page", return_value="text") as page_fn:
            transcribe = transcriber.build_transcriber("chrome")
            self.assertEqual(transcribe(page), "text")
        page_fn.assert_called_once_with("SCREENAI", page)

    def test_normalize_wraps_the_transcriber_and_is_logged(self):
        lines = []
        with patch.object(transcriber, "get_ocr_engine", return_value="ENGINE"), \
                patch.object(transcriber, "oneocr_transcribe_page", return_value="abc"), \
                patch.object(transcriber, "get_normalizer", return_value=FakeNormalizer()) as get_norm:
            transcribe = transcriber.build_transcriber(
                "oneocr", normalize=True, log=lines.append
            )
            self.assertEqual(transcribe(Path("p.png")), "ABC")
        get_norm.assert_called_once_with()
        self.assertIn("[INFO] Persian normalization enabled (hazm)", lines)

    def test_status_hook_receives_engine_loading_messages(self):
        statuses = []
        with patch.object(transcriber, "get_screenai_engine", return_value="S"), \
                patch.object(transcriber, "chrome_transcribe_page"):
            transcriber.build_transcriber("chrome", status=statuses.append)
        self.assertEqual(statuses, ["Loading Chrome Screen AI..."])

    def test_status_stays_silent_when_no_hook_is_given(self):
        with patch.object(transcriber, "get_ocr_engine", return_value="E"), \
                patch.object(transcriber, "oneocr_transcribe_page"), \
                patch("builtins.print") as printed:
            transcriber.build_transcriber("oneocr")
        printed.assert_not_called()

    def test_bina_declined_download_stops_without_an_error(self):
        lines = []
        with patch.object(transcriber, "model_cache_info", return_value=(False, None)), \
                patch.object(transcriber, "repo_size_gb", return_value=1.3):
            result = transcriber.build_transcriber(
                "bina", log=lines.append, confirm_download=lambda size_gb: False
            )
        self.assertIsNone(result)
        self.assertTrue(any("is not downloaded yet" in line for line in lines))
        self.assertIn("Aborted - model not downloaded.", lines)

    def test_bina_without_a_prompt_hook_declines(self):
        with patch.object(transcriber, "model_cache_info", return_value=(False, None)), \
                patch.object(transcriber, "repo_size_gb", return_value=1.3):
            self.assertIsNone(transcriber.build_transcriber("bina", log=lambda m: None))

    def test_bina_uses_the_cache_without_asking(self):
        asked = []
        with patch.object(transcriber, "model_cache_info", return_value=(True, 1.3)), \
                patch.object(transcriber, "load_model", return_value=("P", "M", "cpu")), \
                patch.object(transcriber, "transcribe_page", return_value="text"):
            transcribe = transcriber.build_transcriber(
                "bina", log=lambda m: None,
                confirm_download=lambda size_gb: asked.append(size_gb),
            )
        self.assertEqual(asked, [])
        self.assertIsNotNone(transcribe)

    def test_bina_loads_the_model_once_with_the_configured_token_budget(self):
        page = Path("p.png")
        with patch.object(transcriber, "model_cache_info", return_value=(True, 1.3)), \
                patch.object(transcriber, "load_model", return_value=("PROC", "MODEL", "cuda")) as load, \
                patch.object(transcriber, "transcribe_page", return_value="text") as page_fn:
            transcribe = transcriber.build_transcriber(
                "bina", force_cpu=True, max_new_tokens=77, log=lambda m: None
            )
            self.assertEqual(transcribe(page), "text")
        self.assertTrue(load.call_args.kwargs["force_cpu"])
        self.assertIn("log", load.call_args.kwargs)
        page_fn.assert_called_once_with("PROC", "MODEL", page, 77)


class CliTranscriberWiringTests(unittest.TestCase):
    """The CLI flags must reach the factory unchanged."""

    def _args(self, **overrides):
        values = {
            "engine": "bina", "normalize": False, "cpu": False, "max_new_tokens": 1024,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_flags_are_forwarded_to_the_factory(self):
        with patch.object(main, "build_transcriber", return_value="T") as build:
            result = main.make_transcriber(
                self._args(engine="bina", normalize=True, cpu=True, max_new_tokens=256)
            )
        self.assertEqual(result, "T")
        self.assertEqual(build.call_args.args, ("bina",))
        kwargs = build.call_args.kwargs
        self.assertTrue(kwargs["normalize"])
        self.assertTrue(kwargs["force_cpu"])
        self.assertEqual(kwargs["max_new_tokens"], 256)
        self.assertEqual(kwargs["log"], print)
        self.assertEqual(kwargs["confirm_download"], main.confirm_download)

    def test_whole_document_engines_skip_the_factory(self):
        with patch.object(main, "build_transcriber") as build:
            self.assertIsNone(main.make_transcriber(self._args(engine="inspector")))
        build.assert_not_called()

    def test_console_prompt_accepts_yes_and_declines_everything_else(self):
        for answer in ("y", "Y", "yes", "YES", " yes "):
            with self.subTest(answer=answer), patch("builtins.input", return_value=answer):
                self.assertTrue(main.confirm_download(1.3))
        for answer in ("", "n", "no", "later"):
            with self.subTest(answer=answer), patch("builtins.input", return_value=answer):
                self.assertFalse(main.confirm_download(1.3))

    def test_console_prompt_declines_on_eof(self):
        with patch("builtins.input", side_effect=EOFError):
            self.assertFalse(main.confirm_download(1.3))


if __name__ == "__main__":
    unittest.main()
