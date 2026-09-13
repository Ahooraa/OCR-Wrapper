"""Build page transcribers for the OCR engines.

One place decides how each engine is wired, so the CLI and the GUI cannot drift
apart: pdf-inspector has no page transcriber at all (it extracts a whole PDF in
a single call), and bina needs its model cached before the first page runs.
"""

from model import MODEL_ID, load_model, model_cache_info, repo_size_gb
from normalize import get_normalizer, normalize_transcribe
from ocr import transcribe_page
from chrome_ocr_engine import chrome_transcribe_page, get_screenai_engine
from windows_ocr import get_ocr_engine, oneocr_transcribe_page

# Engines that work on whole documents instead of page by page.
WHOLE_DOCUMENT_ENGINES = ("inspector",)


def uses_page_transcriber(engine: str) -> bool:
    """False for engines that process a whole document in one call."""
    return engine not in WHOLE_DOCUMENT_ENGINES


def build_transcriber(engine, *, normalize=False, force_cpu=False, max_new_tokens=1024,
                      log=print, confirm_download=None, status=None):
    """Return ``transcribe(page_path) -> str`` for ``engine``.

    Returns None when the engine has no page transcriber (pdf-inspector) or when
    the model download was declined - callers should stop without an error.

    log(msg)                  log lines, including the model-cache notice
    confirm_download(size_gb) asked before downloading the bina model; when
                              omitted the download is declined
    status(msg)               short state changes for a UI status line
    """
    report = status if status is not None else (lambda message: None)

    normalizer = None
    if normalize:
        normalizer = get_normalizer()
        log("[INFO] Persian normalization enabled (hazm)")

    if engine == "oneocr":
        report("Loading Windows OCR engine...")
        ocr_engine = get_ocr_engine()
        transcribe = lambda page: oneocr_transcribe_page(ocr_engine, page)
    elif engine == "chrome":
        report("Loading Chrome Screen AI...")
        ocr_engine = get_screenai_engine()
        transcribe = lambda page: chrome_transcribe_page(ocr_engine, page)
    elif engine == "bina":
        cached, _ = model_cache_info()
        if not cached:
            size_gb = repo_size_gb()
            log(f"[INFO] Model {MODEL_ID} is not downloaded yet (~{size_gb:.1f} GB).")
            if confirm_download is None or not confirm_download(size_gb):
                log("Aborted - model not downloaded.")
                return None
        report("Loading model...")
        processor, model, _ = load_model(force_cpu=force_cpu, log=log)
        transcribe = lambda page: transcribe_page(processor, model, page, max_new_tokens)
    else:
        raise ValueError(f"Engine {engine!r} has no page transcriber")

    if normalizer is not None:
        transcribe = normalize_transcribe(transcribe, normalizer)
    return transcribe
