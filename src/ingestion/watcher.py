"""Background file system watcher for incremental document re-indexing.

Monitors data/raw/ using watchdog. When a supported document (.pdf, .docx,
.md, .txt) is created, modified, or deleted, it triggers build_index() asynchronously
in a background thread to keep ChromaDB and BM25 indices synchronized.
"""
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.ingestion.index import DEFAULT_RAW_DIR, build_index
from src.ingestion.parsers import SUPPORTED_EXTENSIONS

logger = logging.getLogger("rag_watcher")
logging.basicConfig(level=logging.INFO)

_observer: Observer | None = None
_debounce_timer: threading.Timer | None = None
_lock = threading.Lock()


def _trigger_reindex():
    global _debounce_timer
    with _lock:
        _debounce_timer = None

    logger.info("⚡ [Auto-Watcher] Triggering incremental document re-indexing...")
    try:
        res = build_index(force_full=False)
        logger.info(
            f"✓ [Auto-Watcher] Indexing complete: {res['total_chunks']} total chunks. "
            f"Embedded: {len(res['docs_embedded'])}, Deleted: {len(res['docs_deleted'])}"
        )
    except Exception as e:
        logger.error(f"❌ [Auto-Watcher] Error during background re-indexing: {e}")


def _schedule_reindex(delay: float = 1.5):
    global _debounce_timer
    with _lock:
        if _debounce_timer is not None:
            _debounce_timer.cancel()
        _debounce_timer = threading.Timer(delay, _trigger_reindex)
        _debounce_timer.daemon = True
        _debounce_timer.start()


class RawDocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info(f"📂 [Auto-Watcher] File created: {event.src_path}")
            _schedule_reindex()

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info(f"📂 [Auto-Watcher] File modified: {event.src_path}")
            _schedule_reindex()

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info(f"📂 [Auto-Watcher] File deleted: {event.src_path}")
            _schedule_reindex()


def start_watcher(raw_dir: Path = DEFAULT_RAW_DIR) -> Observer:
    global _observer
    if _observer is not None and _observer.is_alive():
        return _observer

    raw_dir.mkdir(parents=True, exist_ok=True)
    event_handler = RawDocHandler()
    _observer = Observer()
    _observer.schedule(event_handler, str(raw_dir), recursive=False)
    _observer.daemon = True
    _observer.start()
    logger.info(f"🟢 [Auto-Watcher] Monitoring {raw_dir} for document changes...")
    return _observer


def stop_watcher():
    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join()
        _observer = None
        logger.info("🔴 [Auto-Watcher] File watcher stopped.")
