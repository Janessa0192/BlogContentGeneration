from pathlib import Path


def read_text(path) -> str:
    """Reads UTF-8, falling back to Windows cp1252 for files saved by earlier versions."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def write_text(path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")