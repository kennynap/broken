from pathlib import Path
import pytest
from jax.tools import ListDirectory, InspectFile, CreateDirectory, MoveFile, RenameFile, VerifyPath

def test_filesystem_tools(tmp_path):
    root = tmp_path / "Downloads"
    root.mkdir()
    source = root / "invoice.pdf"
    source.write_text("test", encoding="utf-8")

    listing = ListDirectory(root).execute(".")
    assert listing.ok
    assert listing.data["items"][0]["name"] == "invoice.pdf"

    info = InspectFile(root).execute("invoice.pdf")
    assert info.ok
    assert info.data["suffix"] == ".pdf"

    assert CreateDirectory(root).execute("Documents").ok
    assert MoveFile(root).execute("invoice.pdf", "Documents/invoice.pdf").ok

    assert VerifyPath(root).execute("Documents/invoice.pdf").data["exists"] is True

    assert RenameFile(root).execute("Documents/invoice.pdf", "bill.pdf").ok
    assert VerifyPath(root).execute("Documents/bill.pdf").data["exists"] is True

def test_no_overwrite(tmp_path):
    root = tmp_path / "Downloads"
    root.mkdir()
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")
    result = MoveFile(root).execute("a.txt", "b.txt")
    assert not result.ok
    assert (root / "a.txt").exists()
    assert (root / "b.txt").read_text() == "b"

def test_path_escape_blocked(tmp_path):
    root = tmp_path / "Downloads"
    root.mkdir()
    with pytest.raises(PermissionError):
        ListDirectory(root).execute("../outside")
