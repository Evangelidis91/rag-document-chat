"""Tests for the SHA-256 file hashing logic."""

from rag_engine import compute_file_hash

def test_same_content_gives_same_hash(tmp_path):
    """Two files with IDENTICAL content must produce the SAME hash, even if they
    have different names.
    """
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world")
    file_b.write_text("hello world")  # same content, different name

    assert compute_file_hash(str(file_a)) == compute_file_hash(str(file_b))


def test_different_content_gives_different_hash(tmp_path):
    """Even a tiny content change must produce a DIFFERENT hash."""
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world")
    file_b.write_text("hello there")  # one word different

    assert compute_file_hash(str(file_a)) != compute_file_hash(str(file_b))


def test_hash_is_deterministic(tmp_path):
    """Hashing the same file twice must give the same result."""
    f = tmp_path / "doc.txt"
    f.write_text("repeat me")

    assert compute_file_hash(str(f)) == compute_file_hash(str(f))


def test_hash_has_correct_format(tmp_path):
    """A SHA-256 hash is always 64 hexadecimal characters."""
    f = tmp_path / "doc.txt"
    f.write_text("anything")

    h = compute_file_hash(str(f))
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)