"""Tests for the file classification logic (add / skip / conflict).

Uses a FAKE Chroma collection so we don't need a real database.
"""

import rag_engine

class FakeCollection:
    """A 'test double' that mimics ONLY the parts of a Chroma collection that
    our code actually uses: .count() and .get().
    """

    def __init__(self, metadatas):
        self._metadatas = metadatas

    def count(self):
        return len(self._metadatas)

    def get(self):
        return {"metadatas": self._metadatas}


def test_brand_new_file_is_added(tmp_path, monkeypatch):
    """A file not seen before should be classified as 'to_add'."""
    # Empty collection = nothing indexed yet
    fake = FakeCollection([])
    monkeypatch.setattr(rag_engine, "_get_collection", lambda: fake)

    # Create a new file in the temp folder
    f = tmp_path / "new.txt"
    f.write_text("brand new content")

    classification, _ = rag_engine.classify_files(str(tmp_path))

    assert "new.txt" in classification["to_add"]
    assert classification["to_skip"] == []
    assert classification["conflicts"] == []


def test_identical_file_is_skipped(tmp_path, monkeypatch):
    """A file whose content hash is already indexed should be skipped."""
    f = tmp_path / "doc.txt"
    f.write_text("already indexed content")

    # Pre-compute the hash this file WILL have
    known_hash = rag_engine.compute_file_hash(str(f))

    # The fake collection already contains this exact hash
    fake = FakeCollection([{"file_name": "doc.txt", "file_hash": known_hash}])
    monkeypatch.setattr(rag_engine, "_get_collection", lambda: fake)

    classification, _ = rag_engine.classify_files(str(tmp_path))

    assert "doc.txt" in classification["to_skip"]


def test_same_name_new_content_is_conflict(tmp_path, monkeypatch):
    """A file with the same NAME but DIFFERENT content (hash) should be
    flagged as a conflict.
    """
    f = tmp_path / "doc.txt"
    f.write_text("NEW different content")

    # Collection has 'doc.txt' but with an OLD, different hash
    fake = FakeCollection([
        {"file_name": "doc.txt", "file_hash": "some_old_hash_value"}
    ])
    monkeypatch.setattr(rag_engine, "_get_collection", lambda: fake)

    classification, _ = rag_engine.classify_files(str(tmp_path))

    assert "doc.txt" in classification["conflicts"]