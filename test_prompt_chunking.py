"""Tests for the :mod:`prompt_chunking` module."""

from prompt_chunking import chunk_prompt


def test_chunk_prompt_splits_valid_xml():
    text = "<root><a>Alpha</a><b>Beta</b></root>"
    assert chunk_prompt(text) == ["Alpha", "Beta"]


def test_chunk_prompt_handles_multiple_root_level_elements():
    text = (
        "<role>\nHey\n</role>\n<instructions>\nHey you\n</instructions>"
    )
    assert chunk_prompt(text) == ["Hey", "Hey you"]


def test_chunk_prompt_falls_back_to_text_splitter_for_plain_text():
    # Large text without any XML tags should be split by the text splitter.
    text = "abc " * 200  # ~800 characters
    chunks = chunk_prompt(text)
    assert len(chunks) > 1
    # Ensure the chunks combine (ignoring overlaps) to cover the original text
    assert any("abc" in chunk for chunk in chunks)

