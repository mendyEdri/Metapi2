"""Utilities for splitting system prompts into smaller chunks.

This module exposes a :func:`chunk_prompt` helper that first attempts to
interpret the provided text as XML. When the text contains well-formed XML
tags the content of each tag becomes an individual chunk. If the text either
contains no tags or cannot be parsed as XML we fall back to LangChain's
``RecursiveCharacterTextSplitter`` to create evenly sized chunks.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_prompt(text: str) -> List[str]:
    """Split ``text`` into a list of chunks.

    The function prefers XML-based chunking when possible. It first checks if
    the text appears to contain XML tags. If parsing succeeds, the textual
    content of each element (in document order) is returned. When the text has
    no tags or the XML is malformed we fall back to a ``RecursiveCharacter-
    TextSplitter`` with a ``chunk_size`` of 500 characters and an overlap of
    50 characters.

    Parameters
    ----------
    text:
        The raw system prompt text to split.

    Returns
    -------
    list[str]
        A list of extracted chunks. Whitespace-only chunks are omitted.
    """

    if "<" in text and ">" in text:
        candidates = [(text, False), (f"<root>{text}</root>", True)]
        for candidate, skip_root in candidates:
            try:
                root = ET.fromstring(candidate)
            except ET.ParseError:
                continue
            else:
                return [
                    elem.text.strip()
                    for elem in root.iter()
                    if (not skip_root or elem is not root)
                    and elem.text
                    and elem.text.strip()
                ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [doc.page_content for doc in splitter.create_documents([text])]

