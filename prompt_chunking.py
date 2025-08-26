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

    The function uses multiple strategies to identify semantic chunks:
    1. XML-based chunking for structured prompts
    2. Markdown header-based chunking (## headers)
    3. Section-based chunking (numbered lists, bullet points)
    4. Paragraph-based chunking
    5. RecursiveCharacterTextSplitter as fallback

    Parameters
    ----------
    text:
        The raw system prompt text to split.

    Returns
    -------
    list[str]
        A list of extracted chunks. Whitespace-only chunks are omitted.
    """

    # First try XML-based chunking
    if "<" in text and ">" in text:
        candidates = [(text, False), (f"<root>{text}</root>", True)]
        for candidate, skip_root in candidates:
            try:
                root = ET.fromstring(candidate)
            except ET.ParseError:
                continue
            else:
                chunks = [
                    elem.text.strip()
                    for elem in root.iter()
                    if (not skip_root or elem is not root)
                    and elem.text
                    and elem.text.strip()
                ]
                if chunks:  # Only return if we found valid chunks
                    return chunks

    # Try markdown header-based chunking
    if "##" in text:
        # Split by markdown headers
        sections = []
        current_section = ""
        
        for line in text.split('\n'):
            if line.strip().startswith('##'):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        if len(sections) > 1:  # Only use if we found multiple sections
            return sections
    
    # Try section-based chunking (numbered lists, bullet points)
    if any(pattern in text for pattern in ['1.', '2.', '•', '*', '-']):
        # Split by common section markers
        import re
        section_patterns = [
            r'^\d+\.',  # Numbered lists
            r'^[•*-]\s+',  # Bullet points
            r'^[A-Z][a-z]+:',  # Section headers like "Instructions:"
        ]
        
        sections = []
        current_section = ""
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            if any(re.match(pattern, line_stripped) for pattern in section_patterns):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        if len(sections) > 1:
            return sections
    
    # Try paragraph-based chunking (split by double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    
    # Try single newline separation for simple line-based prompts
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) > 1 and len(text) < 500:  # For short prompts, split by lines
        return lines
    
    # Fallback to RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [doc.page_content for doc in splitter.create_documents([text])]
    
    # If we still only have one chunk, try to split by sentences
    if len(chunks) == 1 and len(text) > 100:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 1:
            # Group sentences into reasonable chunks
            result = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < 200:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        result.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk.strip():
                result.append(current_chunk.strip())
            if len(result) > 1:
                return result
    
    return chunks

