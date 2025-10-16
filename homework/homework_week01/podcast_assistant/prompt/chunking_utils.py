# Document chunking utilities for splitting large documents into smaller, overlapping pieces.

from typing import Any, Dict, Iterable, List


def _sliding_window(seq: Iterable[Any], size: int, step: int) -> List[Dict[str, Any]]:
    if size < 1 or step < 1:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i : i + size]
        result.append({"start": i, "content": batch})
        if i + size > n:
            break

    return result


def chunk_documents(
    documents: Iterable[Dict[str, str]],
    size: int | None = None,
    step: int | None = None,
    content_field_name: str | None = None,
) -> List[Dict[str, str]]:
    """
    Split a collection of documents into smaller chunks using sliding windows.

    Takes documents and breaks their content into overlapping chunks while preserving
    all other document metadata (filename, etc.) in each chunk.

    Args:
        documents: An iterable of document dictionaries. Each document must have a content field.
        size (int, optional): The maximum size of each chunk. Defaults to 2000.
        step (int, optional): The step size between chunks. Defaults to 1000.
        content_field_name (str, optional): The name of the field containing document content.
                                          Defaults to 'content'.

    Returns:
        list: A list of chunk dictionaries. Each chunk contains:
            - All original document fields except the content field
            - 'start': Starting position of the chunk in original content
            - 'content': The chunk content

    Example:
        >>> documents = [{'content': 'long text...', 'filename': 'doc.txt'}]
        >>> chunks = chunk_documents(documents, size=100, step=50)
        >>> # Or with custom content field:
        >>> documents = [{'text': 'long text...', 'filename': 'doc.txt'}]
        >>> chunks = chunk_documents(documents, content_field_name='text')
    """
    from config import ChunkingConfig

    if size is None:
        size = ChunkingConfig.DEFAULT_SIZE.value
    if step is None:
        # Calculate step from default overlap
        step = int(
            ChunkingConfig.DEFAULT_SIZE.value
            * (1 - ChunkingConfig.DEFAULT_OVERLAP.value)
        )
    if content_field_name is None:
        content_field_name = ChunkingConfig.DEFAULT_CONTENT_FIELD.value

    results = []

    for doc in documents:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop(content_field_name)
        chunks = _sliding_window(doc_content, size=size, step=step)
        for chunk in chunks:
            chunk.update(doc_copy)
        results.extend(chunks)

    return results
