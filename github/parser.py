# GitHub data parsing utilities
from typing import Any, Dict, List

import frontmatter

from github.reader import RawRepositoryFile


def parse_data(data_raw: List[RawRepositoryFile]) -> List[Dict[str, Any]]:
    """
    Parse raw GitHub repository files into structured data.

    Args:
        data_raw: List of RawRepositoryFile objects from GitHub

    Returns:
        List of parsed document dictionaries with frontmatter metadata
    """
    data_parsed = []
    for f in data_raw:
        post = frontmatter.loads(f.content)
        data = post.to_dict()
        data["filename"] = f.filename
        data_parsed.append(data)

    return data_parsed
