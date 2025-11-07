# Homework Week 3 Notes

## Wikipedia Search Tool Implementation

### Pattern Source
The `wikipedia_search` tool implementation is based on the pattern from `summary_agent.ipynb` (homework_week2), specifically the `_fetch_single_url` function.

### Similarities with `_fetch_single_url`:
1. Uses `requests.get()` with timeout parameter
2. Uses `response.raise_for_status()` for error handling
3. Uses try/except blocks for error handling
4. Similar function structure and docstring style

### Differences:
- **`_fetch_single_url`**: Uses Jina AI to fetch Wikipedia pages as markdown
- **`wikipedia_search`**: Calls Wikipedia search API directly (`/w/api.php?action=query&list=search`)
- **`_fetch_single_url`**: Saves files locally
- **`wikipedia_search`**: Returns structured data (Pydantic models)

### Conclusion
The pattern/style is from `summary_agent.ipynb`, but adapted to call the Wikipedia search API instead of fetching pages via Jina AI.
