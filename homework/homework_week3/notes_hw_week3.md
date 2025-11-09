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

---

## Why Both LLM-as-a-Judge and Ground Truth Data Are Needed

### The Problem
In practice, both LLM-as-a-Judge and ground truth data are used together because they measure different aspects of agent performance and complement each other.

### Ground Truth Data
- **Purpose**: Objective, retrieval-focused metrics
- **Measures**:
  - **Hit Rate**: Did the agent find the expected Wikipedia sources?
  - **MRR (Mean Reciprocal Rank)**: How high did expected sources rank?
- **What it tells you**: Whether retrieval found the right sources

### LLM-as-a-Judge
- **Purpose**: Subjective, answer-quality metrics
- **Measures**:
  - **Accuracy**: Is the answer factually correct?
  - **Completeness**: Does it cover all aspects?
  - **Relevance**: Does it address the question?
- **What it tells you**: Whether the answer is good, even if sources differ

### Why Both Together?

1. **Different Dimensions**:
   - Ground truth: Retrieval quality (did it find the right sources?)
   - Judge: Generation quality (is the answer good?)

2. **Complementary Validation**:
   - High hit rate + low judge score → Retrieval works, but synthesis is poor
   - Low hit rate + high judge score → Found different but valid sources
   - Both high → Strong performance

3. **Combined Score**:
   ```
   score = (hit_rate^2.0 * judge_score^1.5) / (num_tokens/1000)^0.5
   ```
   This balances retrieval quality (hit rate) and answer quality (judge score).

### In Practice
- **Ground truth alone**: Only measures retrieval, not answer quality
- **Judge alone**: No objective baseline; can be inconsistent or biased
- **Both together**: Objective retrieval metrics + subjective quality assessment

This project uses both in the evaluation workflow: calculate source metrics from ground truth, then run judge evaluation, then combine them into a single score.

---

## Pytest Async Fixture Issue

### The Problem
When using `@pytest.fixture` with an async function, pytest doesn't await the coroutine, causing tests to receive a coroutine object instead of the actual result.

### Why It Happens

**Async functions return coroutines:**
```python
async def my_function():
    return "result"

result = my_function()  # ❌ This is a coroutine object, not "result"
print(type(result))     # <class 'coroutine'>

# You need to await it:
result = await my_function()  # ✅ This is "result"
```

**With `@pytest.fixture` (broken):**
```python
@pytest.fixture
async def my_fixture():
    return await some_async_operation()
```

Pytest doesn't know it's async, so it calls it like a regular function:
```python
# Pytest internally does:
value = my_fixture()  # ❌ Gets a coroutine object, not the result!
# Tries to use coroutine in test → AttributeError
```

**With `@pytest_asyncio.fixture` (fixed):**
```python
@pytest_asyncio.fixture
async def my_fixture():
    return await some_async_operation()
```

pytest-asyncio recognizes it's async and awaits it:
```python
# pytest-asyncio internally does:
coroutine = my_fixture()  # Gets coroutine
value = await coroutine   # ✅ Awaits it to get actual result
# Uses value in test
```

### In Our Specific Case

**Before (broken):**
```python
@pytest.fixture(params=TEST_QUESTIONS)
async def agent_result(request):
    question = request.param
    return await query_wikipedia(question)  # Returns coroutine

# In test:
async def test_search_tool_is_invoked(agent_result):
    # agent_result is a coroutine object, not WikipediaAgentResponse
    tool_names = [call["tool_name"] for call in agent_result.tool_calls]
    # ❌ AttributeError: 'coroutine' object has no attribute 'tool_calls'
```

**After (fixed):**
```python
@pytest_asyncio.fixture(params=TEST_QUESTIONS)
async def agent_result(request):
    question = request.param
    return await query_wikipedia(question)  # Returns WikipediaAgentResponse

# In test:
async def test_search_tool_is_invoked(agent_result):
    # agent_result is WikipediaAgentResponse (awaited by pytest-asyncio)
    tool_names = [call["tool_name"] for call in agent_result.tool_calls]
    # ✅ Works! agent_result has .tool_calls attribute
```

### Summary
- `@pytest.fixture` + async function: Pytest calls the function, gets a coroutine, doesn't await it → test receives a coroutine object
- `@pytest_asyncio.fixture` + async function: pytest-asyncio recognizes async, awaits the coroutine → test receives the actual result

This is why `pytest_asyncio.fixture` is needed for async fixtures: it ensures the coroutine is awaited before the fixture value is passed to tests.

---

## Early Stopping vs Multi-Search: Best Practices

### The Trade-off

**Early stopping makes sense when:**
1. Agent found sufficient information
   - One search can find the right Wikipedia page
   - Answer quality is good
   - Cost efficiency matters

2. Limited search space (Wikipedia)
   - Wikipedia is finite and well-indexed
   - One good query often finds the target page
   - Multiple searches can be redundant

3. Production use
   - Lower latency and cost
   - Users want quick answers
   - Good enough > perfect

**Multi-search makes sense when:**
1. Evaluation and consistency
   - Reproducible behavior for testing
   - Ensures thoroughness
   - Catches edge cases

2. Complex questions
   - Need multiple perspectives
   - Related topics matter
   - Verification across sources

3. Quality over speed
   - Comprehensive answers
   - Research-style queries
   - When accuracy is critical

### Recommended Approach: Adaptive Behavior

1. **Minimum searches for consistency**
   - Set a minimum (e.g., 2-3) for evaluation
   - Ensures some exploration
   - Allows early stopping after minimum

2. **Maximum searches for cost control**
   - Cap at 6-8 searches
   - Prevents runaway behavior
   - Balances thoroughness and efficiency

3. **Quality-based early stopping**
   - Stop early if confidence is high
   - Continue if initial results are poor
   - Use confidence threshold

4. **Different strategies for different use cases**
   - Production: Allow early stopping after minimum
   - Evaluation: Enforce minimums for consistency
   - Research mode: Require more searches

### For Wikipedia Agent

**Current behavior (early stopping):**
```
1 search → finds good results → retrieves pages → stops
```
- ✅ Efficient
- ✅ Works for simple questions
- ❌ Inconsistent for evaluation

**Instruction-based behavior (multi-search):**
```
3-5 broad searches → 8-12 specific searches → retrieves pages
```
- ✅ Thorough
- ✅ Consistent
- ❌ Can be wasteful for simple questions

**Recommended hybrid approach:**
```python
# Pseudo-code for adaptive behavior
def should_continue_searching(search_count, results_quality, confidence):
    # Always do minimum searches for consistency
    if search_count < MIN_SEARCH_CALLS:
        return True

    # Stop early if we have high confidence
    if confidence > 0.9 and results_quality == "good":
        return False

    # Continue if results are poor
    if results_quality == "poor":
        return True

    # Stop at maximum
    if search_count >= MAX_SEARCH_CALLS:
        return False

    return True
```

### Practical Recommendations

**For evaluation (tests):**
- Enforce minimums (3 searches, 2 page retrievals)
- Ensures consistent, testable behavior
- Catches when agent is too lazy

**For production (CLI):**
- Allow early stopping after minimum
- Better user experience
- Lower cost

**For current situation:**
1. Keep test minimums as they are
   - They validate agent can be thorough
   - They catch regression

2. Make instructions adaptive
   - "Do at least 3 searches, but stop early if you find excellent results"
   - "Continue searching if initial results are poor"

3. Consider two modes
   - Evaluation mode: Strict minimums
   - Production mode: Adaptive early stopping

### Summary

- **Early stopping is fine** for production when quality is good
- **For evaluation**, enforce minimums for consistency
- **Use adaptive behavior**: Minimum for consistency, early stop when quality is high, continue if quality is low, cap at maximum

The test failures suggest the agent is being too efficient. That's fine for production, but for evaluation you want consistent, thorough behavior. Consider making the search strategy adaptive based on the use case.

---

## Why Ground Truth is Needed and How It Connects to LLM-as-a-Judge

### Two Different Evaluation Dimensions

**1. Ground Truth → Source Metrics (Retrieval Quality)**
- **Measures**: Did the agent find the right Wikipedia sources?
- **Metrics**: Hit Rate, MRR
- **Purpose**: Objective, retrieval-focused evaluation
- **Example**: Question asks about "customer behavior" → Did it find "Consumer behaviour" page?

**2. LLM-as-a-Judge → Answer Quality (Generation Quality)**
- **Measures**: Is the answer good (accurate, complete, relevant)?
- **Metrics**: Accuracy, Completeness, Relevance, Overall Score
- **Purpose**: Subjective, answer-quality evaluation
- **Example**: Question asks about "customer behavior" → Is the answer factually correct and comprehensive?

### Why You Need Both

They measure different things and complement each other:

**Scenario 1: High Hit Rate + Low Judge Score**
- Agent found the right sources (good retrieval)
- But answer quality is poor (bad synthesis)
- **Conclusion**: Retrieval works, but answer generation needs improvement

**Scenario 2: Low Hit Rate + High Judge Score**
- Agent didn't find expected sources (poor retrieval)
- But answer is still good (found different but valid sources)
- **Conclusion**: Agent found alternative valid sources, or expected sources were wrong

**Scenario 3: Both High**
- Agent found expected sources (good retrieval)
- Answer is high quality (good synthesis)
- **Conclusion**: Strong overall performance

### How They Work Together in Evaluation

The evaluation workflow:
1. Run agent on question → Get answer + sources
2. Calculate source metrics (hit rate, MRR) using ground truth `expected_sources`
3. Run judge evaluation on the answer → Get judge scores
4. Combine both into a single score: `(hit_rate^2.0 * judge_score^1.5) / (num_tokens/1000)^0.5`

### Why Manual Curation Matters

The ground truth `expected_sources` must be accurate because:
- If `expected_sources` are wrong → Hit rate/MRR will be misleading
- If `expected_sources` don't match what the agent actually finds → You can't measure retrieval quality properly
- The judge can't tell you if the agent found the "right" sources — only if the answer is good

### Example

**Question**: "What factors influence customer behavior?"

**Ground truth (manual curation)**:
- `expected_sources: ["Consumer behaviour", "Behavioral economics"]`
- These are the sources that should be found for this question

**Evaluation**:
1. Agent finds: `["Consumer behaviour", "Marketing"]` → Hit rate = 1.0 (found "Consumer behaviour")
2. Judge evaluates answer → Overall score = 0.85 (answer is good)
3. Combined score = `(1.0^2.0 * 0.85^1.5) / (tokens/1000)^0.5`

**Without accurate ground truth**:
- If `expected_sources` were wrong → Hit rate would be wrong → Combined score would be wrong → You can't properly evaluate retrieval quality

### Summary

- **Ground truth (`expected_sources`)** = Did it find the right sources? (Retrieval quality)
- **LLM-as-a-Judge** = Is the answer good? (Generation quality)
- **Both together** = Complete evaluation (retrieval + generation)

You need to manually curate `expected_sources` so the source metrics (hit rate, MRR) are accurate. The judge can't validate retrieval quality — it only evaluates answer quality.

---

## How Many Questions Should Be in Ground Truth?

### Recommended: 15-25 Questions

**Minimum Viable: 10-15 Questions**
- ✅ Fast to curate
- ✅ Quick to run evaluation
- ❌ Less statistical confidence
- ❌ May miss edge cases
- **Use when**: Quick validation, limited time/resources

**Recommended: 15-25 Questions**
- ✅ Good statistical confidence
- ✅ Covers diverse question types
- ✅ Manageable curation effort
- ✅ Reasonable evaluation cost
- **Use when**: Standard evaluation, balanced effort vs. confidence

**Comprehensive: 25-50 Questions**
- ✅ High statistical confidence
- ✅ Broad coverage of edge cases
- ✅ Robust evaluation
- ❌ Significant curation time
- ❌ Longer evaluation runs
- **Use when**: Production evaluation, thorough testing

### Statistical Significance

- **10 questions**: ~10% margin of error
- **20 questions**: ~7% margin of error
- **30+ questions**: ~5% margin of error

### Evaluation Cost

Each question requires:
- Agent query (multiple API calls for searches + retrievals)
- Judge evaluation (1 API call)
- **With 20 questions**: ~20 agent queries + 20 judge calls

### Manual Curation Effort

Each question needs:
1. Run agent query
2. Review sources found
3. Curate expected_sources
4. **Estimate**: 5-10 minutes per question
5. **20 questions** ≈ 2-3 hours of work

### Recommendation for Your Case

**Aim for 15-20 questions**:
1. You currently have 50, but some are irrelevant
2. After removing data quality questions, you'll have ~45
3. Select the best 15-20 that:
   - Cover different aspects of customer behavior
   - Include varied question types (what, how, why)
   - Represent different complexity levels
   - Are most relevant to your use case

### Question Diversity

Aim for:
- **5-7 questions** about user behavior patterns
- **4-5 questions** about decision-making/choice architecture
- **3-4 questions** about user experience/interface behavior
- **2-3 questions** about behavioral economics/psychology
- **1-2 edge cases** or complex questions

### Bottom Line

**15-20 questions is a solid number**:
- Good statistical confidence
- Manageable curation effort
- Reasonable evaluation cost
- Covers diverse question types

**Quality over quantity**: Start with 15-20 well-curated questions rather than 50 with mixed quality.

---

## Agent Behavior Issue: Too Many Page Retrievals vs. Searches

### The Problem

In EVALUATION mode, the agent is doing:
- ✅ **3 searches** (correct - Phase 1)
- ❌ **39+ page retrievals** (too many - should do more searches instead)

**Expected behavior**:
- Phase 1: 3-5 broad searches
- Phase 2: 8-12 specific searches
- Retrieve 5-10 most relevant pages
- **Total: ~11-17 searches**, not 3 searches + 39 retrievals

### Why This Happens

The agent is interpreting instructions as:
1. Do 3 searches
2. Retrieve all pages from those search results

**Should be**:
1. Do 3-5 broad searches
2. Do 8-12 more specific searches
3. Retrieve only the most relevant pages

### The Issue

The instructions focus on **search count** but don't clearly limit **page retrievals**. The agent is prioritizing page retrieval over doing more searches, which is:
- ❌ Inefficient (too many API calls)
- ❌ Not what instructions intend
- ❌ Wastes tokens and time

### Solution

Update instructions to clarify:
- **Search strategy**: Do 11-17 searches total (3-5 broad + 8-12 specific)
- **Retrieval strategy**: Retrieve only the 5-10 most relevant pages, not all pages from search results
- **Balance**: Prioritize doing more searches over retrieving more pages
