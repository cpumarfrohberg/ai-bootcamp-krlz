# Wikipedia Agent Evaluation Strategy (Minimum Viable)

## Core Components

1. **Functional Tests** - Verify correctness (✅ already implemented)
2. **Ground Truth** - Questions with expected sources
3. **Source Metrics** - Hit rate, MRR
4. **Judge** - LLM evaluates answer quality
5. **Combined Score** - Single performance metric

---

## 1. Functional Tests

**Status**: ✅ Already implemented in `tests/rag_agent/test_agent.py`

**Tests**:
- Tool invocation (wikipedia_search called)
- Tool coordination (both tools used)
- Output structure (sources included)

---

## 2. Ground Truth

**Structure**:
```json
{
  "question": "What factors influence customer behavior?",
  "expected_sources": ["Consumer behaviour", "Behavioral economics"]
}
```

**Requirements**:
- 10-20 manually curated questions
- Focus on customer behavior topics
- Include expected Wikipedia sources

---

## 3. Source Metrics

### Hit Rate
- Percentage of questions where at least one expected source is found
- Range: 0.0 to 1.0
- Formula: `(questions with expected source found) / (total questions)`

### MRR (Mean Reciprocal Rank)
- Average of 1/rank of first expected source found
- Range: 0.0 to 1.0
- Formula: `average(1/rank_of_first_expected_source)` or 0.0 if not found

---

## 4. LLM-as-a-Judge

**Purpose**: Evaluate answer quality (accuracy, completeness, relevance)

**Judge Output**:
```json
{
  "overall_score": 0.85,
  "accuracy": 0.90,
  "completeness": 0.80,
  "relevance": 0.90,
  "reasoning": "Brief explanation"
}
```

**Implementation**:
- Model: `DEFAULT_JUDGE_MODEL` (gpt-4o-mini)
- Temperature: 0.1 (for consistency)
- Output: Structured JSON

---

## 5. Combined Score

**Formula**: `score = (hit_rate^2.0 * judge_score^1.5) / (num_tokens/1000)^0.5`

**Parameters**:
- `alpha = 2.0` (prioritizes hit rate)
- `beta = 0.5` (penalizes tokens)
- `gamma = 1.5` (incorporates judge score)

---

## Evaluation Workflow

```
1. Load ground truth
2. For each question:
   a. Run agent query
   b. Calculate source hit rate & MRR
   c. Run judge evaluation
   d. Count tokens
3. Aggregate metrics
4. Calculate combined score
5. Save results (CSV + metadata JSON)
```

---

## Output Format

**CSV** (`evals/results/evaluation.csv`):
```csv
question,hit_rate,mrr,judge_score,num_tokens,combined_score
"What factors influence customer behavior?",1.0,1.0,0.85,2500,1.42
```

**Metadata JSON** (`evals/results/evaluation.metadata.json`):
```json
{
  "timestamp": "2025-01-XX...",
  "model": "gpt-4o-mini",
  "judge_model": "gpt-4o-mini",
  "num_questions": 20,
  "avg_hit_rate": 0.95,
  "avg_mrr": 0.88,
  "avg_judge_score": 0.83,
  "avg_combined_score": 1.35
}
```

---

## Success Criteria

**Minimum**:
- Hit rate > 0.80
- MRR > 0.70
- Judge score > 0.75

**Target**:
- Hit rate > 0.90
- MRR > 0.85
- Judge score > 0.85
- Combined score > 1.20

---

## Implementation Checklist

- [ ] Create ground truth dataset (10-20 questions)
- [ ] Implement source hit rate calculation
- [ ] Implement source MRR calculation
- [ ] Implement judge evaluation function
- [ ] Implement token counting
- [ ] Implement combined score calculation
- [ ] Create evaluation runner
- [ ] Save results (CSV + metadata JSON)
