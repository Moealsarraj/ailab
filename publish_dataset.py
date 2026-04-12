"""Publish Arabic Bench dataset to HuggingFace Datasets."""
import json
from huggingface_hub import HfApi, DatasetCard

# Import dataset
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.tools.arabic_bench.dataset import DATASET, CATEGORIES

def main():
    api = HfApi()
    repo_id = "Moealsarraj/arabic-bench-dataset"

    # Create dataset repo
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Dataset repo ready: {repo_id}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Convert to JSONL format
    jsonl_lines = []
    for case in DATASET:
        jsonl_lines.append(json.dumps(case, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)

    # Upload JSONL data file
    api.upload_file(
        path_or_fileobj=jsonl_content.encode("utf-8"),
        path_in_repo="data/test_cases.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {len(DATASET)} test cases")

    # Upload categories
    cat_content = json.dumps(CATEGORIES, ensure_ascii=False, indent=2)
    api.upload_file(
        path_or_fileobj=cat_content.encode("utf-8"),
        path_in_repo="data/categories.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Uploaded categories")

    # Create README
    readme = f"""---
language:
- ar
- en
license: mit
task_categories:
- text-generation
- translation
- summarization
- question-answering
tags:
- arabic
- nlp
- evaluation
- benchmark
- msa
- dialect
size_categories:
- n<1K
---

# Arabic Bench Dataset

A curated evaluation dataset for benchmarking AI models on Arabic language tasks.

## Overview

- **{len(DATASET)} test cases** across **{len(CATEGORIES)} categories**
- Each case includes a prompt, gold-standard reference answer, and a deliberately imperfect AI response
- Covers Modern Standard Arabic (MSA) and multiple Arabic dialects
- Designed for evaluating: translation, summarization, Q&A, creative writing, grammar, dialect understanding, legal/formal, and medical/scientific tasks

## Categories

| Category | Arabic | Count |
|----------|--------|-------|
"""
    from app.tools.arabic_bench.dataset import DATASET_BY_CATEGORY
    for cat in CATEGORIES:
        count = len(DATASET_BY_CATEGORY.get(cat["id"], []))
        readme += f"| {cat['name']} | {cat['name_ar']} | {count} |\n"

    readme += f"""
## Schema

Each test case contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., "tr-01") |
| `category` | string | Category ID |
| `difficulty` | string | easy, medium, or hard |
| `title` | string | English title |
| `title_ar` | string | Arabic title |
| `prompt` | string | The task prompt |
| `reference` | string | Gold-standard reference answer |
| `ai_input` | string | Deliberately imperfect AI response for evaluation |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Moealsarraj/arabic-bench-dataset", split="train")
```

Or load directly:
```python
import json
with open("data/test_cases.jsonl") as f:
    cases = [json.loads(line) for line in f]
```

## Live Tool

Try the interactive evaluation tool: [Arabic Bench on HuggingFace Spaces](https://huggingface.co/spaces/Moealsarraj/ailab)

## Author

Mohammed AL Sarraj — AI Engineer
"""

    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Dataset published: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
