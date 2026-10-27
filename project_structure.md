# project_structure.md: Lab "First match ≠ right match"

Module 3 track, Week 4 Day 4, required. Solo. Status: 27 Oct 2026, v3.
Repo: https://github.com/ayhandemir7-LSSBB/lab-relevance-scoring-rerankers (fork, `main`). Ayhan forks, clones and commits himself.

## 1. Brief in one paragraph
Fill in the instructor's starter notebook `relevance_scoring_rerankers.ipynb` (Steps 0 to 8) over a Trustworthy AI podcast transcript and the EU AI Act. Build a baseline vector search, improve it with LLM relevance scoring and metadata filtering, plug it into a RAG chain that cites sources, and judge with/without reranking by hand. Submit `lab_proof.md` as the reviewer's entry point via a PR.

## 2. Inputs
| File | What it is |
|---|---|
| `instructions.md`, `rubric.md`, `README.md` | Brief, 23-line grading table, submission rules |
| `relevance_scoring_rerankers.ipynb` | Instructor template: PROVIDED / YOUR TURN / OPTIONAL cells, "Done when" checks, `lab_proof.md` template |
| `data/podcast.txt` | One transcript, about 16k characters, no pages or episodes |
| `data/eu_ai_act.pdf` | Reg. 2024/1689, 144 pages: recitals p.1 to 43, Articles 1 to 113 p.44 to 123 (CHAPTER I to XIII), Annexes I to XIII p.124 to 144 |
| Training: `rag_intro.ipynb`, `rag_pipeline.ipynb`, `rag_openai.ipynb` | Code patterns to reuse |

## 3. Musts
| ID | Must | Template step | Rubric lines |
|---|---|---|---|
| M1 | Setup, both sources chunked with metadata, embeddings tested | Step 0, 1 | 17 |
| M2 | Baseline vector search on the 5 `TEST_QUERIES` | Step 2 | 18 |
| M3 | LLM relevance scoring + combined score + reorder | Step 3 (optional, in scope) | 3, 8, 12, 19 |
| M4 | Metadata filtering, incl. a new field and index rebuild | Step 5 | 1, 6, 21 |
| M5 | RAG chain with citations; baseline vs reranked is a 1 to 2 line swap | Step 6 | 6, 22 |
| M6 | Manual evaluation with vs without reranking, plus cost/time | Step 7 | 2, 7, 10, 11, 14, 15, 23 |
| M7 | `lab_proof.md`: query, evidence, answer, trace, baseline vs reranked, one limitation | Step 8 | Proof of Learning |
| M8 | Documented code, lab files only, no secrets | all | 16, submission |

Not covered on purpose: Step 4 dedicated reranker (rubric 4, 5, 9, 13, 20). Parked.

## 4. Pipeline
```
query
  ▼ metadata filter (WHERE)     e.g. section = article, source_type = regulation
  ▼ Pinecone top 10              text-embedding-3-small, cosine
  ▼ LLM relevance score 0 to 1   gpt-4o-mini, one call per chunk
  ▼ combined = 0.3 × similarity + 0.7 × LLM score → top 5 (WHICH)
  ▼ gpt-4o-mini, context only, cites [source, p.X]
answer + the chunks used
```

## 5. Cards
| ID | Card | Template step | Must | Backup |
|---|---|---|---|---|
| L0 | Setup | Step 0 | M8 | |
| L1 | Load, chunk with metadata, test embedding | Step 1 (1d) | M1 | |
| L2 | Pinecone index + baseline retrieval | Step 2 | M2 | |
| L3 | LLM relevance scoring | Step 3 | M3 | yes |
| L5 | Metadata filtering + v2 schema + index rebuild | Step 5 | M4 | yes |
| L6 | RAG pipeline with citations | Step 6 | M5 | yes |
| L7 | Evaluation with vs without reranking | Step 7 | M6 | |
| L8 | `lab_proof.md` + sanity check | Step 8 | M7, M8 | |

### L0 Setup (Step 0)
- **Creates:** `requirements.txt` (the template's pip list), `.env.example` (2 key names), `.gitignore` (`.env`, `workbench_*.ipynb`, `backups/`, `__pycache__/`, `.ipynb_checkpoints/`, `*_reference.ipynb`).
- **Manual (Ayhan):** copy the template notebook and `data/` files into the repo root, create the environment and `.env`.
- **Checks:** both keys load (True/False only) · both data files exist · all imports work.

### L1 Load, chunk, embed test (Step 1)
- **Fill:** 1d: `embeddings = OpenAIEmbeddings(model="text-embedding-3-small")`, embed `all_chunks[0]`.
- **Checks:** podcast and regulation chunk counts > 0 · every chunk has `source_type`, `source_name`, `chunk_id` · vector length 1536.

### L2 Baseline retrieval (Step 2)
- **Fill:** create `lab-rerankers` if missing (1536, cosine) · `PineconeVectorStore.from_documents(..., ids=chunk_ids)` · `baseline_search(query, k=5)` with `similarity_search_with_score` · store and print `baseline_results` for each test query.
- **Checks:** index vector count equals `len(all_chunks)` · 5 results per query, each with score, source and page/chunk_id.
- **Note down:** one query where the top result is wrong or half-relevant.

### L3 LLM relevance scoring (Step 3)
- **Fill:** `llm_relevance_score(query, doc)` from `rerank_documents` in `rag_pipeline.ipynb`, with a "reply with only a number" prompt and a safe float parse (fallback 0.5 as in the training code, clamp 0 to 1) · `llm_rerank(query, k_initial=10, k_final=5, w_sim=0.3, w_llm=0.7)` returns `(doc, combined)` · baseline vs LLM top 3 side by side for 2+ queries · count LLM calls.
- **Checks:** scores are floats in 0 to 1 · 5 results returned · reranked ids ⊂ baseline top 10 · at least one query where the order changed, or an honest "none".

### L5 Metadata filtering (Step 5)
- **Fill:** `filter={"source_type": {"$eq": ...}}` for regulation vs podcast on one query · show that `section` / `article` filters don't work yet · `chunk_with_metadata_v2` adds `section`, `chapter`, `article`, `annex` (regex on heading lines, carried forward chunk to chunk) and podcast `char_start` · delete the index, recreate it, re-upload, rebind `vectorstore` · filter on `{"article": {"$eq": "Article 5"}}` and `{"section": {"$eq": "article"}}`.
- **Checks:** every returned chunk matches its filter · 112+ of 113 articles tagged · no `None` metadata values · the new index count equals the new chunk count.
- **Known limit:** a chunk spanning two articles is tagged with the first heading.

### L6 RAG pipeline (Step 6)
- **Fill:** `format_context(results)` with `[source, p.X, Article N]` or `[source, chunk_id]` prefixes · `rag_prompt`: context only, cite, admit "I don't know based on the sources" · `answer(query, retrieve_fn=baseline_search)` returns `(answer_text, results)` · switching to `llm_rerank` is the only change · optional filtered retriever, e.g. regulation only.
- **Checks:** both methods return an answer with at least one citation · an out-of-scope question gets "I don't know".

### L7 Evaluation (Step 7)
- **Fill:** loop over `TEST_QUERIES`, time `answer(q, baseline_search)` vs `answer(q, llm_rerank)`, print both answers and the top 3 sources · fill the evaluation table skeleton with `[Ayhan]` cells · show the LLM-call count and seconds per query.
- **Checks:** one row per query and method · no verdict filled in by code.

### L8 lab_proof + sanity check (Step 8)
- **Creates:** `lab_proof_draft.md` from the template's 7 sections, filled with real outputs from one chosen query (evidence table, answer, trace); Ayhan reviews and renames to `lab_proof.md`.
- **Checks:** `.env` not tracked · no key patterns in tracked files or notebook outputs · only lab files in the repo · the lab notebook runs top to bottom.
- **Limitation candidates:** a chunk spanning two articles tagged wrong · the LLM score varies with prompt wording · the podcast paraphrases the 2019 guidelines, so a mixed-source answer can sound right but be unsupported by the Act.

## 6. Working rules (summary; full rules in `agents.md`)
**Stack:** OpenAI `gpt-4o-mini` + `text-embedding-3-small`, Pinecone `lab-rerankers`, LangChain (core, community, text-splitters, openai, pinecone), pypdf, python-dotenv. Codex in VS Code.

**Card prompt (written in Claude chat):**
```
Read agents.md first and follow all its rules. Card: <ID> <name> (<Must>).
Template step: <Step n>. Read: project_structure.md §<card>, relevance_scoring_rerankers.ipynb, <training notebook>.
Don't touch: every file except workbench_<ID>.ipynb.
Build the workbench, fill the TODO cells for this step, add PASS/FAIL checks. Don't commit.
End with a short report: what was built, which cells go where, any risk.
```

**Do:** reuse training and template code · keep template names and parameters · explain each step in markdown · cite sources in every answer · backups on L3, L5, L6.

**Never:** git commands · edit repo files directly · Cohere or sentence-transformers · new packages or features · keys in code or outputs · invented verdicts or scores.

**Definition of Done:** checks PASS · the lab notebook runs top to bottom from a clean kernel · no secrets · Ayhan commits.

## 7. Parked (won't do)
Step 4 dedicated reranker (Cohere or cross-encoder) · query rewriting, HyDE, sub-query, step-back · hybrid BM25 · score threshold instead of fixed k · demo UI.
