# agents.md: Lab "First match ≠ right match"

Read this file first on every card and follow all its rules. Then read the card's section in `project_structure.md`.
Status: 27 Oct 2026, v3.
Repo: https://github.com/ayhandemir7-LSSBB/lab-relevance-scoring-rerankers (fork, work on `main`).

## 1. Goal
Fill in the instructor's starter notebook `relevance_scoring_rerankers.ipynb` (Steps 0 to 8) over two sources: `data/podcast.txt` and `data/eu_ai_act.pdf`.
Show that LLM relevance scoring and metadata filtering pick the *right* chunk, not just the *first* one. Every answer must cite the chunk it came from.

## 2. Stack
| Area | Choice |
|---|---|
| Language | Python, Jupyter notebooks in VS Code. Ayhan creates and activates the environment himself |
| LLM | OpenAI `gpt-4o-mini`, temperature 0 (answers and Step 3 relevance scores) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector store | Pinecone serverless, aws `us-east-1`, cosine, index `lab-rerankers` |
| Framework | `langchain-core`, `langchain-community` (`PyPDFLoader`, `TextLoader`), `langchain-text-splitters`, `langchain-openai`, `langchain-pinecone`, `pinecone` (not `pinecone-client`) |
| Other | `pypdf`, `python-dotenv` |
| Secrets | `.env`: `OPENAI_API_KEY`, `PINECONE_API_KEY` |

No Cohere, no sentence-transformers (Step 4 is parked). The `langchain-community` deprecation warning is expected; ignore it.

## 3. Setup
```bash
pip install -r requirements.txt    # inside Ayhan's own environment
```
`.env` sits next to the notebook. Data sits in `data/`. Run the lab notebook top to bottom from a clean kernel.

## 4. Repo map
```
lab-relevance-scoring-rerankers/
├── README.md, instructions.md, rubric.md     # from the instructor, never edit
├── agents.md                                 # this file
├── project_structure.md                      # cards, Musts, checks
├── relevance_scoring_rerankers.ipynb         # THE lab notebook (instructor template, Ayhan pastes cells in)
├── lab_proof.md                              # reviewer entry point (Ayhan finalises from lab_proof_draft.md)
├── requirements.txt
├── .env.example                              # key names only
├── .gitignore
├── data/podcast.txt, data/eu_ai_act.pdf
├── backups/                                  # git-ignored
└── workbench_<ID>.ipynb                      # git-ignored, one per card, Ayhan deletes before commit
```
`relevance_scoring_rerankers_reference.ipynb` (Claude's reference) may sit in the folder. Never read or copy it unless the prompt says so.

## 5. How a card runs
1. Ayhan names the card. Read this file and the card in `project_structure.md`.
2. Create **`workbench_<ID>.ipynb` only**. Never edit any other file directly.
3. Workbench pattern:
   1. **Folder check**: repo root, `data/` files exist (PASS/FAIL).
   2. **Backup cell**, only on cards L3, L5, L6: copy `relevance_scoring_rerankers.ipynb` to `backups/<ID>_relevance_scoring_rerankers.ipynb`.
   3. **Setup cell**: re-create what earlier steps built (imports, `load_dotenv`, chunks, `vectorstore` connected to the existing index) so the workbench runs alone. Never re-upload unless the card says so.
   4. **Build cells**, one per template TODO cell, in template order. First line of each: `# >>> PASTE INTO: Step <n> / <cell title>`.
   5. **Check cells**: each prints `PASS` or `FAIL` and one reason, matching the template's "Done when".
   6. **`%%writefile` cells**, only for plain files the card names (`requirements.txt`, `.gitignore`, `.env.example`, `lab_proof_draft.md`).
   7. **Undo cell** (L3, L5, L6 only): restores the backup.
4. Ayhan pastes the `PASTE INTO` cells into the matching TODO cells of the lab notebook. `%%writefile` never writes the notebook.
5. Codex writes cells; Ayhan runs them.
6. End with a short report: what was built, which cells go where, any risk.

## 6. Conventions
- Keep the template's PROVIDED cells, variable and function names exactly: `all_chunks`, `podcast_chunks`, `act_chunks`, `embeddings`, `pc`, `index_name`, `vectorstore`, `TEST_QUERIES`, `show_results`, `baseline_search`, `baseline_results`, `llm`, `llm_relevance_score`, `llm_rerank`, `format_context`, `rag_prompt`, `answer`.
- Build from the training notebooks, not from scratch: `rag_intro.ipynb` (embeddings, Pinecone, LCEL), `rag_pipeline.ipynb` (`rerank_documents` for Step 3, metadata filters), `rag_openai.ipynb` (upsert pattern).
- Beginner-friendly: short code, `#` comments, a markdown cell above each block saying what and why.
- Parameters: chunk 1000 / overlap 150, baseline `k=5`, Step 3 `k_initial=10`, `k_final=5`, `w_sim=0.3`, `w_llm=0.7`.
- Upload with `ids=[c.metadata["chunk_id"] for c in all_chunks]` so a re-run overwrites instead of duplicating.
- Pinecone metadata values can't be `None`: leave the key out.
- Step 5 metadata (v2): `section` (`recital` pages 1 to 43, `article` 44 to 123, `annex` 124 to 144; `transcript` for the podcast), `chapter`, `article` (e.g. `"Article 5"`), `annex`, podcast `char_start`.
- RAG prompt: answer only from the context, cite `[source, p.X]` or `[source, chunk_id]`, otherwise say "I don't know based on the sources".
- Manual evaluation cells (Step 7 table, "Stop and think") stay as `[Ayhan]` placeholders.
- No em dashes in markdown.

## 7. Definition of Done (every card)
- Workbench runs top to bottom from a clean kernel; every check prints PASS.
- The lab notebook still runs top to bottom after pasting.
- No keys in code or outputs; `.env` not tracked.
- Report delivered.

## 8. Never do
- Never run git commands, and never print them.
- Never edit files other than the workbench; never edit `README.md`, `instructions.md`, `rubric.md`.
- Never write the lab notebook with `%%writefile`, `nbformat` or any script.
- Never print, hardcode or log API keys.
- Never delete the Pinecone index except in card L5, where the brief requires the rebuild.
- Never change PROVIDED template cells, parameters or names in §6.
- Never add Cohere, sentence-transformers, new packages, features or files the card does not ask for.
- Never fill in manual verdicts or invent scores, quotes or results.
