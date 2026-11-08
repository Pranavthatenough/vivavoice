# Prompts log
Every prompt I gave my AI agent while building VivaVoice, in order.
## Phase 0: Setup

**Prompt:** Asked Claude for a step-by-step plan to build VivaVoice, a voice AI viva examiner, split into phases with a GitHub push after each.

**Result:** Set up Python venv, Ollama, folder structure and the GitHub repo. Moved the project out of OneDrive because syncing the venv folder causes problems.

## Phase 1: Notes → chunks → vector store

**Prompt:** Asked Claude to write a RAG pipeline in Python: read a PDF page by page, remove repeated headers and footers, split pages into ~120-word chunks with 25-word overlap, keep page numbers, embed the chunks with a local Ollama model, store them in ChromaDB, and add `ingest` and `search` commands.

**Problem:** bge-m3 failed to load with "unable to allocate CUDA_Host buffer" (out of memory).

**Fix:** Switched to nomic-embed-text (~270 MB). Trade-off: weaker on Hindi, to revisit in Phase 5 (Hinglish).

**Retrieval test (top result):**

| Question | Expected page | Found page | Score |
|---|---|---|---|
| What is drift velocity? | 1 | | |
| Why does resistivity of semiconductors fall with temperature? | 2 | | |
| What is internal resistance of a cell? | 3 | | |
| When is a Wheatstone bridge balanced? | 4 | | |
| What is a kilowatt hour? | 5 | | |
| What is photosynthesis? (not in notes) | none | | |
## Phase 2: Question generator + grader

**Prompt:** Asked Claude for a terminal examiner: generate one grounded viva question per chunk (5 chunks from different pages) with 2-4 key points, using structured JSON output from the local LLM; grade answers with a yes/no checklist per key point and compute the score in code; save each session as JSON.

**Model:** llama3.1:8b (or llama3.2:3b if memory was the issue)

**Manual check (10 questions):** Grounded: _/10. Fair grade: _/10.

**Notes:** (anything odd you noticed)