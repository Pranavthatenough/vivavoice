# Prompts log

Every prompt I gave my AI agent while building VivaVoice, in order, with the problems I hit and how they were fixed. Numbers marked "Phase 6" come from the evaluation in `evals/`, and the full report is in `evals/RESULTS.md`.

## Phase 0: Setup

**Prompt:** Asked Claude for a step-by-step plan to build VivaVoice, a voice AI viva examiner, split into phases with a GitHub push after each.

**Result:** Set up the Python venv, Ollama, folder structure and the GitHub repo. Moved the project out of OneDrive, because syncing the venv folder causes problems.

## Phase 1: Notes → chunks → vector store

**Prompt:** Asked Claude to write a RAG pipeline in Python: read a PDF page by page, remove repeated headers and footers, split pages into ~120-word chunks with 25-word overlap, keep page numbers, embed the chunks with a local Ollama model, store them in ChromaDB, and add `ingest` and `search` commands.

**Problem:** bge-m3 failed to load with "unable to allocate CUDA_Host buffer" (out of memory).

**Fix at the time:** Switched to nomic-embed-text (~270 MB). Trade-off: weaker on Hindi. (Phase 6 found the real cause of the memory errors; see below.)

**Results:** The manual 5-question check wasn't recorded. Measured properly in Phase 6: the right page was the top result for **95%** of 20 questions worded differently from the notes (95% range 76–99%).

## Phase 2: Question generator + grader

**Prompt:** Asked Claude for a terminal examiner: generate one grounded viva question per chunk (5 chunks from different pages) with 2-4 key points, using structured JSON output from the local LLM; grade answers with a yes/no checklist per key point and compute the score in code; save each session as JSON.

**Model:** llama3.2:3b. llama3.1:8b was planned, but it was never downloaded and wouldn't fit the 4 GB GPU.

**Problem:** The GPU ran out of memory loading the model ("cudaMalloc failed"). **Fix:** restarted the PC to clear memory held by crashed runs.

**Manual check (8 questions across 2 vivas):** Grounded: 7 of 8. Fair grade: not recorded.

**Notes:** One question mixed two topics: it asked about the EMF of cells in parallel *and* the meter bridge's balance point, and needed a derivation the notes don't contain. Cause: the "Already asked" list in the prompt made the model blend topics. Another question was grounded but badly worded ("why a non-ohmic device breaks Ohm's law in different ways").

## Phase 3: Web app (v0.1)

**Prompt:** Asked Claude to turn the terminal examiner into a FastAPI web app with an answer-sheet design, and to fix questions mixing topics.

**What was built:**
- Quote-verified grounding: each key point needs a word-for-word quote from the notes; code checks the quote exists, and regenerates the question if not (max 3 tries).
- Removed the "Already asked" list, which caused topic mixing in Phase 2.
- Evidence highlighting in the notes excerpt (covered / missed).
- Prefetching the next question while the student answers.
- Result sheet with a per-page score heatmap.

**Results:** Not recorded at the time. Measured in Phase 6: **91.7%** of questions pass the quote check, and the median wait for the next question is **0.0 s**, while writing one takes 4.2 s.

## Phase 4: Voice (v0.2)

**Prompt:** Asked Claude to add voice: local speech-to-text, the examiner reading questions and feedback aloud, a hands-free mode, and measuring how answers are spoken.

**What was built:**
- faster-whisper (small, int8, CPU) with word timestamps and silence filtering.
- Browser recording that stops automatically after 2 s of silence, calibrated to room noise.
- Hands-free mode and barge-in (speaking cuts the examiner off).
- Delivery signals: time before speaking, filler words, long pauses, words per minute.
- Whisper prompt includes the question (for subject words) but never the key points (to keep grading fair).

**Problems:**
- av 19 removed an option faster-whisper uses. **Fix:** pinned av<19.
- With Whisper loaded too, the LLM failed with "unable to allocate CUDA_Host buffer", and a viva returned error 500 at question 4. **Fix:** retry LLM calls up to 3 times, keep the model loaded during a viva (keep_alive) so it isn't reloaded mid-viva, retry failed background question writing, and show a clear error without losing the student's answer.
- Restarting the server left the old model loaded in Ollama, so the new one ran out of GPU memory. **Fix:** unload leftover models at startup, load the LLM before Whisper, and unload on shutdown.

**Results (Phase 6 voice eval, 10 read-aloud sentences):** **8.8%** English word error rate at 0.76× real time on the laptop CPU. Full table in Phase 6.

## Phase 5: The smart examiner (v0.3)

**Prompt:** Asked Claude for follow-up questions, adaptive difficulty, examiner personas, Hinglish, and memory across vivas, all within a 16 GB laptop's limits.

**What was built:**
- Follow-ups with an answer-leak guard (code counts answer words in the follow-up; strict allows 0, friendly 2). Half credit for points recovered on follow-up.
- Three difficulty levels (recall / explain / apply) that adapt after each answer.
- Speculative prefetching: the next question is written at a guessed level and rewritten if the guess is wrong.
- Two personas that change tone, hints, starting level, patience and voice, but use an identical grading prompt (fairness).
- Hinglish: Whisper language "hi" with a code-mixed prompt; Hindi filler words counted.
- Memory: page mastery from past sessions steers which pages get asked.

**Design decision:** Did not switch to a multilingual embedding model, because embeddings only find passages and never read answers.

**Results:** Measured in Phase 6: the leak guard agreed with my blind labels on **60%** of follow-ups (95% range 42–75%).

## Phase 6: Evaluation (v0.4)

**Prompt:** Asked Claude for a rigorous evaluation harness that avoids an AI grading its own work, tests every design decision, and survives my laptop's memory crashes.

**What was built:**
- Retrieval eval: 20 reworded questions plus 5 that aren't in the notes; chunk sizes of 60, 120 and 250 words compared.
- A frozen 12-question exam set (4 per level), so prompt changes are compared on the same questions.
- Grading eval: 11 answer types built in code from the notes' quotes (so the right marks are known in advance), including keyword lists, negation and prompt injection; consistency re-grades; a checklist vs direct-score ablation.
- Follow-up eval for the leak guard, per examiner.
- Voice eval: read-aloud recordings transcribed 4 ways (with and without the question in the prompt, no prompt, Whisper base).
- A blind labelling page (the grader's verdicts are hidden), and a report with Cohen's kappa and 95% ranges for every number.
- Crash-safe caching and automatic prompt versioning, so before/after comparisons happen on their own.

**Results at a glance:**

| What | Result |
|---|---|
| Right page as the top search result | 95.0% (20 questions) |
| Questions passing the quote check | 91.7% (12 questions) |
| Questions a human judged answerable from the passage alone | 58.3% |
| Grader: key points judged correctly (built answers) | 73.0% |
| Grader: credit for points never said | 21.2% |
| Grader: agreement with my blind labels on real answers | 64.8%, kappa 0.37 (17 answers) |
| Answers made only of a prompt injection, blocked | 100% (12 of 12) |
| Leak guard agreement with my blind labels | 60.0% |
| Median wait for the next question | 0.0 s instead of 4.2 s |

**Voice eval (10 read-aloud sentences, 4 Whisper setups):**

| Setup | EN WER | EN terms | HI terms | x realtime |
|---|---|---|---|---|
| app (style prompt + question) | 8.8% | 83.3% | 57.1% | 0.76 |
| style prompt only | 9.9% | 83.3% | 42.9% | 0.71 |
| no prompt | 11.0% | 77.8% | 0.0% | 0.71 |
| Whisper base | 14.3% | 77.8% | 71.4% | 0.90 |

**Voice findings:** The Hinglish style prompt is essential (terms 0% → 57%). The question only helps with words it contains: "Kirchhoff" became "Pritchoff" because the question didn't mention it. Kept Whisper small, because base had a much higher English WER and wasn't faster. Filler numbers aren't reliable: there were only 4 fillers in the test.

**Grading: before and after (132 constructed answers, llama3.2:3b)**

| Version | Key points right | False credit | Pure injection blocked | Agreement with my labels |
|---|---|---|---|---|
| Checklist (Phase 2) | 49.4% | 38.9% | 8.3% | 54.5%, kappa −0.05 (7 answers) |
| Per-point (one key point per call, reason first) | 67.5% | 51.0% | 0% | 63.6%, kappa 0.13 (7 answers) |
| Hybrid (per-point + word-overlap guard in code) | 73.0% | 21.2% | 100% | 64.8%, kappa 0.37 (17 answers) |

**Grading findings:**
- The checklist design from Phase 2 was the main weakness: the 3B model lost track when judging several points at once, and marked even answers made of the notes' own sentences as wrong.
- Per-point grading fixed missed points, but made the grader too generous and fell for prompt injection despite instructions.
- A rule in plain code (a point needs at least 2 shared meaningful words) blocked every answer made only of an injection, because a small model can't be reliably prompted out of injection.
- **Not fixed:** an injection hidden inside a partly correct answer still earns extra credit (0 of 12 graded fully right), because the correct part shares words with the other points. Keyword lists and negated answers also still fool it, and 33% of real points are still missed.
- **Ablation:** asking for a 0-5 score directly beat the original checklist (average error 1.83 vs 2.25). The hybrid grader now beats direct scoring on scores within 1 mark (54.5% vs 47%).

**Problems found and fixed during evaluation:**
- **Root cause of all the memory errors:** Windows' page file was off, so the commit limit (RAM + page file) was only 15.4 GB. Free commit was 0.6 GB while 6.2 GB of RAM was physically free, so allocations failed (CUDA_Host, bad_alloc, mkl_malloc, OpenBLAS). Sorting processes by committed memory, not memory in use, showed the Claude desktop app's virtual machine (vmmem) reserving 4.1 GB. Diagnosed by comparing free RAM with free commit using `Get-CimInstance`; fixed by enabling the page file.
- The 3B model sometimes looped and wrote thousands of lines of JSON. **Fix:** lists capped at 4 items in the JSON schema (enforced by Ollama) and reply length capped, so a loop fails fast and is retried.
- When the LLM failed, the follow-up code fell back to a generic question, so the eval silently recorded model failures as leak-guard rejections. **Fix:** failed calls are marked and never saved; bad saved results are redone. This was a fallback that's helpful in the product but corrupted the evaluation.
- The GPU's 4 GB filled up. **Fix:** the LLM moves to the CPU automatically for the rest of the run.
- voice_eval crashed with "OpenBLAS: Memory allocation still failed". **Fix:** OpenBLAS limited to 1 thread, Ollama models unloaded before the voice eval, and the LLM unloaded after each LLM eval.
- Phase 2 vivas were saved without question numbers, so all their questions got the same label ID on the review page. **Fix:** fall back to the question's position.

## Phase 7: Launch (v1.0)

**Prompt:** Asked Claude to prepare the launch: a README built on the real results, one-command setup for Windows, notes for more subjects, and a launch kit (demo video plan, LinkedIn post, resume entry, interview prep).

**What was built:**
- README with the results, the before/after grading story, an architecture diagram, design decisions with their evidence, and honest limitations.
- `scripts/setup.ps1` (checks tools and the page file, installs everything, downloads models, loads the sample notes) and `scripts/start.ps1` (starts the app and opens the browser).
- Original sample notes for machine learning and cloud computing, so VivaVoice was tested on 3 subjects.
- The app now uses the hybrid grader by default, the winner of the Phase 6 evaluation.

**Problems:**
- Uploading a PDF failed with error 500 when the embedding model ran short of memory, and the failed upload left an empty copy of the notes behind. **Fix:** embedding retries and falls back to the CPU, and the notes are only replaced once everything has been embedded successfully.
- While writing the README, two claims weren't backed by the data: that 120-word chunks were the best size, and that computing the score in code beats direct scoring. Both were rewritten to match the results.