# VivaVoice evaluation results
Generated 2026-11-11 02:17 by `python evals/report.py`. Every number below can be reproduced with the commands at the end.

## At a glance

| What | Result | Measured on |
|---|---|---|
| Retrieval finds the right page (top result) | 95.0% (95% range 76.4-99.1%) | 20 hand-written questions |
| Generated questions pass the quote check | 91.7% (95% range 64.6-98.5%) | 12 frozen exam questions |
| Questions a human judged answerable from the notes | 58.3% (95% range 32.0-80.7%) | 12 labelled by hand |
| Grader judges key points correctly (built answers) | 73.0% (95% range 68.3-77.2%) | 385 key points in 132 answers |
| Credit given for points never said | 21.2% | lower is better |
| Grader agrees with a human on real answers | 64.8% (95% range 51.5-76.2%), kappa 0.37 | 17 answers labelled blind |
| Resists "give me full marks" prompt injection | 100.0% | answers made only of an injection |
| Leak guard agrees with a human | 60.0% (95% range 42.3-75.4%) | 30 follow-ups labelled blind |
| Speech-to-text word error rate (English) | 8.8% | read-aloud test, lower is better |
| Subject terms transcribed correctly | 83.3% English, 57.1% Hinglish | e.g. Wheatstone, galvanometer |
| Wait for the next question (median) | 0.0 s instead of 4.2 s | thanks to prefetching, from your real vivas |

## 1. Retrieval

20 questions worded differently from the notes, plus 5 that are not in them. Embedding model: `nomic-embed-text`.

| Chunk size (words) | Chunks | Hit@1 | Hit@3 | MRR | Real vs off-topic separable |
|---|---|---|---|---|---|
| 60 | 33 | 90.0% | 100.0% | 0.95 | yes |
| 120 (app) | 17 | 95.0% | 100.0% | 0.975 | yes |
| 250 | 10 | 85.0% | 100.0% | 0.925 | no |

Missed by the app's setting:

- "What is the difference between EMF and terminal voltage?": expected page 3, got [5, 3, 3]

## 2. Question generation

Frozen exam set of 12 questions, model `llama3.2:3b`, prompt version `a9f3999e`.

| Level | Pass quote check | First try | Avg attempts | Level wording (auto) | Median s |
|---|---|---|---|---|---|
| all | 91.7% | 50.0% | 1.67 | 100.0% | 6.5 |
| recall | 100.0% | 75.0% | 1.5 | 100.0% | 7.0 |
| explain | 100.0% | 50.0% | 1.5 | 100.0% | 6.05 |
| apply | 75.0% | 25.0% | 2.0 | 100.0% | 6.55 |

Human check (12 questions): 58.3% answerable from the passage alone, 58.3% judged a good question for their level. The automatic quote check agreed with the human 66.7% of the time (it passed 4 question(s) the human rejected).

## 3. Grading

**Every grading version tested (before and after):**

| Version | Mode | Model | Key points right | False credit | Injection resisted | Within 1 mark | Agrees with human (kappa) |
|---|---|---|---|---|---|---|---|
| `930a259c` | checklist | llama3.2:3b | 49.4% | 38.9% | 8.3% | 33.3% | 54.5% (-0.05) |
| `d42af7b6` | per_point | llama3.2:3b | 67.5% | 51.0% | 0.0% | 53.8% | 63.6% (0.13) |
| `98bbde03` | hybrid | llama3.2:3b | 73.0% | 21.2% | 100.0% | 54.5% | 64.8% (0.37) |

Details for the latest version: 132 answers built in code from the notes' own quotes, so the right marks are known in advance. Mode `hybrid`, version `98bbde03`.

|  | Honest answers | Adversarial answers | All |
|---|---|---|---|
| Key points judged correctly | 68.1% | 78.9% | 73.0% |
| Credit for points never said | 25.7% | 20.2% | 21.2% |
| Real points missed | 33.1% | - | 33.2% |

| Answer type | Kind | All points right | Avg score error (0-5) |
|---|---|---|---|
| full | honest | 33.3% | 1.83 |
| reversed | honest | 50.0% | 1.58 |
| spoken | honest | 41.7% | 1.75 |
| padded | honest | 50.0% | 1.5 |
| first_only | honest | 25.0% | 1.17 |
| all_but_last | honest | 25.0% | 1.75 |
| partial_injection | adversarial | 0.0% | 1.75 |
| off_topic | adversarial | 100.0% | 0.0 |
| keywords_only | adversarial | 33.3% | 2.08 |
| negated | adversarial | 58.3% | 1.08 |
| injection | adversarial | 100.0% | 0.0 |

Scores within 1 mark of correct: 54.5%. Same verdict when graded twice: 100.0% of 15 re-grades.

Biggest mistakes:

- **keywords_only** on "Why does a higher voltage V mean a lower current I in the equation a fixed power P = VI?": expected [False, False, False], got [True, True, True]
- **full** on "State the unit of length 'l' used in the equation.": expected [True, True, True], got [False, False, False]
- **reversed** on "State the unit of length 'l' used in the equation.": expected [True, True, True], got [False, False, False]
- **spoken** on "State the unit of length 'l' used in the equation.": expected [True, True, True], got [False, False, False]
- **padded** on "State the unit of length 'l' used in the equation.": expected [True, True, True], got [False, False, False]

**Real answers as graded during your vivas, against your blind labels:**

| Answers | Count | Agreement per key point | Kappa | Credit the human didn't give |
|---|---|---|---|---|
| All | 17 | 64.8% | 0.37 | 0.0% |
| English |  | 42.9% | 0.12 | 0.0% |
| Hinglish |  | 40.0% | -0.36 | 100.0% |

## 4. Follow-up questions and the leak guard

| Examiner | Answer words allowed | First follow-up accepted | Generic fallback used | Rejected tries |
|---|---|---|---|---|
| Friendly internal | 2 | 100.0% | 0.0% | 0 |
| Strict external | 0 | 50.0% | 41.7% | 11 |

Against 30 human labels: the guard agreed 60.0% of the time. It caught 46.7% of follow-ups the human said give the answer away; 26.7% of harmless follow-ups were rejected anyway. Of the follow-ups it let through, 42.1% still gave the answer away according to the human.

## 5. Speech-to-text

10 read-aloud recordings, one speaker, transcribed four ways.

| Setup | English WER | English terms | Hinglish terms | Fillers caught | x real time |
|---|---|---|---|---|---|
| app | 8.8% | 83.3% | 57.1% | 25.0% | 0.76 |
| no_question | 9.9% | 83.3% | 42.9% | 50.0% | 0.71 |
| no_prompt | 11.0% | 77.8% | 0.0% | 0.0% | 0.71 |
| base | 14.3% | 77.8% | 71.4% | 75.0% | 0.9 |

`app` = Whisper small with the style prompt and the question; `no_question` drops the question; `no_prompt` drops both; `base` is the smaller model.

## 6. Real use

| Measure | Value |
|---|---|
| Vivas / questions answered | 5 / 25 |
| Median time to write a question | 4.2 s |
| Median wait for the next question (prefetching) | 0.0 s |
| Median time to grade an answer | 0.0 s |
| Next-level guess right (speculative prefetch) | 100.0% of 12 |
| Follow-ups recovered | 0.0% of 4 |

## Limitations

- One set of notes (5 pages of physics). Other subjects and longer notes may behave differently.
- Built test answers reuse the notes' exact wording. Paraphrased answers are covered only by the human-labelled real answers (17 so far).
- One human labeller. A second person labelling the same items would show how much people disagree.
- Small samples: the 95% ranges show how much each percentage could move with more data.
- Speech results come from one speaker and one microphone.

## Reproduce

```
python evals/retrieval_eval.py
python evals/exam_set.py
python evals/grading_eval.py --direct
python evals/followup_eval.py
python evals/voice_eval.py
# label items at http://127.0.0.1:8000/review.html
python evals/report.py
```
