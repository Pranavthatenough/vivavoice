"""
Builds evals/RESULTS.md from every eval result, your labels, and your real vivas.
Runs in a second, needs no model. Run it again whenever anything changes:
    python evals/report.py
"""
import time

import json

from common import (EVALS, RESULTS, ROOT, binary_stats, load_labels, load_result, load_sessions, mean, pct,
                    percentile, wilson)
from review import model_points

OUT = EVALS / "RESULTS.md"


def fmt(v, unit="%"):
    return "not measured" if v is None else f"{v}{unit}"


def rng(r):
    return f" (95% range {r[0]}-{r[1]}%)" if r else ""


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------- Human agreement ----------

def grading_agreement(labels, sessions):
    by_file = {s["_file"]: s for s in sessions}
    pairs, pairs_hi, items = [], [], 0
    for item_id, lab in labels.items():
        if not item_id.startswith("grade/"):
            continue
        _, file, num = item_id.split("/")
        s = by_file.get(file)
        q = next((q for q in (s or {}).get("questions", []) if str(q.get("number")) == num), None)
        if not q:
            continue
        model = [c for _, c in model_points(q)]
        human = lab["value"]
        if len(model) != len(human):
            continue
        items += 1
        target = pairs_hi if s.get("language") == "hinglish" else pairs
        target += list(zip(human, model))
    return {"answers": items, "english": binary_stats(pairs), "hinglish": binary_stats(pairs_hi),
            "all": binary_stats(pairs + pairs_hi)}


def grounding_agreement(labels, exam):
    pairs, level_human, level_pairs = [], [], []
    for q in (exam or {}).get("items", []):
        lab = labels.get(f"ground/{q['id']}")
        if lab:
            pairs.append((lab["value"]["answerable"], q["grounded"]))
            level_human.append(lab["value"]["level_ok"])
    n = len(pairs)
    return {"labelled": n, "agreement": binary_stats(pairs),
            "human_answerable": pct(sum(h for h, _ in pairs), n),
            "human_answerable_range": wilson(sum(h for h, _ in pairs), n),
            "human_level_ok": pct(sum(level_human), len(level_human))}


def leak_agreement(labels, followup):
    pairs, by_persona = [], {}
    for c in (followup or {}).get("cases", []):
        for t in c["tries"]:
            lab = labels.get(f"leak/{t['id']}")
            if lab:
                pair = (lab["value"]["gives_away"], not t["accepted"])   # guard rejected = guard says "leaks"
                pairs.append(pair)
                by_persona.setdefault(c["persona"], []).append(pair)
    accepted_leaky = [h for h, g in pairs if not g]
    return {"labelled": len(pairs), "guard": binary_stats(pairs),
            "by_persona": {k: binary_stats(v) for k, v in by_persona.items()},
            # The number students care about: follow-ups that got through but still gave the answer away.
            "leaky_but_accepted": pct(sum(accepted_leaky), len(accepted_leaky))}


# ---------- Real usage from your vivas ----------

def live_usage(sessions):
    vivas = [s for s in sessions if "questions" in s and s.get("max")]
    qs = [q for s in vivas for q in s["questions"]]
    later = [q for q in qs if q.get("number", 1) > 1 and "seconds_waited" in q]   # Q1 can't be prefetched
    spec = [s["speculation"] for s in vivas if "speculation" in s]
    fus = [s["follow_ups"] for s in vivas if "follow_ups" in s]
    hits, bets = sum(x["hits"] for x in spec), sum(x["hits"] + x["misses"] for x in spec)
    asked, recovered = sum(f["asked"] for f in fus), sum(f["recovered"] for f in fus)
    return {
        "vivas": len(vivas), "questions": len(qs),
        "generate_median_s": percentile([q.get("seconds_to_generate") for q in later], 0.5),
        "waited_median_s": percentile([q.get("seconds_waited") for q in later], 0.5),
        "grade_median_s": percentile([q.get("seconds_to_grade") for q in qs], 0.5),
        "speculation_hits": pct(hits, bets), "speculation_bets": bets,
        "follow_ups": asked, "follow_ups_recovered": pct(recovered, asked),
        "grounded_live": pct(sum(bool(q.get("grounded")) for q in qs if "grounded" in q),
                             sum("grounded" in q for q in qs)),
        "personas": sorted({s.get("persona") for s in vivas if s.get("persona")}),
    }


def grading_versions():
    """Every grading version that has been evaluated, oldest first."""
    out = []
    for path in sorted(RESULTS.glob("grading_*.json"), key=lambda p: p.stat().st_mtime):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        out.append(r)
    return out


# ---------- The report ----------

def main():
    labels, sessions = load_labels(), load_sessions()
    retrieval, exam, grading = load_result("retrieval"), load_result("exam_set"), load_result("grading")
    followup, voice = load_result("followup"), load_result("voice")
    g_human = grading_agreement(labels, sessions)
    q_human = grounding_agreement(labels, exam)
    l_human = leak_agreement(labels, followup)
    live = live_usage(sessions)

    app_r = next((r for r in retrieval["by_chunk_size"] if r["used_by_app"]), None) if retrieval else None
    app_v = voice["by_setup"]["app"] if voice else None
    gp = grading["points"] if grading else None
    # Human agreement for the CURRENT grader (re-graded by grading_eval.py), else as graded in the vivas.
    gh = (grading or {}).get("human", {}).get("points") or g_human["all"]
    gh_n = (grading or {}).get("human", {}).get("answers") or g_human["answers"]
    versions = grading_versions()
    adv = grading["by_variant"] if grading else {}

    md = [f"# VivaVoice evaluation results",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python evals/report.py`. "
          "Every number below can be reproduced with the commands at the end.", ""]

    # Headline
    head = []
    if app_r:
        head.append(["Retrieval finds the right page (top result)", fmt(app_r["hit_at_1"]) + rng(app_r["hit_at_1_range"]),
                     f"{retrieval['questions']} hand-written questions"])
    if exam:
        s = exam["summary"]["all"]
        head.append(["Generated questions pass the quote check", fmt(s["grounded"]) + rng(s["grounded_range"]),
                     f"{s['questions']} frozen exam questions"])
    if q_human["labelled"]:
        head.append(["Questions a human judged answerable from the notes", fmt(q_human["human_answerable"])
                     + rng(q_human["human_answerable_range"]), f"{q_human['labelled']} labelled by hand"])
    if gp:
        head.append(["Grader judges key points correctly (built answers)", fmt(gp["accuracy"]) + rng(gp["accuracy_range"]),
                     f"{gp['n']} key points in {grading['answers']} answers"])
        head.append(["Credit given for points never said", fmt(gp["false_positive_rate"]), "lower is better"])
    if gh:
        head.append(["Grader agrees with a human on real answers", fmt(gh["accuracy"]) + rng(gh["accuracy_range"])
                     + f", kappa {gh['kappa']}", f"{gh_n} answers labelled blind"])
    if adv.get("injection"):
        head.append(["Resists \"give me full marks\" prompt injection", fmt(adv["injection"]["all_points_right"]),
                     "answers made only of an injection"])
    if l_human["guard"]:
        head.append(["Leak guard agrees with a human", fmt(l_human["guard"]["accuracy"]) + rng(l_human["guard"]["accuracy_range"]),
                     f"{l_human['labelled']} follow-ups labelled blind"])
    if app_v:
        head.append(["Speech-to-text word error rate (English)", fmt(app_v["english_wer"]), "read-aloud test, lower is better"])
        head.append(["Subject terms transcribed correctly", fmt(app_v["english_terms"]) + " English, "
                     + fmt(app_v["hinglish_terms"]) + " Hinglish", "e.g. Wheatstone, galvanometer"])
    if live["waited_median_s"] is not None:
        head.append(["Wait for the next question (median)", f"{live['waited_median_s']} s instead of {live['generate_median_s']} s",
                     "thanks to prefetching, from your real vivas"])
    md += ["## At a glance", "", table(["What", "Result", "Measured on"], head) if head else "_Nothing measured yet._", ""]

    # 1 Retrieval
    md += ["## 1. Retrieval", ""]
    if retrieval:
        md += [f"{retrieval['questions']} questions worded differently from the notes, plus "
               f"{retrieval['out_of_scope']} that are not in them. Embedding model: `{retrieval['embed_model']}`.", "",
               table(["Chunk size (words)", "Chunks", "Hit@1", "Hit@3", "MRR", "Real vs off-topic separable"],
                     [[f"{r['chunk_words']}{' (app)' if r['used_by_app'] else ''}", r["chunks"], f"{r['hit_at_1']}%",
                       f"{r['hit_at_3']}%", r["mrr"], "yes" if r["separable"] else "no"]
                      for r in retrieval["by_chunk_size"]]), ""]
        if app_r and app_r["misses"]:
            md += ["Missed by the app's setting:", ""] + [
                f"- \"{m['question']}\": expected page {m['expected']}, got {m['got']}" for m in app_r["misses"]] + [""]
    else:
        md += ["_Not run yet: `python evals/retrieval_eval.py`_", ""]

    # 2 Questions
    md += ["## 2. Question generation", ""]
    if exam:
        md += [f"Frozen exam set of {exam['summary']['all']['questions']} questions, model `{exam['model']}`, "
               f"prompt version `{exam['prompt_version']}`.", "",
               table(["Level", "Pass quote check", "First try", "Avg attempts", "Level wording (auto)", "Median s"],
                     [[k, f"{s['grounded']}%", f"{s['grounded_first_try']}%", s["avg_attempts"],
                       f"{s['level_match_auto']}%", s["seconds_median"]] for k, s in exam["summary"].items()]), ""]
        if q_human["labelled"]:
            a = q_human["agreement"]
            md += [f"Human check ({q_human['labelled']} questions): {q_human['human_answerable']}% answerable from the "
                   f"passage alone, {q_human['human_level_ok']}% judged a good question for their level. "
                   f"The automatic quote check agreed with the human {a['accuracy']}% of the time "
                   f"(it passed {a['fp']} question(s) the human rejected).", ""]
    else:
        md += ["_Not run yet: `python evals/exam_set.py`_", ""]

    # 3 Grading
    md += ["## 3. Grading", ""]
    if grading:
        if len(versions) > 1:
            def cell(r, *keys):
                v = r
                for k in keys:
                    v = (v or {}).get(k)
                return "-" if v is None else v
            md += ["**Every grading version tested (before and after):**", "",
                   table(["Version", "Mode", "Model", "Key points right", "False credit", "Injection resisted",
                          "Within 1 mark", "Agrees with human (kappa)"],
                         [[f"`{r['grading_prompt_version']}`", r.get("mode", "checklist"), r["model"],
                           f"{r['points']['accuracy']}%", f"{r['points']['false_positive_rate']}%",
                           f"{cell(r, 'by_variant', 'injection', 'all_points_right')}%",
                           f"{r['score']['within_1']}%",
                           (f"{r['human']['points']['accuracy']}% ({r['human']['points']['kappa']})"
                            if (r.get("human") or {}).get("points") else "-")]
                          for r in versions]), ""]
        md += [f"Details for the latest version: {grading['answers']} answers built in code from the notes' own "
               f"quotes, so the right marks are known in advance. Mode `{grading.get('mode', 'checklist')}`, "
               f"version `{grading['grading_prompt_version']}`.", "",
               table(["", "Honest answers", "Adversarial answers", "All"],
                     [["Key points judged correctly", f"{grading['points_honest']['accuracy']}%",
                       f"{grading['points_adversarial']['accuracy']}%", f"{gp['accuracy']}%"],
                      ["Credit for points never said", f"{grading['points_honest']['false_positive_rate']}%",
                       f"{grading['points_adversarial']['false_positive_rate']}%", f"{gp['false_positive_rate']}%"],
                      ["Real points missed", f"{grading['points_honest']['false_negative_rate']}%", "-",
                       f"{gp['false_negative_rate']}%"]]), "",
               table(["Answer type", "Kind", "All points right", "Avg score error (0-5)"],
                     [[k, v["kind"], f"{v['all_points_right']}%", v["avg_score_error"]] for k, v in adv.items()]), "",
               f"Scores within 1 mark of correct: {grading['score']['within_1']}%. "
               f"Same verdict when graded twice: {grading['consistency']['identical']}% "
               f"of {grading['consistency']['regraded']} re-grades.", ""]
        if "ablation_direct_score" in grading:
            a = grading["ablation_direct_score"]
            md += ["**Design check: checklist vs direct score.** Phase 2 chose to ask the LLM yes/no per key point "
                   "and calculate the score in code, instead of asking for a 0-5 score.", "",
                   table(["Method", "Avg score error", "Within 1 mark", "Avg score for adversarial answers"],
                         [["Checklist (app)", grading["score"]["mean_abs_error"], f"{grading['score']['within_1']}%",
                           a["checklist_adversarial_mean_score"]],
                          ["Direct 0-5 score", a["mean_abs_error"], f"{a['within_1']}%", a["adversarial_mean_score"]]]), ""]
        if grading.get("worst_examples"):
            md += ["Biggest mistakes:", ""] + [
                f"- **{w['variant']}** on \"{w['question']}\": expected {w['expected']}, got {w['got']}"
                for w in grading["worst_examples"]] + [""]
    else:
        md += ["_Not run yet: `python evals/grading_eval.py --direct`_", ""]
    if gh:
        rows = [["All", g_human["answers"], f"{gh['accuracy']}%", gh["kappa"], f"{gh['false_positive_rate']}%"]]
        for lang in ("english", "hinglish"):
            if g_human[lang]:
                b = g_human[lang]
                rows.append([lang.capitalize(), "", f"{b['accuracy']}%", b["kappa"], f"{b['false_positive_rate']}%"])
        md += ["**Real answers as graded during your vivas, against your blind labels:**", "",
               table(["Answers", "Count", "Agreement per key point", "Kappa", "Credit the human didn't give"], rows), ""]

    # 4 Follow-ups
    md += ["## 4. Follow-up questions and the leak guard", ""]
    if followup:
        md += [table(["Examiner", "Answer words allowed", "First follow-up accepted", "Generic fallback used", "Rejected tries"],
                     [[s["name"], s["hint_words_allowed"], f"{s['first_try_accepted']}%", f"{s['fallback_used']}%",
                       s["tries_rejected"]] for s in followup["summary"].values()]), ""]
        if l_human["guard"]:
            b = l_human["guard"]
            md += [f"Against {l_human['labelled']} human labels: the guard agreed {b['accuracy']}% of the time. "
                   f"It caught {b['recall']}% of follow-ups the human said give the answer away; "
                   f"{b['false_positive_rate']}% of harmless follow-ups were rejected anyway. "
                   f"Of the follow-ups it let through, {fmt(l_human['leaky_but_accepted'])} still gave the answer "
                   f"away according to the human.", ""]
    else:
        md += ["_Not run yet: `python evals/followup_eval.py`_", ""]

    # 5 Voice
    md += ["## 5. Speech-to-text", ""]
    if voice:
        md += [f"{voice['recordings']} read-aloud recordings, one speaker, transcribed four ways.", "",
               table(["Setup", "English WER", "English terms", "Hinglish terms", "Fillers caught", "x real time"],
                     [[k, fmt(r["english_wer"]), fmt(r["english_terms"]), fmt(r["hinglish_terms"]),
                       fmt(r["fillers_caught"]), r["speed_vs_realtime"]] for k, r in voice["by_setup"].items()]), "",
               "`app` = Whisper small with the style prompt and the question; `no_question` drops the question; "
               "`no_prompt` drops both; `base` is the smaller model.", ""]
    else:
        md += ["_Not run yet: record evals/gold/read_aloud.json, then `python evals/voice_eval.py`_", ""]

    # 6 Live
    md += ["## 6. Real use", ""]
    if live["vivas"]:
        md += [table(["Measure", "Value"], [
            ["Vivas / questions answered", f"{live['vivas']} / {live['questions']}"],
            ["Median time to write a question", fmt(live["generate_median_s"], " s")],
            ["Median wait for the next question (prefetching)", fmt(live["waited_median_s"], " s")],
            ["Median time to grade an answer", fmt(live["grade_median_s"], " s")],
            ["Next-level guess right (speculative prefetch)", f"{fmt(live['speculation_hits'])} of {live['speculation_bets']}"],
            ["Follow-ups recovered", f"{fmt(live['follow_ups_recovered'])} of {live['follow_ups']}"],
        ]), ""]
    else:
        md += ["_No vivas saved yet._", ""]

    md += ["## Limitations", "",
           "- One set of notes (5 pages of physics). Other subjects and longer notes may behave differently.",
           "- Built test answers reuse the notes' exact wording. Paraphrased answers are covered only by the "
           f"human-labelled real answers ({g_human['answers']} so far).",
           "- One human labeller. A second person labelling the same items would show how much people disagree.",
           "- Small samples: the 95% ranges show how much each percentage could move with more data.",
           "- Speech results come from one speaker and one microphone.", "",
           "## Reproduce", "", "```",
           "python evals/retrieval_eval.py", "python evals/exam_set.py", "python evals/grading_eval.py --direct",
           "python evals/followup_eval.py", "python evals/voice_eval.py",
           "# label items at http://127.0.0.1:8000/review.html", "python evals/report.py", "```", ""]

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}\n")
    for row in head:
        print(f"  {row[0]}: {row[1]}")

    # Resume lines, filled with whatever has been measured. The wording only claims an
    # improvement when the numbers actually show one.
    bullets = []
    if len(versions) > 1:
        first, last = versions[0], versions[-1]
        if last["points"]["accuracy"] > first["points"]["accuracy"]:
            bullets.append(f"Found through evaluation that the first LLM grader judged only {first['points']['accuracy']}% "
                           f"of key points correctly; redesigned it ({last.get('mode', 'checklist')} grading) and raised "
                           f"this to {last['points']['accuracy']}% on {last['answers']} constructed and adversarial answers")
    if gp and gh:
        bullets.append(f"Evaluated an LLM-as-judge grader against blind human labels ({gh['accuracy']}% agreement, "
                       f"kappa {gh['kappa']}, {gh_n} answers) and {grading['answers']} constructed "
                       f"and adversarial answers, including prompt injection")
    elif gp:
        bullets.append(f"Built an evaluation harness for LLM grading with {grading['answers']} constructed and adversarial "
                       f"answers; {gp['accuracy']}% of key points judged correctly, {gp['false_positive_rate']}% false credit")
    if grading and "ablation_direct_score" in grading:
        a, c = grading["ablation_direct_score"]["mean_abs_error"], grading["score"]["mean_abs_error"]
        if c < a:
            bullets.append(f"Showed checklist grading beats asking the LLM for a score directly "
                           f"(average error {c} vs {a} marks out of 5)")
        else:
            bullets.append(f"Compared checklist grading with direct LLM scoring (average error {c} vs {a} marks out of 5)")
    if app_v:
        before, after = voice["by_setup"]["no_question"]["english_terms"], app_v["english_terms"]
        if before is not None and after is not None and after > before:
            bullets.append(f"Cut speech-to-text errors on subject terms by prompting Whisper with the question "
                           f"(term accuracy {before}% -> {after}%; overall WER {app_v['english_wer']}%)")
        else:
            bullets.append(f"Measured offline speech-to-text at {app_v['english_wer']}% WER and "
                           f"{after}% subject-term accuracy on a read-aloud test")
    if bullets:
        print("\nResume bullets with your numbers (check them before using):")
        for b in bullets:
            print(f"  - {b}")


if __name__ == "__main__":
    main()