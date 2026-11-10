// VivaVoice review page: collect blind human labels for the evaluation.

const $ = (id) => document.getElementById(id);
const TABS = [
  { type: "grading", name: "Grading", empty: "No answers to review yet. Finish a viva first; its answers appear here." },
  { type: "grounding", name: "Questions", empty: "Run python evals/exam_set.py first; its questions appear here." },
  { type: "leak", name: "Follow-ups", empty: "Run python evals/followup_eval.py first; its follow-ups appear here." },
];
const state = { items: [], labels: {}, tab: "grading", index: 0, draft: null };

function escapeHTML(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const tabItems = () => state.items.filter((i) => i.type === state.tab);
const current = () => tabItems()[state.index];

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

// ---------- Tabs ----------

function renderTabs() {
  $("tabs").innerHTML = TABS.map((t) => {
    const all = state.items.filter((i) => i.type === t.type);
    const done = all.filter((i) => state.labels[i.id]).length;
    return `<button type="button" class="tab" data-type="${t.type}" aria-pressed="${t.type === state.tab}">
      ${t.name} <span class="tab-count">${done} of ${all.length}</span></button>`;
  }).join("");
  for (const b of $("tabs").querySelectorAll(".tab")) {
    b.addEventListener("click", () => {
      state.tab = b.dataset.type;
      state.index = firstUnlabelled();
      setStatus("");
      render();
    });
  }
}

function firstUnlabelled() {
  const i = tabItems().findIndex((it) => !state.labels[it.id]);
  return i === -1 ? 0 : i;
}

// ---------- One item ----------

// A pair of Yes / No buttons; `key` is where the answer goes in the draft.
function yesNo(key, question, help = "") {
  const v = state.draft[key];
  return `<fieldset class="yesno" data-key="${key}">
    <legend>${question}</legend>
    ${help ? `<p class="hint">${help}</p>` : ""}
    <div class="yesno-buttons">
      <button type="button" class="choice-btn" data-val="true" aria-pressed="${v === true}">Yes</button>
      <button type="button" class="choice-btn" data-val="false" aria-pressed="${v === false}">No</button>
    </div></fieldset>`;
}

function excerptBlock(item, open = false) {
  return `<details class="excerpt-details"${open ? " open" : ""}><summary>Passage from the notes, page ${item.page}</summary>
    <blockquote>${escapeHTML(item.excerpt)}</blockquote></details>`;
}

function render() {
  renderTabs();
  const items = tabItems();
  const item = current();
  $("actions").hidden = !item;
  if (!item) {
    $("item-num").textContent = "";
    $("item").innerHTML = `<p class="empty-note">${TABS.find((t) => t.type === state.tab).empty}</p>`;
    $("keys-hint").textContent = "";
    return;
  }
  const saved = state.labels[item.id]?.value;
  $("item-num").textContent = `${state.index + 1}`;
  let html = `<p class="progress">${state.index + 1} of ${items.length}${saved ? ", already labelled" : ""}</p>`;

  if (item.type === "grading") {
    state.draft = Array.isArray(saved) ? [...saved] : item.points.map(() => null);
    html += `<h2 class="question">${escapeHTML(item.question)}</h2>
      <p class="label-small">The student's answer${item.language === "hinglish" ? " (Hinglish)" : ""}</p>
      <p class="student-answer">${escapeHTML(item.answer)}</p>
      <p class="label-small">Did the answer state each point correctly?</p>
      <p class="hint">Judge only what is said, not what the student probably meant. A vague hint is a no.</p>
      <ol class="point-checks">${item.points.map((p, i) => `
        <li><span class="point-text">${escapeHTML(p)}</span>
          <span class="yesno-buttons" data-point="${i}">
            <button type="button" class="choice-btn" data-val="true" aria-pressed="${state.draft[i] === true}">Yes</button>
            <button type="button" class="choice-btn" data-val="false" aria-pressed="${state.draft[i] === false}">No</button>
          </span></li>`).join("")}</ol>
      ${excerptBlock(item)}`;
    $("keys-hint").textContent = "Keys: 1-4 flip a point, Enter saves.";
  } else if (item.type === "grounding") {
    state.draft = saved ? { ...saved } : { answerable: null, level_ok: null };
    const levelHelp = {
      recall: "Recall: asks for a fact, definition or unit stated in the passage.",
      explain: "Explain: asks why or how something happens.",
      apply: "Apply: asks what would happen in a specific changed situation.",
    }[item.level];
    html += `${excerptBlock(item, true)}
      <h2 class="question">${escapeHTML(item.question)}</h2>
      ${yesNo("answerable", "Can it be fully answered from this passage alone?", "If you'd need knowledge from outside the passage, answer no.")}
      ${yesNo("level_ok", `Is it a good ${item.level} question?`, levelHelp)}`;
    $("keys-hint").textContent = "Keys: Y / N for the first question, then the second. Enter saves.";
  } else {
    state.draft = saved ? { ...saved } : { gives_away: null };
    html += `<h2 class="question">${escapeHTML(item.question)}</h2>
      <p class="label-small">The student said</p>
      <p class="student-answer">${escapeHTML(item.answer)}</p>
      <p class="label-small">The point they missed</p>
      <p class="missing-point">${escapeHTML(item.missing)}</p>
      <p class="label-small">The examiner's follow-up</p>
      <p class="probe-q">${escapeHTML(item.follow_up)}</p>
      ${yesNo("gives_away", "Does the follow-up give the missing point away?",
        "Pointing in the right direction is fine. Stating the point, or turning it into a yes/no question about it, gives it away.")}`;
    $("keys-hint").textContent = "Keys: Y or N, then Enter saves.";
  }
  $("item").innerHTML = html;
  wire(item);
}

function wire(item) {
  for (const group of $("item").querySelectorAll(".yesno-buttons")) {
    for (const btn of group.querySelectorAll(".choice-btn")) {
      btn.addEventListener("click", () => {
        const val = btn.dataset.val === "true";
        if (group.dataset.point !== undefined) state.draft[Number(group.dataset.point)] = val;
        else state.draft[group.closest(".yesno").dataset.key] = val;
        for (const b of group.querySelectorAll(".choice-btn")) b.setAttribute("aria-pressed", String(b === btn));
      });
    }
  }
}

function complete() {
  const d = state.draft;
  return Array.isArray(d) ? d.every((v) => v !== null) : Object.values(d).every((v) => v !== null);
}

async function save() {
  const item = current();
  if (!item) return;
  if (!complete()) {
    setStatus("Answer every question for this item first, or skip it.", true);
    return;
  }
  try {
    const res = await fetch("/api/review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: item.id, value: state.draft }),
    });
    if (!res.ok) throw new Error(`The server answered ${res.status}.`);
    state.labels[item.id] = { value: Array.isArray(state.draft) ? [...state.draft] : { ...state.draft } };
    setStatus("Saved.");
    next();
  } catch (e) {
    setStatus(`Couldn't save: ${e.message}`, true);
  }
}

function next() {
  const items = tabItems();
  const after = items.findIndex((it, i) => i > state.index && !state.labels[it.id]);
  if (after !== -1) state.index = after;
  else if (items.every((it) => state.labels[it.id])) setStatus("Everything in this tab is labelled. Run python evals/report.py to see the results.");
  else state.index = Math.min(state.index + 1, items.length - 1);
  render();
}

$("save-btn").addEventListener("click", save);
$("skip-btn").addEventListener("click", () => {
  setStatus("");
  state.index = Math.min(state.index + 1, tabItems().length - 1);
  render();
});
$("prev-btn").addEventListener("click", () => {
  setStatus("");
  state.index = Math.max(state.index - 1, 0);
  render();
});

// Keyboard: numbers flip grading points, Y/N answer the next open yes/no, Enter saves.
document.addEventListener("keydown", (e) => {
  if (e.target.closest("summary, input, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
  const item = current();
  if (!item) return;
  if (e.key === "Enter") { e.preventDefault(); save(); return; }
  if (item.type === "grading" && /^[1-9]$/.test(e.key)) {
    const i = Number(e.key) - 1;
    if (i < state.draft.length) {
      state.draft[i] = state.draft[i] === true ? false : true;
      const group = $("item").querySelector(`[data-point="${i}"]`);
      for (const b of group.querySelectorAll(".choice-btn")) b.setAttribute("aria-pressed", String(b.dataset.val === String(state.draft[i])));
    }
  }
  if (item.type !== "grading" && /^[yn]$/i.test(e.key)) {
    const key = Object.keys(state.draft).find((k) => state.draft[k] === null) ?? Object.keys(state.draft).at(-1);
    state.draft[key] = e.key.toLowerCase() === "y";
    const group = $("item").querySelector(`.yesno[data-key="${key}"]`);
    for (const b of group.querySelectorAll(".choice-btn")) b.setAttribute("aria-pressed", String(b.dataset.val === String(state.draft[key])));
  }
});

async function load() {
  try {
    const res = await fetch("/api/review");
    const data = await res.json();
    state.items = data.items;
    state.labels = data.labels;
    const firstTab = TABS.find((t) => state.items.some((i) => i.type === t.type && !state.labels[i.id]));
    state.tab = firstTab ? firstTab.type : "grading";
    state.index = firstUnlabelled();
    render();
  } catch (e) {
    setStatus(`Couldn't load items: ${e.message}. Is the server running?`, true);
  }
}
load();