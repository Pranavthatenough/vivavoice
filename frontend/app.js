// VivaVoice - the web page. Talks to the FastAPI server in backend/app.py.

const $ = (id) => document.getElementById(id);
const state = { sessionId: null, total: 0, timerId: null, startedAt: 0 };

// ---------- Helpers ----------

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `The server answered ${res.status}.`);
  return body;
}

function postJSON(path, data) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

function show(screen) {
  for (const s of ["home", "viva", "report"]) $(`screen-${s}`).hidden = s !== screen;
  window.scrollTo(0, 0);
}

function setStatus(id, text, isError = false) {
  $(id).textContent = text;
  $(id).classList.toggle("error", isError);
}

function escapeHTML(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function prettyName(name) {
  return name.replace(/[_-]+/g, " ");
}

// Highlight the evidence for each key point inside the notes excerpt.
function highlight(text, points) {
  const spans = points
    .filter((p) => p.span)
    .map((p) => ({ start: p.span[0], end: p.span[1], cls: p.covered ? "hl-covered" : "hl-missed" }))
    .sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const s of spans) {
    if (s.start < pos) continue;                 // skip overlapping quotes
    html += escapeHTML(text.slice(pos, s.start));
    html += `<mark class="${s.cls}">${escapeHTML(text.slice(s.start, s.end))}</mark>`;
    pos = s.end;
  }
  return html + escapeHTML(text.slice(pos));
}

function drawCircle(el) {
  el.classList.remove("drawn");
  void el.offsetWidth;                           // restart the animation
  el.classList.add("drawn");
}

// ---------- Home ----------

async function loadNotes() {
  const list = $("notes-list");
  list.innerHTML = "";
  try {
    const notes = await api("/api/notes");
    if (!notes.length) {
      list.innerHTML = `<li class="empty">No notes yet. Add a PDF below to begin.</li>`;
      return;
    }
    for (const n of notes) {
      const li = document.createElement("li");
      li.innerHTML = `<div><span class="name">${escapeHTML(prettyName(n.name))}</span>
        <span class="meta">${n.chunks} passages</span></div>`;
      const btn = document.createElement("button");
      btn.className = "btn";
      btn.type = "button";
      btn.textContent = "Start viva";
      btn.addEventListener("click", () => startViva(n.name, btn));
      li.appendChild(btn);
      list.appendChild(li);
    }
  } catch (e) {
    setStatus("home-status", `Couldn't load your notes: ${e.message}`, true);
  }
}

$("upload-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setStatus("home-status", `Reading ${file.name}. This can take a minute for long notes.`);
  const form = new FormData();
  form.append("file", file);
  try {
    const { name } = await api("/api/notes", { method: "POST", body: form });
    setStatus("home-status", `Added ${prettyName(name)}.`);
    loadNotes();
  } catch (err) {
    setStatus("home-status", err.message, true);
  }
  e.target.value = "";
});

// ---------- Viva ----------

async function startViva(notes, btn) {
  btn.disabled = true;
  setStatus("home-status", "The examiner is reading your notes and writing the first question. The first one can take up to a minute.");
  try {
    const data = await postJSON("/api/sessions", { notes });
    state.sessionId = data.session_id;
    state.total = data.total;
    $("topbar-note").textContent = prettyName(notes);
    setStatus("home-status", "");
    show("viva");
    showQuestion(data.question);
  } catch (e) {
    setStatus("home-status", e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function showQuestion(q) {
  $("qnum").textContent = `Q${q.number}`;
  $("progress").textContent = `Question ${q.number} of ${state.total}`;
  $("question").textContent = q.question;
  $("answer").value = "";
  $("answer").disabled = false;
  $("submit-btn").disabled = false;
  $("answer-form").hidden = false;
  document.querySelector("#answer-form .actions").hidden = false;
  $("marking").hidden = true;
  setStatus("viva-status", "");
  $("answer").focus();
  startTimer();
}

function startTimer() {
  clearInterval(state.timerId);
  state.startedAt = Date.now();
  const tick = () => {
    const s = Math.floor((Date.now() - state.startedAt) / 1000);
    $("timer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  tick();
  state.timerId = setInterval(tick, 1000);
}

$("answer-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const answer = $("answer").value.trim() || "pass";
  clearInterval(state.timerId);
  $("answer").disabled = true;
  $("submit-btn").disabled = true;
  setStatus("viva-status", "The examiner is marking your answer.");
  try {
    const r = await postJSON(`/api/sessions/${state.sessionId}/answer`, { answer });
    setStatus("viva-status", "");
    showMarking(r);
  } catch (err) {
    setStatus("viva-status", err.message, true);
    $("answer").disabled = false;
    $("submit-btn").disabled = false;
  }
});

$("answer").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("answer-form").requestSubmit();
});

function showMarking(r) {
  document.querySelector("#answer-form .actions").hidden = true;
  $("score-text").textContent = `${r.score}/5`;
  $("feedback").textContent = r.feedback;
  $("points").innerHTML = r.key_points
    .map((p) => `<li class="${p.covered ? "covered" : "missed"}"><span class="sr-only">${p.covered ? "Covered: " : "Missed: "}</span>${escapeHTML(p.point)}</li>`)
    .join("");
  $("excerpt-caption").textContent = `From your notes, page ${r.page}`;
  $("excerpt").innerHTML = highlight(r.source_text, r.key_points);
  $("next-btn").textContent = r.done ? "See your result" : "Next question";
  $("next-btn").disabled = false;
  $("marking").hidden = false;
  drawCircle($("score-mark"));
  $("marking").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  state.done = r.done;
}

$("next-btn").addEventListener("click", async () => {
  $("next-btn").disabled = true;
  if (state.done) return showReport();
  setStatus("viva-status", "");
  try {
    showQuestion(await postJSON(`/api/sessions/${state.sessionId}/next`, {}));
    window.scrollTo(0, 0);
  } catch (err) {
    setStatus("viva-status", err.message, true);
    $("next-btn").disabled = false;
  }
});

// ---------- Report ----------

async function showReport() {
  try {
    const rep = await api(`/api/sessions/${state.sessionId}/report`);
    $("total-text").textContent = `${rep.total}/${rep.max}`;
    const weak = rep.revise_pages;
    $("verdict").textContent = weak.length
      ? `Revise page${weak.length > 1 ? "s" : ""} ${weak.join(", ").replace(/, (\d+)$/, " and $1")} before your viva.`
      : "Well answered. No pages need revising from this round.";

    $("pages").innerHTML = rep.pages
      .map((p) => {
        const level = p.score === null ? "" : p.score >= 4 ? "ok" : p.score >= 2.5 ? "weak" : "bad";
        const label = p.score === null ? "not asked" : `${p.score}/5`;
        return `<li ${level ? `data-level="${level}"` : ""} aria-label="Page ${p.page}: ${label}">
          <span>p${p.page}</span><span class="pg-score">${p.score === null ? "" : p.score}</span></li>`;
      })
      .join("");

    $("q-list").innerHTML = rep.questions
      .map((q) => `<li>${escapeHTML(q.question)}
        <span class="q-meta"><span class="q-score">${q.score}/5</span> from page ${q.page}, answered in ${q.seconds_to_answer}s</span></li>`)
      .join("");

    show("report");
    drawCircle($("total-mark"));
  } catch (err) {
    setStatus("viva-status", err.message, true);
  }
}

// ---------- Navigation ----------

function goHome() {
  clearInterval(state.timerId);
  $("topbar-note").textContent = "";
  show("home");
  loadNotes();
}
$("home-link").addEventListener("click", goHome);
$("again-btn").addEventListener("click", goHome);

loadNotes();