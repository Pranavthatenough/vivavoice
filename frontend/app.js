// VivaVoice - the web page. Talks to the FastAPI server in backend/app.py.

const $ = (id) => document.getElementById(id);
const state = {
  sessionId: null,
  total: 0,
  done: false,
  timerId: null,
  startedAt: 0,
  questionReadyAt: 0,   // when the question had been read out (or shown)
  turn: 0,              // goes up on every new question; old async work checks it and stops
  stt: false,           // speech-to-text available on the server
  tts: "browser",       // "piper" or "browser"
};
const settings = { speak: true, handsFree: false };

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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function listWords(items) {
  return items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
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

// ---------- Settings (remembered in this browser) ----------

function loadSettings() {
  try {
    Object.assign(settings, JSON.parse(localStorage.getItem("vivavoice-settings") || "{}"));
  } catch { /* storage blocked: use defaults */ }
  $("opt-speak").checked = settings.speak;
  $("opt-handsfree").checked = settings.handsFree;
}

function saveSettings() {
  settings.speak = $("opt-speak").checked;
  settings.handsFree = $("opt-handsfree").checked;
  try { localStorage.setItem("vivavoice-settings", JSON.stringify(settings)); } catch { /* ignore */ }
  if (!settings.speak) stopSpeaking();
}
$("opt-speak").addEventListener("change", saveSettings);
$("opt-handsfree").addEventListener("change", saveSettings);

// ---------- The examiner's voice ----------

let speaking = null;   // { stop() } for whatever is talking now

function pickBrowserVoice() {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"));
  // Prefer an Indian English voice, then any voice that works offline.
  return voices.find((v) => v.lang === "en-IN" && v.localService)
    || voices.find((v) => v.lang === "en-IN")
    || voices.find((v) => v.localService)
    || voices[0];
}

function stopSpeaking() {
  if (speaking) speaking.stop();
  speaking = null;
}

// Speak the text; resolves when it has finished (or was stopped).
async function speak(text) {
  stopSpeaking();
  if (!settings.speak || !text) return;

  if (state.tts === "piper") {
    try {
      const res = await fetch("/api/speak", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      });
      if (res.ok) {
        const url = URL.createObjectURL(await res.blob());
        const audio = new Audio(url);
        await new Promise((resolve) => {
          speaking = { stop: () => { audio.pause(); resolve(); } };
          audio.onended = audio.onerror = resolve;
          audio.play().catch(resolve);
        });
        URL.revokeObjectURL(url);
        return;
      }
    } catch { /* fall back to the browser voice */ }
    state.tts = "browser";
  }

  if (!("speechSynthesis" in window)) return;
  await new Promise((resolve) => {
    const u = new SpeechSynthesisUtterance(text);
    const v = pickBrowserVoice();
    if (v) u.voice = v;
    u.rate = 0.98;
    // Safety net: some browsers never fire "end".
    const timer = setTimeout(resolve, 4000 + text.length * 90);
    const finish = () => { clearTimeout(timer); resolve(); };
    u.onend = u.onerror = finish;
    speaking = { stop: () => { speechSynthesis.cancel(); finish(); } };
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  });
  speaking = null;
}

// ---------- The examiner's ears: recording with automatic stop ----------

const SILENCE_TO_STOP_MS = 2000;   // stop after this much quiet once you've started talking
const NOTHING_HEARD_MS = 12000;    // give up if you haven't said anything by then
const MAX_RECORDING_MS = 90000;

let rec = null;   // the recording in progress

async function startRecording() {
  if (rec) return;
  stopSpeaking();                      // talking over the examiner cuts it off
  const turn = state.turn;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    setStatus("mic-status", "The microphone is blocked. Allow it in the browser's address bar, then try again.", true);
    return;
  }
  if (turn !== state.turn) { stream.getTracks().forEach((t) => t.stop()); return; }

  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);

  // Listen to the loudness to know when you start and stop talking.
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);

  rec = {
    recorder, stream, ctx, chunks, turn,
    openedAt: performance.now(),
    waited: (performance.now() - state.questionReadyAt) / 1000,
    heardSpeech: false,
    noiseFloor: null,
    calibration: [],
    loudFrames: 0,
    quietSince: null,
  };
  recorder.start(250);
  $("mic-btn").setAttribute("aria-pressed", "true");
  $("mic-label").textContent = "Stop";
  setStatus("mic-status", "Listening. Stop talking for 2 seconds when you're done.");

  const r = rec;
  const tick = () => {
    if (rec !== r) return;
    analyser.getFloatTimeDomainData(samples);
    let sum = 0;
    for (const x of samples) sum += x * x;
    const rms = Math.sqrt(sum / samples.length);
    const now = performance.now();
    const elapsed = now - r.openedAt;

    // First 0.4 s: measure the room's background noise.
    if (elapsed < 400) r.calibration.push(rms);
    else if (r.noiseFloor === null) r.noiseFloor = r.calibration.reduce((a, b) => a + b, 0) / Math.max(r.calibration.length, 1);

    const threshold = Math.max(0.012, (r.noiseFloor ?? 0.01) * 2.5);
    const loud = r.noiseFloor !== null && rms > threshold;
    r.loudFrames = loud ? r.loudFrames + 1 : 0;
    if (r.loudFrames >= 3) r.heardSpeech = true;          // 3 loud frames in a row = speech
    if (loud) r.quietSince = null;
    else if (r.quietSince === null) r.quietSince = now;

    $("mic-btn").style.setProperty("--level", Math.min(1, rms / (threshold * 3)).toFixed(2));

    if (r.heardSpeech && r.quietSince && now - r.quietSince > SILENCE_TO_STOP_MS) return stopRecording();
    if (!r.heardSpeech && elapsed > NOTHING_HEARD_MS) return stopRecording();
    if (elapsed > MAX_RECORDING_MS) return stopRecording();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function stopRecording() {
  const r = rec;
  if (!r) return;
  rec = null;
  $("mic-btn").setAttribute("aria-pressed", "false");
  $("mic-btn").style.setProperty("--level", 0);
  $("mic-label").textContent = "Speak your answer";

  const stopped = new Promise((resolve) => (r.recorder.onstop = resolve));
  r.recorder.stop();
  await stopped;
  r.stream.getTracks().forEach((t) => t.stop());
  r.ctx.close();
  if (r.turn !== state.turn) return;   // the question changed while recording

  if (!r.heardSpeech) {
    setStatus("mic-status", "I didn't hear anything. Check the microphone, move a little closer, and try again.", true);
    return;
  }
  const blob = new Blob(r.chunks, { type: r.recorder.mimeType || "audio/webm" });
  await sendRecording(blob, r.waited, r.turn);
}

async function sendRecording(blob, waited, turn) {
  $("mic-btn").disabled = true;
  setStatus("mic-status", "Writing down what you said.");
  const form = new FormData();
  form.append("audio", blob, "answer.webm");
  form.append("waited", waited.toFixed(2));
  try {
    const t = await api(`/api/sessions/${state.sessionId}/transcribe`, { method: "POST", body: form });
    if (turn !== state.turn) return;
    if (!t.text) {
      setStatus("mic-status", "I couldn't make out any words. Try again, speaking a little louder.", true);
      return;
    }
    const box = $("answer");
    box.value = box.value.trim() ? `${box.value.trim()} ${t.text}` : t.text;
    if (settings.handsFree) {
      setStatus("mic-status", "Got it. Handing in your answer.");
      await sleep(1200);
      if (turn === state.turn && !$("submit-btn").disabled) $("answer-form").requestSubmit();
    } else {
      setStatus("mic-status", "Check the words. Fix anything I misheard, then hand it in.");
      box.focus();
    }
  } catch (err) {
    if (turn === state.turn) setStatus("mic-status", err.message, true);
  } finally {
    $("mic-btn").disabled = false;
  }
}

$("mic-btn").addEventListener("click", () => (rec ? stopRecording() : startRecording()));

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
    state.done = false;
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

async function showQuestion(q) {
  const turn = ++state.turn;
  if (rec) stopRecording();
  $("qnum").textContent = `Q${q.number}`;
  $("progress").textContent = `Question ${q.number} of ${state.total}`;
  $("question").textContent = q.question;
  $("answer").value = "";
  $("answer").disabled = false;
  $("submit-btn").disabled = false;
  $("mic-btn").disabled = false;
  $("mic-row").hidden = !state.stt;
  $("answer-form").hidden = false;
  document.querySelector("#answer-form .actions").hidden = false;
  $("marking").hidden = true;
  setStatus("viva-status", "");
  setStatus("mic-status", "");
  startTimer();

  // Read the question out. The clock for "time before you started" begins after it.
  state.questionReadyAt = performance.now();
  if (settings.speak) {
    setStatus("mic-status", "The examiner is asking.");
    await speak(`Question ${q.number}. ${q.question}`);
    if (turn !== state.turn) return;
    if (!rec) setStatus("mic-status", "");
    state.questionReadyAt = performance.now();
  }
  if (settings.handsFree && state.stt && !rec && turn === state.turn) startRecording();
  else if (!rec) $(state.stt ? "mic-btn" : "answer").focus();
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
  if (rec) {
    // Still recording: write down what was said first. In hands-free mode that hands it in by itself.
    await stopRecording();
    if (settings.handsFree) return;
  }
  const answer = $("answer").value.trim() || "pass";
  const turn = state.turn;
  clearInterval(state.timerId);
  stopSpeaking();
  $("answer").disabled = true;
  $("submit-btn").disabled = true;
  $("mic-btn").disabled = true;
  setStatus("mic-status", "");
  setStatus("viva-status", "The examiner is marking your answer.");
  try {
    const r = await postJSON(`/api/sessions/${state.sessionId}/answer`, { answer });
    if (turn !== state.turn) return;
    setStatus("viva-status", "");
    showMarking(r, turn);
  } catch (err) {
    setStatus("viva-status", err.message, true);
    $("answer").disabled = false;
    $("submit-btn").disabled = false;
    $("mic-btn").disabled = false;
  }
});

$("answer").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("answer-form").requestSubmit();
});

// One line in the examiner's hand about how the answer was spoken.
function describeDelivery(v) {
  if (!v) return "";
  const s = v.signals;
  const parts = [];
  if (v.seconds_before_speaking != null) parts.push(`You started speaking after ${v.seconds_before_speaking} s.`);
  if (s.fillers.length) {
    const counts = {};
    for (const f of s.fillers) counts[f] = (counts[f] || 0) + 1;
    const list = Object.entries(counts).map(([w, n]) => (n > 1 ? `"${w}" ×${n}` : `"${w}"`));
    parts.push(`${s.fillers.length} filler${s.fillers.length > 1 ? "s" : ""}: ${listWords(list)}.`);
  } else {
    parts.push("No filler words.");
  }
  if (s.pauses.length) parts.push(`${s.pauses.length} long pause${s.pauses.length > 1 ? "s" : ""}, the longest ${Math.max(...s.pauses.map((p) => p.seconds))} s.`);
  if (s.words_per_minute) parts.push(`${s.words_per_minute} words a minute.`);
  return parts.join(" ");
}

async function showMarking(r, turn) {
  document.querySelector("#answer-form .actions").hidden = true;
  $("score-text").textContent = `${r.score}/5`;
  $("feedback").textContent = r.feedback;
  $("points").innerHTML = r.key_points
    .map((p) => `<li class="${p.covered ? "covered" : "missed"}"><span class="sr-only">${p.covered ? "Covered: " : "Missed: "}</span>${escapeHTML(p.point)}</li>`)
    .join("");
  const delivery = describeDelivery(r.voice);
  $("delivery").textContent = delivery;
  $("delivery").hidden = !delivery;
  $("excerpt-caption").textContent = `From your notes, page ${r.page}`;
  $("excerpt").innerHTML = highlight(r.source_text, r.key_points);
  $("next-btn").textContent = r.done ? "See your result" : "Next question";
  $("next-btn").disabled = false;
  $("marking").hidden = false;
  drawCircle($("score-mark"));
  $("marking").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  state.done = r.done;

  await speak(`${r.score} out of 5. ${r.feedback}`);
  // Hands-free: carry on to the next question by itself, like a real viva.
  if (settings.handsFree && turn === state.turn && !$("next-btn").disabled) {
    await sleep(1500);
    if (turn === state.turn && !$("next-btn").disabled) $("next-btn").click();
  }
}

$("next-btn").addEventListener("click", async () => {
  $("next-btn").disabled = true;
  stopSpeaking();
  if (state.done) return showReport();
  setStatus("viva-status", "");
  try {
    const q = await postJSON(`/api/sessions/${state.sessionId}/next`, {});
    window.scrollTo(0, 0);
    showQuestion(q);
  } catch (err) {
    setStatus("viva-status", err.message, true);
    $("next-btn").disabled = false;
  }
});

// ---------- Report ----------

async function showReport() {
  state.turn++;
  try {
    const rep = await api(`/api/sessions/${state.sessionId}/report`);
    $("total-text").textContent = `${rep.total}/${rep.max}`;
    const weak = rep.revise_pages;
    const verdict = weak.length
      ? `Revise page${weak.length > 1 ? "s" : ""} ${listWords(weak.map(String))} before your viva.`
      : "Well answered. No pages need revising from this round.";
    $("verdict").textContent = verdict;

    $("pages").innerHTML = rep.pages
      .map((p) => {
        const level = p.score === null ? "" : p.score >= 4 ? "ok" : p.score >= 2.5 ? "weak" : "bad";
        const label = p.score === null ? "not asked" : `${p.score}/5`;
        return `<li ${level ? `data-level="${level}"` : ""} aria-label="Page ${p.page}: ${label}">
          <span>p${p.page}</span><span class="pg-score">${p.score === null ? "" : p.score}</span></li>`;
      })
      .join("");

    const v = rep.voice;
    $("voice-report").hidden = !v;
    if (v) {
      const parts = [`You spoke ${v.spoken_answers} of your answers.`];
      if (v.avg_seconds_before_speaking != null) parts.push(`On average you started after ${v.avg_seconds_before_speaking} s.`);
      if (v.avg_words_per_minute) parts.push(`Your pace was about ${v.avg_words_per_minute} words a minute.`);
      parts.push(`${v.total_fillers} filler word${v.total_fillers === 1 ? "" : "s"} and ${v.total_long_pauses} long pause${v.total_long_pauses === 1 ? "" : "s"} in total.`);
      $("voice-summary").textContent = parts.join(" ");
    }

    $("q-list").innerHTML = rep.questions
      .map((q) => `<li>${escapeHTML(q.question)}
        <span class="q-meta"><span class="q-score">${q.score}/5</span> from page ${q.page}, answered in ${q.seconds_to_answer}s</span></li>`)
      .join("");

    show("report");
    drawCircle($("total-mark"));
    speak(`You scored ${rep.total} out of ${rep.max}. ${verdict}`);
  } catch (err) {
    setStatus("viva-status", err.message, true);
  }
}

// ---------- Navigation ----------

function goHome() {
  state.turn++;
  if (rec) stopRecording();
  stopSpeaking();
  clearInterval(state.timerId);
  $("topbar-note").textContent = "";
  show("home");
  loadNotes();
}
$("home-link").addEventListener("click", goHome);
$("again-btn").addEventListener("click", goHome);

// ---------- Start ----------

async function checkVoice() {
  try {
    const v = await api("/api/voice");
    state.stt = v.stt;
    state.tts = v.tts;
  } catch { /* text mode only */ }
  if (!state.stt) $("opt-handsfree").closest("label").hidden = true;
}

if ("speechSynthesis" in window) speechSynthesis.getVoices();   // starts loading the voice list
loadSettings();
checkVoice();
loadNotes();