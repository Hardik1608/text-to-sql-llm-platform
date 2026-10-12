// Text-to-SQL UI: one question -> generated SQL, safety verdict, result table, pipeline details.
// Plain JS, no dependencies. Every value from the database is inserted with textContent.
"use strict";

const $ = (id) => document.getElementById(id);
const DEFAULT_DB = "concert_singer";
const EXAMPLES = {
  concert_singer: [
    "How many singers are from France?",
    "Show the stadium name and the number of concerts in each stadium.",
    "What is the name of the youngest singer?",
  ],
  pets_1: [
    "What is the average age of students who own a dog?",
    "How many pets are older than 2?",
    "Find the last name of the students who have a cat.",
  ],
  car_1: [
    "Which car maker produces the most models?",
    "What is the average horsepower of cars made in 1980?",
    "List the names of car makers from Germany.",
  ],
  flight_2: [
    "How many flights depart from Aberdeen?",
    "Which airline has the most flights?",
    "List the airports in the city Anthony.",
  ],
  world_1: [
    "Which countries in Europe have a population over 50 million?",
    "What is the most spoken official language in Asia?",
    "How many cities are in China?",
  ],
};
const GENERIC_EXAMPLES = ["How many rows does the main table have?", "List all distinct values of the first text column."];
const STATUS = {
  ok: ["ok", "Executed"],
  rejected: ["bad", "Blocked by safety layer"],
  exec_error: ["warn", "Database error"],
  cannot_answer: ["warn", "Cannot answer from this schema"],
  llm_error: ["bad", "Model unavailable"],
};
const STAGES = [
  ["retrieval_ms", "Retrieval", "--s1"],
  ["generation_ms", "Generation", "--s2"],
  ["validation_ms", "Validation", "--s3"],
  ["execution_ms", "Execution", "--s4"],
];
const SQL_TOKEN = /('(?:[^']|'')*'|"(?:[^"]|"")*")|(\b\d+(?:\.\d+)?\b)|(\b(?:SELECT|FROM|WHERE|JOIN|INNER|LEFT|RIGHT|OUTER|ON|AS|AND|OR|NOT|IN|IS|NULL|GROUP|BY|ORDER|HAVING|LIMIT|OFFSET|DISTINCT|COUNT|SUM|AVG|MIN|MAX|UNION|INTERSECT|EXCEPT|ALL|WITH|CASE|WHEN|THEN|ELSE|END|LIKE|BETWEEN|EXISTS|ASC|DESC|CREATE|TABLE|PRIMARY|KEY|FOREIGN|REFERENCES|TEXT|NUMBER|INTEGER|REAL|CAST)\b)/gi;

const MODE_INFO = {
  prompt: ["Prompting", "--s1"],
  rag: ["Schema RAG", "--s2"],
  qlora: ["QLoRA", "--s3"],
  qlora_rag: ["QLoRA + RAG", "--s4"],
};
const COMPARE_STATUS = {
  rejected: ["bad", "Blocked"],
  exec_error: ["warn", "Database error"],
  cannot_answer: ["warn", "Cannot answer"],
  llm_error: ["bad", "Model error"],
  unavailable: ["neutral", "Not available"],
};

let mode = "qlora_rag";
let view = "single";
let busy = false;
const shown = { single: false, compare: false };

// ---------------------------------------------------------------- helpers

function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (_) { /* private mode / blocked storage: ignore */ }
  return null;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let body = null;
  try { body = await res.json(); } catch (_) { body = null; }
  return { status: res.status, body };
}

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderSQL(target, sql) {
  target.replaceChildren();
  let last = 0;
  for (const m of sql.matchAll(SQL_TOKEN)) {
    if (m.index > last) target.append(sql.slice(last, m.index));
    const cls = m[1] ? "str" : m[2] ? "num" : "kw";
    target.append(el("span", { class: cls }, m[0]));
    last = m.index + m[0].length;
  }
  target.append(sql.slice(last));
}

function formatMs(ms) {
  if (ms === undefined || ms === null) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
}

// ---------------------------------------------------------------- setup

async function loadDatabases() {
  const select = $("db-select");
  const { status, body } = await api("/databases");
  if (status !== 200 || !body) return;
  const saved = storage("t2s.db");
  for (const id of body.databases) select.append(el("option", { value: id }, id));
  select.value = body.databases.includes(saved) ? saved : body.databases.includes(DEFAULT_DB) ? DEFAULT_DB : body.databases[0];
  onDatabaseChange();
}

async function loadAdapters() {
  const { status, body } = await api("/adapters");
  if (status !== 200 || !body) return;
  const adapter = body.default;
  $("model-line").textContent = adapter
    ? `${body.base_model} · LoRA adapter ${adapter} · Schema RAG`
    : `${body.base_model} · Schema RAG`;
  if (!adapter) {
    for (const btn of document.querySelectorAll('[data-mode^="qlora"]')) {
      btn.disabled = true;
      btn.title = "No LoRA adapter is registered";
    }
    if (mode.startsWith("qlora")) setMode("rag");
  }
}

async function checkReady() {
  const pill = $("status-pill");
  try {
    const { status, body } = await api("/ready");
    const up = status === 200 && body && body.llm;
    pill.className = `status-pill ${up ? "up" : "down"}`;
    $("status-text").textContent = up ? "Model online" : "Model offline";
  } catch (_) {
    pill.className = "status-pill down";
    $("status-text").textContent = "API unreachable";
  }
}

function onDatabaseChange() {
  storage("t2s.db", $("db-select").value);
  loadChips();
  if (!$("schema-panel").hidden) loadSchema();
}

async function loadChips() {
  const db = $("db-select").value;
  let questions = EXAMPLES[db] || GENERIC_EXAMPLES;
  let benchmark = false;
  if (view === "compare") {
    // held-out Spider questions: the answer is known, so the comparison can grade each approach
    const { status, body } = await api(`/databases/${encodeURIComponent(db)}/questions?limit=6`);
    if (status === 200 && body && body.questions.length) {
      questions = body.questions.map((q) => q.question);
      benchmark = true;
    }
  }
  if (db !== $("db-select").value) return; // the user switched database while we were loading
  $("examples-note").hidden = !benchmark;
  const box = $("examples");
  box.replaceChildren();
  for (const q of questions) {
    const chip = el("button", { type: "button", class: "chip" }, q);
    chip.addEventListener("click", () => { $("question").value = q; $("question").focus(); });
    box.append(chip);
  }
}

function setView(next) {
  view = next;
  document.body.classList.toggle("view-compare", next === "compare");
  for (const tab of document.querySelectorAll("#view-tabs button")) {
    const on = tab.dataset.view === next;
    tab.classList.toggle("active", on);
    tab.setAttribute("aria-selected", String(on));
  }
  $("run-btn").querySelector(".label").textContent = next === "compare" ? "Compare 4 approaches" : "Generate SQL";
  $("result").hidden = !(next === "single" && shown.single);
  $("compare").hidden = !(next === "compare" && shown.compare);
  $("empty").hidden = shown[next];
  storage("t2s.view", next);
  loadChips();
}

async function loadSchema() {
  const res = await fetch(`/databases/${encodeURIComponent($("db-select").value)}`);
  renderSQL($("schema-code"), res.ok ? await res.text() : "Schema unavailable.");
}

function setMode(next) {
  mode = next;
  for (const btn of document.querySelectorAll("#mode-group button")) {
    const on = btn.dataset.mode === next;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-checked", String(on));
  }
  storage("t2s.mode", next);
}

// ---------------------------------------------------------------- run

async function run() {
  const question = $("question").value.trim();
  if (busy) return;
  if (question.length < 3) { $("question").focus(); return; }
  busy = true;
  const btn = $("run-btn");
  const idle = btn.querySelector(".label").textContent;
  btn.classList.add("loading");
  btn.disabled = true;
  btn.querySelector(".label").textContent = view === "compare" ? "Running all 4…" : "Generating…";
  const db_id = $("db-select").value;
  try {
    if (view === "compare") {
      const { status, body } = await api("/compare", { method: "POST", body: JSON.stringify({ question, db_id }) });
      if (body && body.results) renderCompare(body);
      else renderCompareError(status, body);
    } else {
      const { status, body } = await api("/query", {
        method: "POST",
        body: JSON.stringify({ question, db_id, mode, max_rows: 200 }),
      });
      if (body && body.status) render(body);
      else renderError(status, body);
    }
  } catch (err) {
    if (view === "compare") renderCompareError(0, { detail: String(err) });
    else renderError(0, { detail: String(err) });
  } finally {
    busy = false;
    btn.classList.remove("loading");
    btn.disabled = false;
    btn.querySelector(".label").textContent = idle;
  }
}

function errorText(status, body) {
  const detail = body && body.detail
    ? (typeof body.detail === "string" ? body.detail : body.detail.map((d) => d.msg).join("; "))
    : "Request failed.";
  return `HTTP ${status || "—"}: ${detail}`;
}

function renderError(status, body) {
  render({ status: "llm_error", error: errorText(status, body), timings: {}, usage: {} });
}

function render(r) {
  shown.single = true;
  $("empty").hidden = true;
  $("result").hidden = false;

  // badges + meta
  const [tone, label] = STATUS[r.status] || ["neutral", r.status];
  const badges = $("badges");
  badges.replaceChildren(el("span", { class: `badge ${tone}` }, label));
  if (r.cached) badges.append(el("span", { class: "badge neutral" }, "Cache hit"));
  if (r.repaired) badges.append(el("span", { class: "badge neutral" }, "Self-repaired"));
  if (r.routed) badges.append(el("span", { class: "badge neutral" }, `Routed to ${r.db_id}`));
  $("result-meta").textContent = r.model
    ? `${r.adapter ? `LoRA ${r.adapter}` : r.model} · ${formatMs(r.timings && r.timings.total_ms)}`
    : "";

  // alert for non-ok outcomes
  const alert = $("alert");
  alert.replaceChildren();
  alert.hidden = r.status === "ok";
  if (r.status !== "ok") {
    alert.className = `alert ${tone === "bad" ? "bad" : "warn"}`;
    if (r.violations && r.violations.length) {
      alert.append(el("strong", {}, "The generated SQL was not executed:"));
      const list = el("ul");
      for (const v of r.violations) {
        const item = el("li");
        item.append(el("code", {}, v.code), ` ${v.message}`);
        list.append(item);
      }
      alert.append(list);
    } else if (r.status === "cannot_answer") {
      alert.textContent = "The model says this question cannot be answered from the selected database.";
    } else {
      alert.textContent = r.error || "Something went wrong.";
    }
  }

  // SQL
  const hasSql = r.sql && r.sql !== "CANNOT_ANSWER";
  $("sql-block").hidden = !hasSql;
  if (hasSql) renderSQL($("sql-code"), r.sql);

  // table
  const showTable = r.status === "ok" && Array.isArray(r.columns);
  $("table-block").hidden = !showTable;
  if (showTable) renderTable(r);

  renderTimings(r.timings || {});
  renderContext(r);
}

function renderTable(r) {
  const rows = r.rows || [];
  const n = r.row_count ?? rows.length;
  $("row-count").textContent = `${n} row${n === 1 ? "" : "s"}${r.truncated ? " (truncated)" : ""}`;

  // One value ("How many …?") reads better as a figure than as a 1x1 table
  const scalar = rows.length === 1 && r.columns.length === 1;
  $("answer").hidden = !scalar;
  $("table-wrap").hidden = scalar;
  if (scalar) {
    const v = rows[0][0];
    $("answer-value").textContent = v === null ? "NULL" : typeof v === "number" ? v.toLocaleString() : String(v);
    $("answer-label").textContent = r.columns[0];
    return;
  }

  // a column is numeric when every non-null value is a number: right-align header and cells
  const numeric = r.columns.map((_, i) =>
    rows.length > 0 && rows.every((row) => row[i] === null || typeof row[i] === "number"));
  const table = $("result-table");
  table.replaceChildren();
  const head = el("thead");
  const hr = el("tr");
  r.columns.forEach((c, i) => hr.append(el("th", numeric[i] ? { scope: "col", class: "num" } : { scope: "col" }, c)));
  head.append(hr);
  const tbody = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    row.forEach((v, i) => {
      if (v === null) tr.append(el("td", { class: "null" }, "NULL"));
      else tr.append(el("td", numeric[i] ? { class: "num" } : {}, String(v)));
    });
    tbody.append(tr);
  }
  if (rows.length === 0) {
    const tr = el("tr");
    tr.append(el("td", { colspan: String(r.columns.length || 1), class: "null" }, "No rows"));
    tbody.append(tr);
  }
  table.append(head, tbody);
}

function renderTimings(t) {
  const bar = $("timing-bar");
  const legend = $("timing-legend");
  bar.replaceChildren();
  legend.replaceChildren();
  const parts = STAGES.filter(([k]) => t[k] !== undefined);
  const sum = parts.reduce((acc, [k]) => acc + t[k], 0) || 1;
  for (const [key, name, color] of STAGES) {
    const ms = t[key];
    if (ms !== undefined) {
      const seg = el("span", { title: `${name}: ${formatMs(ms)}` });
      seg.style.width = `${(ms / sum) * 100}%`;
      seg.style.background = `var(${color})`;
      bar.append(seg);
    }
    const li = el("li");
    const sw = el("span", { class: "sw" });
    sw.style.background = `var(${color})`;
    li.append(sw, el("span", {}, name), el("span", { class: "val" }, ms === undefined ? "skipped" : formatMs(ms)));
    legend.append(li);
  }
  const total = el("li", { class: "timing-total" });
  total.append(el("span", {}, "Total"), el("span", { class: "val" }, formatMs(t.total_ms)));
  legend.append(total);
}

function renderContext(r) {
  const dl = $("context-list");
  dl.replaceChildren();
  const add = (label, content) => {
    const dd = el("dd");
    if (content instanceof Node) dd.append(content); else dd.textContent = content;
    dl.append(el("dt", {}, label), dd);
  };
  const modeNames = { prompt: "Prompting", rag: "Schema RAG", qlora: "QLoRA", qlora_rag: "QLoRA + RAG" };
  add("Approach", modeNames[r.mode] || r.mode || "–");
  if (r.context_tables && r.context_tables.length) {
    const tags = document.createDocumentFragment();
    for (const t of r.context_tables) tags.append(el("span", { class: "tag" }, t));
    add("Tables", tags);
  }
  if (r.value_hints && r.value_hints.length) {
    const tags = document.createDocumentFragment();
    for (const h of r.value_hints) tags.append(el("span", { class: "tag" }, h));
    add("Value hints", tags);
  }
  if (r.num_examples) add("Few-shot examples", `${r.num_examples} retrieved`);
  if (r.usage) add("Tokens", `${r.usage.prompt_tokens ?? 0} in · ${r.usage.completion_tokens ?? 0} out`);
}

// ---------------------------------------------------------------- comparison

function previewValue(columns, rows) {
  if (!rows || rows.length === 0) return "no rows";
  if (rows.length === 1 && columns && columns.length === 1) {
    const v = rows[0][0];
    return v === null ? "NULL" : typeof v === "number" ? v.toLocaleString() : String(v);
  }
  return `${rows.length} row${rows.length === 1 ? "" : "s"}`;
}

function renderCompareError(status, body) {
  renderCompare({ results: [], agreement: [], gold: null, error: errorText(status, body) });
}

function renderCompare(c) {
  shown.compare = true;
  $("empty").hidden = true;
  $("compare").hidden = false;
  const results = c.results || [];
  const total = results.filter((r) => r.status !== "unavailable").length;

  // headline: graded against the reference answer when known, else agreement between approaches
  const title = $("compare-title");
  const sub = $("compare-sub");
  const gold = c.gold;
  if (c.error) {
    title.textContent = "Comparison failed";
    sub.textContent = c.error;
  } else if (gold && !gold.error) {
    title.textContent = `${c.correct_count} of ${total} approaches correct`;
    sub.textContent = `Spider ${gold.split} question · expected answer: ${previewValue(gold.columns, gold.rows)}`;
  } else {
    const top = (c.agreement[0] || []).length;
    title.textContent = top === total && total > 1 ? "All approaches agree" : `${top} of ${total} approaches agree`;
    sub.textContent = gold && gold.error
      ? "The reference query failed on this database; results are grouped by agreement."
      : "No reference answer for this question; results are grouped by agreement.";
  }
  const goldToggle = $("gold-toggle");
  goldToggle.hidden = !(gold && gold.sql);
  $("gold-sql").hidden = true;
  goldToggle.textContent = "Show reference SQL";
  if (gold && gold.sql) renderSQL($("gold-sql"), gold.sql);

  const maxMs = Math.max(1, ...results.map((r) => (r.timings && r.timings.total_ms) || 0));
  const grid = $("compare-grid");
  grid.replaceChildren(...results.map((r) => modeCard(r, maxMs)));
}

function modeCard(r, maxMs) {
  const [name, color] = MODE_INFO[r.mode] || [r.mode, "--muted"];
  const card = el("article", { class: `mode-card${r.status === "unavailable" ? " dim" : ""}` });
  card.style.setProperty("--mode", `var(${color})`);

  const head = el("div");
  head.append(el("div", { class: "mode-name" }, name));
  head.append(el("div", { class: "mode-model" }, r.adapter ? `LoRA ${r.adapter}` : r.model || "—"));

  // verdict: graded (✓/✗), otherwise the outcome or the agreement with other approaches
  let tone;
  let label;
  if (r.status !== "ok") [tone, label] = COMPARE_STATUS[r.status] || ["neutral", r.status];
  else if (r.correct === true) [tone, label] = ["ok", "✓ Correct"];
  else if (r.correct === false) [tone, label] = ["bad", "✗ Wrong"];
  else {
    const peers = (r.agrees_with || []).length - 1;
    [tone, label] = peers > 0 ? ["neutral", `Agrees with ${peers} other${peers === 1 ? "" : "s"}`] : ["warn", "Differs from others"];
  }
  const verdict = el("span", { class: `verdict ${tone}` }, label);

  const body = el("div");
  const hasSql = r.sql && r.sql !== "CANNOT_ANSWER";
  if (hasSql) {
    const pre = el("pre", { class: "code" });
    renderSQL(pre, r.sql);
    body.append(pre);
  }
  const out = el("div");
  if (r.status === "ok" && Array.isArray(r.columns)) {
    const rows = r.rows || [];
    if (rows.length === 1 && r.columns.length === 1) {
      out.append(el("div", { class: "mini-answer" }, previewValue(r.columns, rows)), el("div", { class: "mini-label" }, r.columns[0]));
    } else if (rows.length === 0) {
      out.append(el("div", { class: "mini-note" }, "No rows"));
    } else {
      const table = el("table", { class: "mini-table" });
      const hr = el("tr");
      for (const col of r.columns) hr.append(el("th", {}, col));
      table.append(hr);
      for (const row of rows.slice(0, 5)) {
        const tr = el("tr");
        for (const v of row) tr.append(el("td", {}, v === null ? "NULL" : String(v)));
        table.append(tr);
      }
      out.append(table);
      const total = r.row_count ?? rows.length;
      if (total > 5) out.append(el("div", { class: "mini-more" }, `+ ${total - 5} more rows`));
    }
  } else {
    const v = (r.violations || [])[0];
    out.append(el("div", { class: "mini-note" }, v ? v.message : r.error || (r.status === "cannot_answer" ? "The model declined: not answerable from this schema." : "")));
  }
  body.append(out);

  card.append(head, verdict, body);
  if (r.status !== "unavailable") {
    const ms = (r.timings && r.timings.total_ms) || 0;
    const foot = el("div", { class: "mode-foot" });
    const track = el("div", { class: "lat-track", title: `Total time: ${formatMs(ms)}` });
    const fill = el("div", { class: "lat-fill" });
    fill.style.width = `${(ms / maxMs) * 100}%`;
    track.append(fill);
    const text = el("div", { class: "lat-text" });
    const tokens = r.usage ? (r.usage.prompt_tokens ?? 0) + (r.usage.completion_tokens ?? 0) : 0;
    text.append(el("span", {}, formatMs(ms)), el("span", {}, `${tokens} tokens`));
    foot.append(track, text);
    card.append(foot);
  }
  return card;
}

// ---------------------------------------------------------------- wiring

function initTheme() {
  const saved = storage("t2s.theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    storage("t2s.theme", next);
  });
}

function init() {
  initTheme();
  $("monitoring-link").href = `${location.protocol}//${location.hostname}:3000/`;
  const savedMode = storage("t2s.mode");
  if (savedMode) setMode(savedMode);
  for (const btn of document.querySelectorAll("#mode-group button")) {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  }
  for (const tab of document.querySelectorAll("#view-tabs button")) {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  }
  $("gold-toggle").addEventListener("click", () => {
    const pre = $("gold-sql");
    pre.hidden = !pre.hidden;
    $("gold-toggle").setAttribute("aria-expanded", String(!pre.hidden));
    $("gold-toggle").textContent = pre.hidden ? "Show reference SQL" : "Hide reference SQL";
  });
  $("db-select").addEventListener("change", onDatabaseChange);
  $("schema-toggle").addEventListener("click", () => {
    const panel = $("schema-panel");
    panel.hidden = !panel.hidden;
    $("schema-toggle").setAttribute("aria-expanded", String(!panel.hidden));
    $("schema-toggle").textContent = panel.hidden ? "View schema" : "Hide schema";
    if (!panel.hidden) loadSchema();
  });
  $("run-btn").addEventListener("click", run);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
  });
  $("copy-sql").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("sql-code").textContent);
      $("copy-sql").textContent = "Copied";
      setTimeout(() => { $("copy-sql").textContent = "Copy"; }, 1200);
    } catch (_) { /* clipboard blocked on plain http: ignore */ }
  });
  if (storage("t2s.view") === "compare") setView("compare");
  loadDatabases();
  loadAdapters();
  checkReady();
  setInterval(checkReady, 15000);
}

init();
