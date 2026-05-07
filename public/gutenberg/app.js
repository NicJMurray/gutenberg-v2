const numberFormat = new Intl.NumberFormat("en-GB");
const byteFormat = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 1,
  minimumFractionDigits: 0,
});

const elements = {
  year: document.getElementById("year"),
  frequencyStatus: document.getElementById("frequencyStatus"),
  loadBar: document.getElementById("loadBar"),
  settingsForm: document.getElementById("settingsForm"),
  dropTarget: document.getElementById("dropTarget"),
  fileInput: document.getElementById("fileInput"),
  fileName: document.getElementById("fileName"),
  fileMeta: document.getElementById("fileMeta"),
  topN: document.getElementById("topN"),
  topNValue: document.getElementById("topNValue"),
  showCounts: document.getElementById("showCounts"),
  showContext: document.getElementById("showContext"),
  showUnmatched: document.getElementById("showUnmatched"),
  searchInput: document.getElementById("searchInput"),
  totalWords: document.getElementById("totalWords"),
  uniqueWords: document.getElementById("uniqueWords"),
  matchedWords: document.getElementById("matchedWords"),
  missingWords: document.getElementById("missingWords"),
  resultTitle: document.getElementById("resultTitle"),
  resultMeta: document.getElementById("resultMeta"),
  downloadRare: document.getElementById("downloadRare"),
  downloadMissing: document.getElementById("downloadMissing"),
  emptyState: document.getElementById("emptyState"),
  tableWrap: document.getElementById("tableWrap"),
  resultHead: document.getElementById("resultHead"),
  resultBody: document.getElementById("resultBody"),
  missingPanel: document.getElementById("missingPanel"),
  missingMeta: document.getElementById("missingMeta"),
  missingHead: document.getElementById("missingHead"),
  missingBody: document.getElementById("missingBody"),
};

const state = {
  fileText: "",
  fileName: "",
  fileBytes: 0,
  frequencyReady: false,
  currentJob: 0,
  results: null,
};

const worker = new Worker(new URL("./frequency-worker.js", import.meta.url), {
  type: "module",
});

elements.year.textContent = new Date().getFullYear();
elements.topNValue.value = elements.topN.value;

worker.addEventListener("message", (event) => {
  const message = event.data;

  if (message.type === "frequency-progress") {
    updateFrequencyProgress(message.loaded, message.total);
    return;
  }

  if (message.type === "frequency-ready") {
    state.frequencyReady = true;
    elements.frequencyStatus.textContent = `${numberFormat.format(message.rows)} frequency rows ready`;
    elements.loadBar.classList.add("is-ready");
    elements.loadBar.style.width = "100%";
    analyzeCurrentFile();
    return;
  }

  if (message.type === "analysis-result" && message.jobId === state.currentJob) {
    state.results = message.result;
    renderResults();
    return;
  }

  if (message.type === "analysis-error") {
    showEmpty(`Analysis failed: ${message.message}`);
  }

  if (message.type === "frequency-error") {
    elements.frequencyStatus.textContent = `Frequency list failed: ${message.message}`;
    elements.loadBar.style.width = "100%";
    elements.loadBar.style.background = "var(--red)";
  }
});

worker.postMessage({ type: "load-frequency" });

elements.fileInput.addEventListener("change", () => {
  const [file] = elements.fileInput.files;
  if (file) {
    loadFile(file);
  }
});

elements.dropTarget.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropTarget.classList.add("is-dragging");
});

elements.dropTarget.addEventListener("dragleave", () => {
  elements.dropTarget.classList.remove("is-dragging");
});

elements.dropTarget.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropTarget.classList.remove("is-dragging");
  const [file] = event.dataTransfer.files;
  if (file) {
    loadFile(file);
  }
});

for (const input of [
  elements.topN,
  elements.showCounts,
  elements.showContext,
  elements.showUnmatched,
]) {
  input.addEventListener("input", () => {
    elements.topNValue.value = elements.topN.value;
    analyzeCurrentFile();
  });
}

elements.searchInput.addEventListener("input", () => {
  renderResults();
});

elements.downloadRare.addEventListener("click", () => {
  if (!state.results) {
    return;
  }
  const columns = resultColumns();
  const csv = toCsv(state.results.rareRows, columns);
  downloadBlob(csv, fileStem("rare-words", "csv"), "text/csv;charset=utf-8");
});

elements.downloadMissing.addEventListener("click", () => {
  if (!state.results) {
    return;
  }
  const words = state.results.unmatchedRows.map((row) => row.word).join("\n");
  downloadBlob(`${words}\n`, fileStem("missing-words", "txt"), "text/plain;charset=utf-8");
});

async function loadFile(file) {
  state.fileName = file.name;
  state.fileBytes = file.size;
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${formatBytes(file.size)} text file`;
  showBusy("Reading file");

  try {
    state.fileText = await file.text();
    analyzeCurrentFile();
  } catch (error) {
    showEmpty(`Could not read ${file.name}: ${error.message}`);
  }
}

function analyzeCurrentFile() {
  if (!state.fileText) {
    return;
  }

  if (!state.frequencyReady) {
    showBusy("Waiting for frequency list");
    return;
  }

  const jobId = state.currentJob + 1;
  state.currentJob = jobId;
  showBusy("Analyzing text");

  worker.postMessage({
    type: "analyze",
    jobId,
    text: state.fileText,
    options: {
      topN: Number(elements.topN.value),
      includeContext: elements.showContext.checked,
      showUnmatched: elements.showUnmatched.checked,
    },
  });
}

function renderResults() {
  if (!state.results) {
    showEmpty(state.fileText ? "Analyzing text" : "No file selected.");
    return;
  }

  const stats = state.results.stats;
  elements.totalWords.textContent = numberFormat.format(stats.totalTokens);
  elements.uniqueWords.textContent = numberFormat.format(stats.uniqueWords);
  elements.matchedWords.textContent = numberFormat.format(stats.matchedWords);
  elements.missingWords.textContent = numberFormat.format(stats.unmatchedWords);
  elements.resultTitle.textContent = `Selected ${numberFormat.format(state.results.rareRows.length)} words`;
  elements.resultMeta.textContent = `${state.fileName} analyzed in ${Math.round(stats.elapsedMs)} ms`;

  const rows = filteredRows(state.results.rareRows);
  const columns = resultColumns();
  renderTable(elements.resultHead, elements.resultBody, columns, rows);

  elements.emptyState.hidden = rows.length > 0;
  elements.emptyState.textContent = state.results.rareRows.length > 0
    ? "No words match the current filter."
    : "No matched words found.";
  elements.emptyState.classList.remove("is-busy");
  elements.tableWrap.hidden = rows.length === 0;
  elements.downloadRare.disabled = state.results.rareRows.length === 0;
  elements.downloadMissing.disabled = state.results.unmatchedRows.length === 0;

  renderMissingTable();
}

function renderMissingTable() {
  if (!state.results || !elements.showUnmatched.checked || state.results.unmatchedRows.length === 0) {
    elements.missingPanel.hidden = true;
    return;
  }

  const rows = state.results.unmatchedRows;
  const columns = missingColumns();
  elements.missingMeta.textContent = numberFormat.format(rows.length);
  renderTable(elements.missingHead, elements.missingBody, columns, rows);
  elements.missingPanel.hidden = false;
}

function resultColumns() {
  const columns = [
    { key: "word", label: "Word" },
    { key: "frequency", label: "Frequency", format: formatNumber, className: "number-cell" },
  ];

  if (elements.showCounts.checked) {
    columns.splice(1, 0, { key: "count", label: "Count", format: formatNumber, className: "number-cell" });
  }

  if (elements.showContext.checked) {
    columns.push({ key: "context", label: "Context", className: "context-cell" });
  }

  return columns;
}

function missingColumns() {
  const columns = [
    { key: "word", label: "Word" },
    { key: "count", label: "Count", format: formatNumber, className: "number-cell" },
  ];

  if (elements.showContext.checked) {
    columns.push({ key: "context", label: "Context", className: "context-cell" });
  }

  return columns;
}

function filteredRows(rows) {
  const needle = elements.searchInput.value.trim().toLowerCase();
  if (!needle) {
    return rows;
  }
  return rows.filter((row) => row.word.includes(needle));
}

function renderTable(head, body, columns, rows) {
  head.replaceChildren();
  body.replaceChildren();

  const headerRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = column.label;
    headerRow.append(th);
  }
  head.append(headerRow);

  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      if (column.className) {
        td.className = column.className;
      }
      const value = row[column.key] ?? "";
      td.textContent = column.format ? column.format(value) : value;
      tr.append(td);
    }
    fragment.append(tr);
  }
  body.append(fragment);
}

function showBusy(message) {
  elements.emptyState.hidden = false;
  elements.emptyState.textContent = message;
  elements.emptyState.classList.add("is-busy");
  elements.tableWrap.hidden = true;
  elements.missingPanel.hidden = true;
  elements.downloadRare.disabled = true;
  elements.downloadMissing.disabled = true;
  elements.resultMeta.textContent = message;
}

function showEmpty(message) {
  elements.emptyState.hidden = false;
  elements.emptyState.textContent = message;
  elements.emptyState.classList.remove("is-busy");
  elements.tableWrap.hidden = true;
  elements.missingPanel.hidden = true;
  elements.downloadRare.disabled = true;
  elements.downloadMissing.disabled = true;
}

function updateFrequencyProgress(loaded, total) {
  if (!total) {
    elements.loadBar.style.width = "35%";
    return;
  }

  const percent = Math.max(8, Math.min(96, Math.round((loaded / total) * 100)));
  elements.loadBar.style.width = `${percent}%`;
  elements.frequencyStatus.textContent = `Loading frequency list ${percent}%`;
}

function formatNumber(value) {
  if (value === "" || value === null || value === undefined) {
    return "";
  }
  return numberFormat.format(Number(value));
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${numberFormat.format(bytes)} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${byteFormat.format(bytes / 1024)} KB`;
  }
  return `${byteFormat.format(bytes / 1024 / 1024)} MB`;
}

function toCsv(rows, columns) {
  const header = columns.map((column) => escapeCsv(column.label)).join(",");
  const lines = rows.map((row) =>
    columns.map((column) => escapeCsv(row[column.key] ?? "")).join(","),
  );
  return [header, ...lines].join("\n");
}

function escapeCsv(value) {
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function fileStem(kind, extension) {
  const source = state.fileName.replace(/\.[^.]+$/, "") || "text";
  const safe = source.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  return `${safe || "text"}-${kind}.${extension}`;
}
