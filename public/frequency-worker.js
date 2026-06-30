let frequencyMap = null;
let frequencyMeta = null;
let loadingPromise = null;

self.addEventListener("message", (event) => {
  const message = event.data;

  if (message.type === "load-frequency") {
    loadFrequencyMap()
      .then(() => {
        self.postMessage({
          type: "frequency-ready",
          rows: frequencyMeta.rows,
          bytes: frequencyMeta.bytes,
        });
      })
      .catch((error) => {
        self.postMessage({ type: "frequency-error", message: error.message });
      });
    return;
  }

  if (message.type === "analyze") {
    handleAnalysis(message);
  }
});

async function handleAnalysis(message) {
  try {
    await loadFrequencyMap();
    const result = analyzeText(message.text, message.options);
    self.postMessage({
      type: "analysis-result",
      jobId: message.jobId,
      result,
    });
  } catch (error) {
    self.postMessage({
      type: "analysis-error",
      jobId: message.jobId,
      message: error.message,
    });
  }
}

async function loadFrequencyMap() {
  if (frequencyMap) {
    return frequencyMap;
  }

  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = loadFrequencyMapInternal();
  frequencyMap = await loadingPromise;
  return frequencyMap;
}

async function loadFrequencyMapInternal() {
  const manifestResponse = await fetch("data/manifest.json", { cache: "force-cache" });
  if (!manifestResponse.ok) {
    throw new Error(`manifest returned ${manifestResponse.status}`);
  }

  frequencyMeta = await manifestResponse.json();
  const csvUrl = new URL(frequencyMeta.file, manifestResponse.url);
  const csvResponse = await fetch(csvUrl, { cache: "force-cache" });
  if (!csvResponse.ok) {
    throw new Error(`frequency CSV returned ${csvResponse.status}`);
  }

  const map = new Map();
  let loaded = 0;

  if (csvResponse.body && typeof TextDecoderStream !== "undefined") {
    const reader = csvResponse.body.pipeThrough(new TextDecoderStream()).getReader();
    let carry = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      loaded += value.length;
      carry += value;
      const lines = carry.split(/\r?\n/);
      carry = lines.pop() ?? "";

      for (const line of lines) {
        parseFrequencyLine(line, map);
      }

      self.postMessage({
        type: "frequency-progress",
        loaded,
        total: frequencyMeta.bytes,
      });
    }

    if (carry) {
      parseFrequencyLine(carry, map);
    }
  } else {
    const csvText = await csvResponse.text();
    loaded = csvText.length;
    for (const line of csvText.split(/\r?\n/)) {
      parseFrequencyLine(line, map);
    }
    self.postMessage({
      type: "frequency-progress",
      loaded,
      total: frequencyMeta.bytes,
    });
  }

  return map;
}

function parseFrequencyLine(line, map) {
  const clean = line.charCodeAt(0) === 0xfeff ? line.slice(1) : line;
  if (!clean || clean === "word,frequency") {
    return;
  }

  const comma = clean.indexOf(",");
  if (comma <= 0) {
    return;
  }

  const word = clean.slice(0, comma).toLowerCase();
  const frequency = Number(clean.slice(comma + 1));
  if (word && Number.isFinite(frequency)) {
    map.set(word, frequency);
  }
}

function analyzeText(text, options) {
  const started = performance.now();
  const tokens = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const counts = new Map();

  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  const matchedRows = [];
  const unmatchedRows = [];

  for (const [word, count] of counts) {
    const frequency = frequencyMap.get(word);
    if (frequency === undefined) {
      unmatchedRows.push({ word, count });
    } else {
      matchedRows.push({ word, count, frequency });
    }
  }

  matchedRows.sort(
    (a, b) =>
      a.frequency - b.frequency ||
      b.count - a.count ||
      a.word.localeCompare(b.word),
  );
  unmatchedRows.sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));

  const rareRows = matchedRows.slice(0, options.topN);
  const visibleUnmatchedRows = options.showUnmatched ? unmatchedRows.slice(0, 2000) : [];

  if (options.includeContext) {
    addContexts(text, [...rareRows, ...visibleUnmatchedRows]);
  }

  return {
    rareRows,
    unmatchedRows: visibleUnmatchedRows,
    stats: {
      totalTokens: tokens.length,
      uniqueWords: counts.size,
      matchedWords: matchedRows.length,
      unmatchedWords: unmatchedRows.length,
      elapsedMs: performance.now() - started,
    },
  };
}

function addContexts(text, rows) {
  if (rows.length === 0) {
    return;
  }

  const sentences = splitSentences(text);
  const lowered = sentences.map((sentence) => sentence.toLowerCase());
  const remaining = new Map(rows.map((row) => [row.word, row]));

  for (let index = 0; index < lowered.length && remaining.size > 0; index += 1) {
    for (const [word, row] of remaining) {
      if (hasWholeWord(lowered[index], word)) {
        const snippet = [
          sentences[index - 1] ?? "",
          sentences[index] ?? "",
          sentences[index + 1] ?? "",
        ]
          .map((part) => part.trim())
          .filter(Boolean)
          .join(" ");
        row.context = snippet;
        remaining.delete(word);
      }
    }
  }

  for (const row of rows) {
    row.context = row.context ?? "";
  }
}

function splitSentences(text) {
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
    return [...segmenter.segment(text)].map((segment) => segment.segment);
  }

  return text.split(/(?<=[.!?])\s+/);
}

function hasWholeWord(sentence, word) {
  const index = sentence.indexOf(word);
  if (index === -1) {
    return false;
  }

  let cursor = index;
  while (cursor !== -1) {
    const before = cursor === 0 ? "" : sentence[cursor - 1];
    const after = sentence[cursor + word.length] ?? "";
    if (!isAsciiLetter(before) && !isAsciiLetter(after)) {
      return true;
    }
    cursor = sentence.indexOf(word, cursor + 1);
  }

  return false;
}

function isAsciiLetter(value) {
  return value >= "a" && value <= "z";
}
