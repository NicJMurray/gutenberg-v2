# Gutenberg Rare Word Explorer

Personal tool at `rare-words.njmurray.com` for finding the least frequent English words in an uploaded text file. All text analysis happens in the browser; the server only delivers the app and its frequency data.

## How it works

1. The source corpus is `data/frequencies.csv`, a word-to-frequency list.
2. `scripts/build.mjs` copies that file into `public/data/` with a content hash in its filename and writes `public/data/manifest.json`. The manifest tells the browser which immutable CSV file to load and how large it is.
3. `public/app.js` starts `public/frequency-worker.js` and asks it to load the manifest and frequency CSV.
4. The Web Worker streams the CSV where the browser supports streams, builds an in-memory `Map<word, frequency>`, and reports load progress to the page. The map stays in the worker rather than blocking the UI thread.
5. When a `.txt` file is dropped or selected, the page reads it locally and sends its text plus the selected options to the worker.
6. The worker lowercases the text, tokenises only `[a-z]+` words, counts unique words, looks each word up in the frequency map, then sorts matched words from rarest to commonest. It returns the requested top N plus, optionally, unmatched words and a first-use sentence context.
7. The page renders the table, allows an in-table filter, and creates CSV/TXT downloads entirely in the browser.

No uploaded text is sent to the Worker or any third-party service.

## Code map

| File | Responsibility |
| --- | --- |
| `data/frequencies.csv` | Canonical frequency dataset. |
| `scripts/build.mjs` | Creates the hashed public CSV and manifest from the canonical dataset. |
| `public/app.js` | Page state, file input/drop zone, controls, tables, and downloads. |
| `public/frequency-worker.js` | Frequency-map loading, tokenisation, counting, ranking, and context extraction. |
| `public/index.html` / `public/styles.css` | Static interface. |
| `src/worker.js` | Cloudflare Worker that serves static assets and falls back to `index.html` for extensionless routes. |
| `wrangler.toml` | Worker entry point and custom-domain route. |

## Interpretation and limits

- A smaller frequency number means the word is rarer in the source corpus.
- Words not found in the dataset are reported as unmatched; they are not automatically “rarer” than matched words. They may be names, typos, inflections, or simply absent from the corpus.
- Tokenisation is deliberately simple and English-oriented. Numbers, apostrophes, hyphens, accented characters, and non-Latin scripts are not treated as part of a word.
- Context is a convenience feature: it finds the first sentence containing each selected word and includes its neighbouring sentences.

## Things worth changing

- Replace or extend `data/frequencies.csv` to change the reference corpus. The build step will generate a new hashed asset automatically.
- Adjust the tokeniser in `frequency-worker.js` if the tool needs to preserve apostrophes, hyphenated words, or non-English text.
- Adjust `visibleUnmatchedRows` in the worker if the unmatched-word cap of 2,000 becomes restrictive.
