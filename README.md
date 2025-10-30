# Gutenberg Word-Frequency Explorer (Streamlit)

Streamlit app that highlights the rarest words in any uploaded `.txt` file by comparing it to word frequencies from a local CSV.

## Features
- Upload plain-text files and tokenize lowercase alphabetic words
- Look up corpus frequencies from a local CSV (no BigQuery required)
- Choose how many rare words to keep (10–2,000) and optionally display counts from your text
- Optional dictionary filter drops words that lack a WordNet definition
- Displays WordNet definitions alongside matched words when available
- Shows IPA pronunciations beside each word (lemmatized with CMU fallback)
- Optional context snippet shows each word's first occurrence with surrounding sentences
- Download the selected rare words (CSV) and unmatched words (TXT)

## Quick start (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```
The app opens at <http://localhost:8501>.

### Frequency CSV
- Place your frequency file at `data/frequencies.csv` or set `FREQ_CSV_PATH=/path/to/your.csv`.
- Expected columns: `word` and `frequency`.

## Deploying on Render
1. Create a new Web Service from this repo.
2. Environment: `Python`; build command: `pip install -r requirements.txt`.
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
4. If you store the CSV outside the repo, add env var:
   - `FREQ_CSV_PATH` -> `/opt/render/project/src/data/frequencies.csv` (or wherever you place it)
5. Ensure the CSV is available at build/runtime (committed in `data/` or uploaded as a secret file and referenced by `FREQ_CSV_PATH`).

## Files
- `app.py`: Streamlit UI and logic for tokenizing text, loading frequencies from CSV, and presenting results.
- `requirements.txt`: Minimal dependencies to run the app.
- `Procfile`: Process declaration for Render / Heroku-style platforms.
