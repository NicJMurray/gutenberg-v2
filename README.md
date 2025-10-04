# Gutenberg Word-Frequency Explorer (Streamlit)

Streamlit app that highlights the rarest words in any uploaded `.txt` file by comparing it to the `words-466717.text_tools.top_5m_1grams` table in BigQuery.

## Features
- Upload plain-text files and tokenize lowercase alphabetic words
- Query Google BigQuery for corpus frequencies (defaults to the `words-466717.text_tools.top_5m_1grams` table)
- Choose how many rare words to keep (10–2000) and optionally display counts from your text
- Optional dictionary filter drops words that lack a WordNet definition
- Displays WordNet definitions alongside matched words when available
- Shows IPA pronunciations beside each word (lemmatized with CMU fallback, em dash when unavailable)
- Optional context snippet shows each word's first occurrence with surrounding sentences
- Download the selected rare words (CSV) and unmatched words (TXT)

## Quick start (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
streamlit run app.py
```
The app will open at <http://localhost:8501>. Upload a text file, pick how many rare words to show, and download the CSV/TXT outputs.

## Deploying on Render
1. Create a new **Web Service** from this repo.
2. Set the environment to `Python` and use the build command `pip install -r requirements.txt`.
3. Use the start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` (Render reads the `Procfile` automatically if you prefer).
4. Add environment variables:
   - `GOOGLE_APPLICATION_CREDENTIALS` → `/opt/render/project/src/service-account.json` (or any path you copy your credential file to at deploy time).
5. Upload the service-account JSON as a Render secret file, then reference it from the env var.

## Files
- `app.py`: Streamlit UI and logic for tokenizing text, querying BigQuery, and presenting results.
- `bq_loader.py`: Alternate helper for pulling the full 5M-word table if needed.
- `requirements.txt`: Minimal dependencies to run the app.
- `Procfile`: Process declaration for Render / Heroku-style platforms.
