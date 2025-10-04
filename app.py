# app.py  – Streamlit + BigQuery (select N rare words)
# ─────────────────────────────────────────────────────
import re
from functools import lru_cache

import pandas as pd
import streamlit as st
from google.cloud import bigquery
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import eng_to_ipa as ipa
import pronouncing

# ─── BigQuery connection settings ────────────────────
BQ_PROJECT = "words-466717"
BQ_TABLE   = "text_tools.top_5m_1grams"
bq_client  = bigquery.Client(project=BQ_PROJECT)

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


lemmatizer = WordNetLemmatizer()
ARPA_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ", "AX": "ə", "AXR": "ɚ", "IX": "ɨ"
}

POS_PRIORITY = ("n", "v", "a", "r")

def candidate_lemmas(word: str) -> list[str]:
    base = word.lower()
    candidates: list[str] = []
    if base not in candidates:
        candidates.append(base)
    for pos in POS_PRIORITY:
        lemma = lemmatizer.lemmatize(base, pos)
        if lemma not in candidates:
            candidates.append(lemma)
    return candidates


def arpabet_to_ipa(phones: str) -> str:
    ipa_parts: list[str] = []
    for token in phones.split():
        stress = ''
        if token and token[-1].isdigit():
            digit = token[-1]
            if digit == '1':
                stress = 'ˈ'
            elif digit == '2':
                stress = 'ˌ'
            token = token[:-1]
        base = token
        if base in ("AH", "ER") and stress == '':
            # treat unstressed schwa sounds more gently
            ipa_value = 'ə' if base == 'AH' else 'ɚ'
        else:
            ipa_value = ARPA_TO_IPA.get(base)
        if not ipa_value:
            return ''
        ipa_parts.append(f"{stress}{ipa_value}")
    return ''.join(ipa_parts)

@lru_cache(maxsize=50000)
def get_definition(word: str) -> str:
    """Return a short WordNet definition for the supplied word, if available."""
    word = word.lower()
    for synset in wn.synsets(word):
        definition = synset.definition().strip()
        if definition:
            return definition.replace('_', ' ')
    return ''

@lru_cache(maxsize=50000)
def get_pronunciation(word: str) -> str:
    """Return IPA pronunciation using eng_to_ipa with CMU fallback."""
    for candidate in candidate_lemmas(word):
        converted = ' '.join(ipa.convert(candidate).replace('*', '').split())
        if converted and converted.lower() != candidate.lower():
            return f'/{converted}/'

        for phones in pronouncing.phones_for_word(candidate):
            ipa_value = arpabet_to_ipa(phones)
            if ipa_value:
                return f'/{ipa_value}/'
    return '—'


def build_context_map(words: set[str], sentences: list[str]) -> dict[str, str]:
    """Return mapping of word to first-use context snippet."""
    contexts: dict[str, str] = {}
    remaining = set(words)
    if not remaining:
        return contexts

    patterns: dict[str, re.Pattern[str]] = {}
    lowered_sentences = [s.lower() for s in sentences]

    for idx, sentence_lower in enumerate(lowered_sentences):
        if not remaining:
            break
        to_remove: list[str] = []
        for word in list(remaining):
            pattern = patterns.get(word)
            if pattern is None:
                patterns[word] = pattern = re.compile(rf"\b{re.escape(word)}\b")
            if pattern.search(sentence_lower):
                prev_sentence = sentences[idx - 1].strip() if idx > 0 else ''
                curr_sentence = sentences[idx].strip()
                next_sentence = sentences[idx + 1].strip() if idx + 1 < len(sentences) else ''
                snippet = ' '.join(part for part in (prev_sentence, curr_sentence, next_sentence) if part)
                contexts[word] = f'"{snippet}"' if snippet else ''
                to_remove.append(word)
        for word in to_remove:
            remaining.remove(word)
    return contexts

@lru_cache
def fetch_frequencies(words: tuple[str]) -> pd.DataFrame:
    """Return DataFrame(word, frequency) for supplied words."""
    words = sorted({w.lower() for w in words})
    if not words:
        return pd.DataFrame(columns=["word", "frequency"])

    sql = f"""
        SELECT word, frequency
        FROM `{BQ_PROJECT}.{BQ_TABLE}`
        WHERE word IN UNNEST(@words)
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("words", "STRING", words)]
    )
    return bq_client.query(sql, job_cfg).result().to_dataframe()


# ─── Streamlit UI ─────────────────────────────────────
st.set_page_config(page_title="Word‑Frequency Explorer", page_icon="📚")
st.title("Word‑Frequency Explorer 📚")

st.markdown("""
<style>
[data-testid="stDataFrame"] div[role="gridcell"] {
    white-space: normal !important;
}
[data-testid="stDataFrame"] div[role="grid"] {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Settings")
    top_n = st.slider(
        "How many rare words to return?",
        min_value=10,
        max_value=2000,
        value=100,
        step=10,
        help="Select between 10 and 2 000 words",
    )
    show_user_counts = st.checkbox("Show counts from your text", value=True)
    omit_non_dictionary = st.checkbox("Omit words not in dictionary", value=False)
    include_context = st.checkbox("Show first-use context snippet", value=False)

upload = st.file_uploader("Upload a plain‑text (.txt) file", type=["txt"])

if upload:
    # ── tokenize ──
    raw_text = upload.read().decode("utf-8", errors="ignore")
    text_lower = raw_text.lower()
    tokens = re.findall(r"[a-z]+", text_lower)
    if not tokens:
        st.warning("No alphabetic words found in the file.")
        st.stop()

    sentences = nltk.sent_tokenize(raw_text)

    if omit_non_dictionary:
        original_count = len(tokens)
        tokens = [t for t in tokens if get_definition(t)]
        filtered_out = original_count - len(tokens)
        if not tokens:
            st.warning("No words remained after applying the dictionary filter.")
            st.stop()
        if filtered_out:
            st.caption(f"Dictionary filter removed {filtered_out} word(s) without dictionary definitions.")

    user_counts = (
        pd.Series(tokens)
        .value_counts()
        .rename_axis("word")
        .reset_index(name="user_count")
    )

    # ── fetch frequencies only for words present ──
    freq_df = fetch_frequencies(tuple(user_counts["word"]))
    joined  = user_counts.merge(freq_df, on="word", how="left")

    matched   = joined.dropna(subset=["frequency"]).copy()
    unmatched = joined[joined["frequency"].isna()].copy()

    # ── pick N rarest (lowest frequency) ──
    rareN = matched.sort_values("frequency", ascending=True).head(top_n)
    rareN = rareN.assign(
        pronunciation=rareN["word"].map(get_pronunciation),
        definition=rareN["word"].map(get_definition),
    )
    rareN["pronunciation"] = rareN["pronunciation"].apply(lambda val: val if isinstance(val, str) and val.strip() else "—")
    rareN["definition"] = rareN["definition"].apply(lambda val: val if isinstance(val, str) and val.strip() else "—")

    if include_context:
        context_map = build_context_map(set(rareN["word"]), sentences)
        rareN = rareN.assign(context=rareN["word"].map(lambda w: context_map.get(w, "")))
        rareN["context"] = rareN["context"].apply(lambda val: val if isinstance(val, str) and val.strip() else "—")

    # ─── display ───
    st.subheader(f"Selected {top_n} words")
    cols = ["word", "pronunciation", "definition"]
    if include_context:
        cols.append("context")
    if show_user_counts:
        cols.append("user_count")
    cols.append("frequency")

    column_config = {
        "pronunciation": st.column_config.TextColumn("IPA", width="medium"),
        "definition": st.column_config.TextColumn("Definition", width="large"),
    }
    if include_context:
        column_config["context"] = st.column_config.TextColumn("Context", width="large")

    table_height = min(720, 200 + 36 * max(len(rareN), 1))

    st.dataframe(
        rareN[cols],
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_config=column_config,
    )

    st.subheader("Words not found in the 5 M‑word list")
    st.write(", ".join(unmatched["word"].tolist()) or "—")

    # ── downloads ──
    st.download_button(
        f"Download selected {top_n} (CSV)",
        rareN.to_csv(index=False).encode(),
        file_name=f"rare_{top_n}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download unmatched words (TXT)",
        "\n".join(unmatched["word"]).encode(),
        file_name="unmatched.txt",
        mime="text/plain",
    )
else:
    st.info("⬆️ Upload a .txt file to begin.")
