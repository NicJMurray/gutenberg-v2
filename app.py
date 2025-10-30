import os
import re
from functools import lru_cache
from typing import Dict, List, Set

import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import eng_to_ipa as ipa
import pronouncing


# Ensure minimal NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)


# Frequency CSV configuration (permanent resource)
DEFAULT_FREQ_CSV = os.path.join(os.path.dirname(__file__), "data", "frequencies.csv")
FREQ_CSV_PATH = os.environ.get("FREQ_CSV_PATH", DEFAULT_FREQ_CSV)


lemmatizer = WordNetLemmatizer()

# ARPAbet to IPA mapping
ARPA_TO_IPA: Dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
    "AX": "ə",
    "AXR": "ɚ",
    "IX": "ɨ",
}

POS_PRIORITY = ("n", "v", "a", "r")


def candidate_lemmas(word: str) -> List[str]:
    base = word.lower()
    candidates: List[str] = [base]
    for pos in POS_PRIORITY:
        lemma = lemmatizer.lemmatize(base, pos)
        if lemma not in candidates:
            candidates.append(lemma)
    return candidates


def arpabet_to_ipa(phones: str) -> str:
    parts: List[str] = []
    for token in phones.split():
        stress = ""
        if token and token[-1].isdigit():
            d = token[-1]
            if d == "1":
                stress = "ˈ"
            elif d == "2":
                stress = "ˌ"
            token = token[:-1]
        ipa_val = ARPA_TO_IPA.get(token)
        if not ipa_val:
            return ""
        parts.append(f"{stress}{ipa_val}")
    return "".join(parts)


@lru_cache(maxsize=50000)
def get_definition(word: str) -> str:
    try:
        for syn in wn.synsets(word.lower()):
            definition = syn.definition().strip()
            if definition:
                return definition.replace("_", " ")
    except LookupError:
        return ""
    return ""


def _try_pronounce(token: str) -> str:
    converted = " ".join(ipa.convert(token).replace("*", "").split())
    if converted and converted.lower() != token.lower():
        return f"/{converted}/"
    for phones in pronouncing.phones_for_word(token):
        ipa_value = arpabet_to_ipa(phones)
        if ipa_value:
            return f"/{ipa_value}/"
    return ""


@lru_cache(maxsize=50000)
def get_pronunciation(word: str) -> str:
    # Try candidates (base + lemmatized)
    for cand in candidate_lemmas(word):
        val = _try_pronounce(cand)
        if val:
            return val
    # Backoff to shorter stem if possible (e.g., wanly -> wan)
    suffixes = ("ly", "ness", "ment", "ing", "ed", "er", "est", "ies", "es", "s")
    w = word.lower()
    for suf in suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            val = _try_pronounce(stem)
            if val:
                return val
    return ""


def build_context_map(words: Set[str], sentences: List[str]) -> Dict[str, str]:
    contexts: Dict[str, str] = {}
    remaining = set(words)
    if not remaining:
        return contexts

    patterns: Dict[str, re.Pattern[str]] = {}
    lowered = [s.lower() for s in sentences]
    for idx, s in enumerate(lowered):
        if not remaining:
            break
        found: List[str] = []
        for w in list(remaining):
            pat = patterns.get(w)
            if pat is None:
                pat = patterns[w] = re.compile(rf"\b{re.escape(w)}\b")
            if pat.search(s):
                prev_s = sentences[idx - 1].strip() if idx > 0 else ""
                cur_s = sentences[idx].strip()
                next_s = sentences[idx + 1].strip() if idx + 1 < len(sentences) else ""
                snippet = " ".join(p for p in (prev_s, cur_s, next_s) if p)
                contexts[w] = f'"{snippet}"' if snippet else ""
                found.append(w)
        for w in found:
            remaining.remove(w)
    return contexts


@lru_cache
def fetch_frequencies(words: tuple[str]) -> pd.DataFrame:
    """Return DataFrame(word, frequency) for supplied words using a local CSV file."""
    words = sorted({w.lower() for w in words})
    if not words:
        return pd.DataFrame(columns=["word", "frequency"])

    if not os.path.isfile(FREQ_CSV_PATH):
        st.error(
            f"Frequency CSV not found at '{FREQ_CSV_PATH}'. Place your CSV there or set FREQ_CSV_PATH."
        )
        st.stop()

    wanted = set(words)
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        FREQ_CSV_PATH,
        usecols=["word", "frequency"],
        chunksize=200_000,
    ):
        chunk["word"] = chunk["word"].astype(str).str.lower()
        sub = chunk[chunk["word"].isin(wanted)]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=["word", "frequency"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.groupby("word", as_index=False)["frequency"].min()
    return df


# Streamlit UI
st.set_page_config(page_title="Word Frequency Explorer", page_icon=":mag:", layout="wide")
st.title("Word Frequency Explorer")

st.markdown(
    """
<style>
[data-testid=\"stDataFrame\"] div[role=\"gridcell\"] { white-space: normal !important; }
[data-testid=\"stDataFrame\"] div[role=\"grid\"] { width: 100% !important; }
/* Expand main container to reduce margins */
main .block-container { max-width: 100% !important; padding-left: 1rem; padding-right: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    top_n = st.slider(
        "How many rare words to return?",
        min_value=10,
        max_value=2000,
        value=100,
        step=10,
        help="Select between 10 and 2,000 words",
    )
    show_user_counts = st.checkbox("Show counts from your text", value=True)
    omit_non_dictionary = st.checkbox("Omit words not in dictionary", value=False)
    include_context = st.checkbox("Show first-use context snippet", value=True)
    show_unmatched = st.checkbox(
        "Show unmatched words table",
        value=False,
        help="Compute and display words from your text that were not found in the frequency CSV."
    )

upload = st.file_uploader("Upload a plain-text (.txt) file", type=["txt"])

if upload:
    # Tokenize
    raw_text = upload.read().decode("utf-8", errors="ignore")
    text_lower = raw_text.lower()
    tokens = re.findall(r"[a-z]+", text_lower)
    if not tokens:
        st.warning("No alphabetic words found in the file.")
        st.stop()

    try:
        sentences = nltk.sent_tokenize(raw_text)
    except LookupError:
        # Fallback if punkt is unavailable: naive split on punctuation
        sentences = re.split(r"(?<=[.!?])\s+", raw_text)

    if omit_non_dictionary:
        original_count = len(tokens)
        tokens = [t for t in tokens if get_definition(t)]
        filtered_out = original_count - len(tokens)
        if not tokens:
            st.warning("No words remained after applying the dictionary filter.")
            st.stop()
        if filtered_out:
            st.caption(
                f"Dictionary filter removed {filtered_out} word(s) without dictionary definitions."
            )

    user_counts = (
        pd.Series(tokens).value_counts().rename_axis("word").reset_index(name="user_count")
    )

    # Fetch frequencies only for words present
    freq_df = fetch_frequencies(tuple(user_counts["word"]))
    joined = user_counts.merge(freq_df, on="word", how="left")

    matched = joined.dropna(subset=["frequency"]).copy()
    unmatched = joined[joined["frequency"].isna()].copy()

    # Pick N rarest (lowest frequency)
    rareN = matched.sort_values("frequency", ascending=True).head(top_n)
    rareN = rareN.assign(
        pronunciation=rareN["word"].map(get_pronunciation),
        definition=rareN["word"].map(get_definition),
    )
    rareN["pronunciation"] = rareN["pronunciation"].apply(
        lambda v: v if isinstance(v, str) and v.strip() else ""
    )
    rareN["definition"] = rareN["definition"].apply(
        lambda v: v if isinstance(v, str) and v.strip() else ""
    )

    if include_context:
        # Build context map for all words seen so we can apply to both tables
        context_map = build_context_map(set(joined["word"]), sentences)
        rareN = rareN.assign(context=rareN["word"].map(lambda w: context_map.get(w, "")))
        rareN["context"] = rareN["context"].apply(
            lambda v: v if isinstance(v, str) and v.strip() else ""
        )

    # Display main table
    st.subheader(f"Selected {top_n} words")
    # Column order: word, IPA, context, definition, count, frequency
    cols = ["word", "pronunciation"]
    if include_context:
        cols.append("context")
    cols.append("definition")
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

    # Unmatched words table (optional for performance)
    if show_unmatched and not unmatched.empty:
        st.subheader("Unmatched words (not found in frequency CSV)")
        st.caption(
            "These appeared in your text but not in the frequency CSV; treated as rarest by absence."
        )
        um = unmatched.assign(
            pronunciation=unmatched["word"].map(get_pronunciation),
            definition=unmatched["word"].map(get_definition),
        )
        um["pronunciation"] = um["pronunciation"].apply(
            lambda v: v if isinstance(v, str) and v.strip() else ""
        )
        um["definition"] = um["definition"].apply(
            lambda v: v if isinstance(v, str) and v.strip() else ""
        )
        if include_context:
            um = um.assign(context=um["word"].map(lambda w: context_map.get(w, "")))
            um["context"] = um["context"].apply(lambda v: v if isinstance(v, str) and v.strip() else "")
        # Column order for unmatched: word, IPA, context, definition, count
        um_cols = ["word", "pronunciation"]
        if include_context:
            um_cols.append("context")
        um_cols.append("definition")
        um_cols.append("user_count")
        st.dataframe(
            um[um_cols].sort_values("user_count", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=min(600, 200 + 30 * max(len(um), 1)),
            column_config={
                "pronunciation": st.column_config.TextColumn("IPA", width="medium"),
                "definition": st.column_config.TextColumn("Definition", width="large"),
                "user_count": st.column_config.NumberColumn("Count in your text"),
            },
        )
