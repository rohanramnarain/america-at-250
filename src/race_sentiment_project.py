import os
import re
import math
from pathlib import Path

import requests
import pdfplumber
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers import pipeline


# ---------- Paths & directories ----------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"

for d in [RAW_PDF_DIR, PROCESSED_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------- Textbook corpus configuration (all CC/open) ----------

TEXTBOOK_PDFS = [
    {
        "id": "openstax_us_history",
        "title": "U.S. History (OpenStax)",
        # OpenStax OER PDF mirror
        "url": "https://openlibrary-repo.ecampusontario.ca/jspui/bitstream/123456789/426/1/USHistory-LR.pdf",  # 
        "pub_year": 2014,
    },
    {
        "id": "history_in_the_making",
        "title": "History in the Making: U.S. to 1877",
        "url": "https://ung.edu/university-press/_uploads/files/us-history/US-History-I-Full-Text%20.pdf",  # 
        "pub_year": 2013,
    },
    {
        "id": "american_yawp_v1",
        "title": "The American Yawp, Vol. I: To 1877",
        "url": "https://www.americanyawp.com/text/wp-content/uploads/Locke_American-Yawp_V1.pdf",  # 
        "pub_year": 2019,
    },
    {
        "id": "american_yawp_v2",
        "title": "The American Yawp, Vol. II: Since 1877",
        "url": "https://ia803205.us.archive.org/11/items/locke-american-yawp-v-2/Locke_American-Yawp_V2.pdf",  # 
        "pub_year": 2019,
    },
    # You can add additional OER US-history PDFs here later.
]


# ---------- Step 1: Download PDFs ----------

def download_textbook_pdfs(textbook_list=TEXTBOOK_PDFS, target_dir=RAW_PDF_DIR):
    """Download all configured textbooks into data/raw_pdfs."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for tb in textbook_list:
        fname = f"{tb['id']}.pdf"
        out_path = target_dir / fname
        if out_path.exists():
            print(f"Already have {fname}")
            continue
        if not tb.get("url"):
            print(f"No URL configured for {tb['id']}, please add PDF manually.")
            continue

        print(f"Downloading {tb['title']} ...")
        resp = requests.get(tb["url"], stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {out_path}")


# ---------- Step 2: Extract text & approximate historical year ----------

# Match year ranges like "1800–1860" or "1865-77"
YEAR_RANGE_REGEX = re.compile(r"(\d{3,4})\s*[–-]\s*(\d{2,4})")


def infer_year_from_text(text: str, fallback_year: int) -> int:
    """
    Find the last year-range in the text and return its midpoint.
    If none found, return fallback_year.
    """
    matches = YEAR_RANGE_REGEX.findall(text)
    if matches:
        start_str, end_str = matches[-1]
        start = int(start_str)

        # Handle abbreviated end years like "77" meaning "1877" if start is 18xx
        if len(end_str) == 2 and len(start_str) == 4:
            century = start // 100
            end = century * 100 + int(end_str)
        else:
            end = int(end_str)

        year = int((start + end) / 2)
        return year

    return fallback_year


def extract_sentences_from_pdf(pdf_path: Path, book_id: str, book_title: str, pub_year: int):
    """
    Extract page-level text, infer a historical year per page,
    and split into sentences.
    """
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        last_year = pub_year
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip():
                continue

            inferred_year = infer_year_from_text(text, last_year)
            last_year = inferred_year

            sentences = sent_tokenize(text)
            for sent in sentences:
                clean_sent = " ".join(sent.split())
                if not clean_sent:
                    continue
                records.append(
                    {
                        "book_id": book_id,
                        "book_title": book_title,
                        "pub_year": pub_year,
                        "hist_year": inferred_year,
                        "page": page_num,
                        "sentence": clean_sent,
                    }
                )
    return records


# ---------- Race lexicon & detection ----------

# NOTE: These terms include historically used labels that can be offensive today,
# because you are explicitly interested in how textbooks' language changes over time.

RACE_TERMS = {
    "Black/African American": [
        "african american",
        "african-american",
        "african americans",
        "black american",
        "black americans",
        "black people",
        "enslaved africans",
        "freedmen",
        "freedman",
        "freedwoman",
        "negro",
        "negroes",
        "colored people",
    ],
    "Indigenous/Native": [
        "native american",
        "native americans",
        "american indian",
        "american indians",
        "indian tribes",
        "indigenous people",
        "indigenous peoples",
        "first nations",
        "first peoples",
        "tribal nations",
    ],
    "Asian/Asian American": [
        "asian american",
        "asian americans",
        "chinese immigrant",
        "chinese immigrants",
        "japanese american",
        "japanese americans",
        "filipino american",
        "filipino americans",
        "korean american",
        "korean americans",
        "asian immigrants",
    ],
    "Latino/Hispanic": [
        "latino",
        "latina",
        "latinos",
        "latinas",
        "hispanic",
        "hispanics",
        "mexican american",
        "mexican americans",
        "chicano",
        "chicana",
        "chicanos",
        "chicanas",
        "puerto rican",
        "puerto ricans",
        "cuban american",
        "cuban americans",
    ],
    "White/European American": [
        "white american",
        "white americans",
        "white people",
        "whites",
        "european american",
        "european americans",
        "anglo-american",
        "anglo americans",
        "white settlers",
        "european settlers",
    ],
}


def detect_races(sentence: str, race_terms=RACE_TERMS):
    """Return a list of race categories explicitly mentioned in the sentence."""
    s = sentence.lower()
    found = set()
    for race, terms in race_terms.items():
        for term in terms:
            if term in s:
                found.add(race)
                break
    return list(found)


# ---------- Sentiment models: VADER + DistilBERT ----------

def ensure_nltk():
    """Download required NLTK resources if missing."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab/english.pickle")
    except LookupError:
        nltk.download("punkt_tab")
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")


ensure_nltk()

# VADER (rule-based) 
sia = SentimentIntensityAnalyzer()

# DistilBERT sentiment model (binary POSITIVE/NEGATIVE) 
TRF_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
trf_sent = pipeline("sentiment-analysis", model=TRF_MODEL_NAME)


def score_sentence_both_models(sentence: str):
    """
    Return:
      - VADER compound score in [-1, 1]
      - transformer label ("POSITIVE"/"NEGATIVE")
      - transformer confidence
      - transformer signed score in [-1, 1] (POS -> +conf, NEG -> -conf)
    """
    vader_score = sia.polarity_scores(sentence)["compound"]

    # DistilBERT inference (truncate very long sentences)
    out = trf_sent(sentence[:512])[0]
    label = out["label"]
    conf = float(out["score"])

    signed = conf if label.upper().startswith("POS") else -conf
    return vader_score, label, conf, signed


# ---------- Add race + sentiment to each sentence ----------

def add_sentiment_and_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sentence, detect race mentions and score sentiment with VADER + transformer.
    Each (sentence, race) pair becomes one row.
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sent = row["sentence"]
        races = detect_races(sent)
        if not races:
            continue  # skip sentences that don't explicitly mention any race terms

        vader_score, trf_label, trf_conf, trf_signed = score_sentence_both_models(sent)

        for race in races:
            rows.append(
                {
                    "book_id": row["book_id"],
                    "book_title": row["book_title"],
                    "pub_year": row["pub_year"],
                    "hist_year": row["hist_year"],
                    "page": row["page"],
                    "race": race,
                    "sentence": sent,
                    "sentiment_vader": vader_score,
                    "trf_label": trf_label,
                    "trf_confidence": trf_conf,
                    "sentiment_trf_signed": trf_signed,
                }
            )

    return pd.DataFrame(rows)


# ---------- Aggregation & visualization ----------

def to_decade(y):
    if pd.isna(y):
        return None
    return int(math.floor(y / 10.0) * 10)


def main():
    # 1. Download PDFs
    download_textbook_pdfs()

    # 2. Extract sentences from each textbook
    all_records = []
    for tb in TEXTBOOK_PDFS:
        pdf_path = RAW_PDF_DIR / f"{tb['id']}.pdf"
        if not pdf_path.exists():
            print(f"Missing PDF for {tb['id']} – expected at {pdf_path}")
            continue

        print(f"Extracting from {pdf_path.name} ...")
        recs = extract_sentences_from_pdf(
            pdf_path,
            book_id=tb["id"],
            book_title=tb["title"],
            pub_year=tb["pub_year"],
        )
        all_records.extend(recs)

    sent_df = pd.DataFrame(all_records)
    sent_df.to_csv(PROCESSED_DIR / "all_sentences_raw.csv", index=False)
    print(f"Extracted {len(sent_df)} sentences total")

    # 3. Add race tags + both sentiment scores
    race_sent_df = add_sentiment_and_race(sent_df)
    race_sent_df.to_csv(
        PROCESSED_DIR / "race_sentences_with_sentiment_both_models.csv",
        index=False,
    )
    print(f"Kept {len(race_sent_df)} race-mention sentences")

    # 4. Aggregate to race × decade
    race_sent_df["decade"] = race_sent_df["hist_year"].apply(to_decade)
    race_sent_df = race_sent_df[
        (race_sent_df["decade"].notna()) & (race_sent_df["decade"] >= 1770)
    ]

    # VADER-based time series (main figure)
    ts_vader = (
        race_sent_df.groupby(["race", "decade"])["sentiment_vader"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_vader": "sentiment"})
        .sort_values(["race", "decade"])
    )

    # Transformer-based time series (for robustness / optional appendix)
    ts_trf = (
        race_sent_df.groupby(["race", "decade"])["sentiment_trf_signed"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_trf_signed": "sentiment"})
        .sort_values(["race", "decade"])
    )

    ts_vader.to_csv(
        PROCESSED_DIR / "race_sent_timeseries_vader.csv", index=False
    )
    ts_trf.to_csv(
        PROCESSED_DIR / "race_sent_timeseries_transformer.csv", index=False
    )

    ts_vader_export = ts_vader.rename(columns={"sentiment": "sentiment_vader"})
    ts_trf_export = ts_trf.rename(
        columns={"sentiment": "sentiment_transformer"}
    )
    ts_counts = (
        race_sent_df.groupby(["race", "decade"])
        .size()
        .reset_index(name="n_sentences")
    )
    ts_combined = (
        ts_vader_export.merge(ts_trf_export, on=["race", "decade"], how="outer")
        .merge(ts_counts, on=["race", "decade"], how="outer")
        .sort_values(["race", "decade"])
    )
    ts_combined.to_csv(
        PROCESSED_DIR / "race_sent_timeseries_combined.csv", index=False
    )

    # 5. Visualization: main VADER figure
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for race, sub in ts_vader.groupby("race"):
        sub = sub.sort_values("decade")
        plt.plot(sub["decade"], sub["sentiment"], marker="o", label=race)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Approximate historical decade")
    plt.ylabel("Average sentiment (VADER compound score)")
    plt.title(
        "Sentiment in U.S. History Textbooks When Describing Racial Groups Over Time\n"
        "(VADER sentiment)"
    )
    plt.legend(title="Racial group")
    plt.tight_layout()

    out_path = FIG_DIR / "race_sentiment_timeseries_vader.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved VADER figure to {out_path}")

    # Transformer-based visualization for quick comparison with VADER
    plt.figure(figsize=(10, 6))
    for race, sub in ts_trf.groupby("race"):
        sub = sub.sort_values("decade")
        plt.plot(sub["decade"], sub["sentiment"], marker="o", label=race)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Approximate historical decade")
    plt.ylabel("Average sentiment (signed transformer score)")
    plt.title(
        "Sentiment in U.S. History Textbooks When Describing Racial Groups Over Time\n"
        "(DistilBERT SST-2 sentiment)"
    )
    plt.legend(title="Racial group")
    plt.tight_layout()

    out_path2 = FIG_DIR / "race_sentiment_timeseries_transformer.png"
    plt.savefig(out_path2, dpi=300)
    print(f"Saved transformer figure to {out_path2}")


if __name__ == "__main__":
    main()
