# bq_loader.py
from functools import lru_cache
import pandas as pd
from google.cloud import bigquery

PROJECT = "words-466717"               #  <-- keep exactly this
TABLE   = "text_tools.top_5m_1grams"   #  <-- dataset.table name

@lru_cache
def load_top_5m() -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT)
    query = f"""
        SELECT word, frequency
        FROM `{PROJECT}.{TABLE}`
        ORDER BY frequency DESC
    """
    return client.query(query).result().to_dataframe()
