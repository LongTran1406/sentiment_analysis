import pandas as pd
from sqlalchemy import text
from db.connect import get_engine

def create_table():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id SERIAL PRIMARY KEY,
                comment_text TEXT,
                toxicity FLOAT,
                obscene FLOAT,
                sexual_explicit FLOAT,
                identity_attack FLOAT,
                insult FLOAT,
                threat FLOAT
            )
        """))

def insert_sample_data(df):
    engine = get_engine()
    df.to_sql("sentiment_data", engine, if_exists="append", index=False)

def fetch_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM sentiment_data", engine)
    return df
