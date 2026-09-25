import pandas as pd
import polars as pl
import re

class Preprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        # Remove non-alphanumeric noise, keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_df_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses a pandas DataFrame."""
        df = df.copy()
        
        for col in ['business_name', 'business_address']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
                
        # Keep country tags as-is, just handle NAs and lowercase
        if 'country' in df.columns:
            df['country'] = df['country'].fillna("unknown").str.lower().str.strip()
            
        return df

    def preprocess_df_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocesses a polars DataFrame."""
        # Polars expression for regex replace and strip
        
        # Clean business_name and business_address
        for col in ['business_name', 'business_address']:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col)
                    .fill_null("")
                    .str.to_lowercase()
                    .str.replace_all(r'[^a-z0-9\s]', ' ')
                    .str.replace_all(r'\s+', ' ')
                    .str.strip_chars()
                )
                
        if 'country' in df.columns:
            df = df.with_columns(
                pl.col('country')
                .fill_null("unknown")
                .str.to_lowercase()
                .str.strip_chars()
            )
            
        return df
