import pandas as pd
import polars as pl
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_source_data(self, split: str, source_id: int, use_polars=False):
        """
        Loads data from TSV files. 
        split: 'train' or 'test'
        source_id: 1, 2, or 3
        """
        file_path = self.data_dir / split / f"{split}_source{source_id}.tsv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Loading {file_path}")
        
        # Load TSV strictly with sep="\t"
        if use_polars:
            # Polars is faster for large files
            df = pl.read_csv(
                file_path,
                separator="\t",
                quote_char=None,  # Do not parse quotes
                ignore_errors=True,
                infer_schema_length=10000
            )
            # Standardize columns to string
            df = df.cast({col: pl.Utf8 for col in df.columns})
            return df
        else:
            # Pandas fallback
            df = pd.read_csv(
                file_path, 
                sep="\t",
                dtype=str,
                quoting=3, # csv.QUOTE_NONE
                on_bad_lines="skip"
            )
            return df

    def load_all_sources(self, split: str, use_polars=False):
        """Loads all 3 sources for a given split"""
        df1 = self.load_source_data(split, 1, use_polars)
        df2 = self.load_source_data(split, 2, use_polars)
        df3 = self.load_source_data(split, 3, use_polars)
        return df1, df2, df3
