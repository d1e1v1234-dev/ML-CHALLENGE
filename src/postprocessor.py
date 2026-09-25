import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class Postprocessor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_candidates(self, df_candidates: pd.DataFrame, df_s1: pd.DataFrame):
        """
        df_candidates should have columns ['source1_entity_id', 'candidate_entity_ids']
        Must ensure every s1_id is present.
        """
        logger.info("Writing candidate_pairs.tsv")
        
        # Ensure all S1 entities are present
        s1_ids = df_s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
        
        # Merge with candidates, filling missing with empty string
        final_df = pd.merge(s1_ids, df_candidates, on='source1_entity_id', how='left')
        final_df['candidate_entity_ids'] = final_df['candidate_entity_ids'].fillna("")
        
        out_path = os.path.join(self.output_dir, "candidate_pairs.tsv")
        # Ensure no quotes around IDs, sep is \t
        final_df.to_csv(out_path, sep='\t', index=False, quoting=3) # quoting=3 is csv.QUOTE_NONE
        logger.info(f"Saved {out_path}")

    def write_matching_results(self, df_preds: pd.DataFrame, df_s1: pd.DataFrame, threshold: float):
        """
        df_preds: DataFrame with ['source1_entity_id', 'candidate_entity_id', 'pred_prob']
        """
        logger.info("Writing matching_results.tsv")
        
        # Filter by threshold
        df_matches = df_preds[df_preds['pred_prob'] >= threshold]
        
        # Group by S1 id
        grouped = df_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ",".join(x)).reset_index()
        grouped.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
        
        # Ensure all S1 entities are present
        s1_ids = df_s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
        
        final_df = pd.merge(s1_ids, grouped, on='source1_entity_id', how='left')
        final_df['matched_entity_ids'] = final_df['matched_entity_ids'].fillna("")
        
        out_path = os.path.join(self.output_dir, "matching_results.tsv")
        final_df.to_csv(out_path, sep='\t', index=False, quoting=3)
        logger.info(f"Saved {out_path}")
