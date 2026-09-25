import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)

class Blocker:
    def __init__(self, top_k=15):
        self.top_k = top_k
        # Limit max_features to drastically reduce math complexity and RAM
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=2, max_df=0.8, max_features=50000)

    def fit_transform(self, df_s1: pd.DataFrame, df_candidates: pd.DataFrame):
        """
        Creates blocking blocks by finding nearest neighbors, heavily optimized for memory.
        """
        # Fit vectorizer only on S1 to save memory and time
        s1_texts = (df_s1['business_name'].fillna('') + " " + df_s1['business_address'].fillna(''))
        self.vectorizer.fit(s1_texts)
        X_s1 = self.vectorizer.transform(s1_texts)
        
        n_samples = X_s1.shape[0]
        actual_k = min(self.top_k, len(df_candidates))
        
        best_scores = np.full((n_samples, actual_k), -1.0, dtype=np.float32)
        best_indices = np.zeros((n_samples, actual_k), dtype=int)
        
        cand_ids = df_candidates['entity_id'].values
        
        cand_chunk_size = 100000  # Extremely safe memory limit (prevents OOM on Kaggle)
        s1_batch_size = 500
        
        for cand_start in tqdm(range(0, len(df_candidates), cand_chunk_size), desc="Processing Candidate Chunks"):
            cand_end = min(cand_start + cand_chunk_size, len(df_candidates))
            
            cand_texts_chunk = (df_candidates['business_name'].iloc[cand_start:cand_end].fillna('') + " " + df_candidates['business_address'].iloc[cand_start:cand_end].fillna(''))
            X_cand_chunk = self.vectorizer.transform(cand_texts_chunk)
            X_cand_chunk_T = X_cand_chunk.T.tocsc()
            
            for s1_start in range(0, n_samples, s1_batch_size):
                s1_end = min(s1_start + s1_batch_size, n_samples)
                X_s1_batch = X_s1[s1_start:s1_end]
                
                # dot product
                sim_matrix = X_s1_batch.dot(X_cand_chunk_T).toarray()
                
                # Merge with running best
                current_best_scores = best_scores[s1_start:s1_end]
                current_best_indices = best_indices[s1_start:s1_end]
                
                combined_scores = np.hstack((current_best_scores, sim_matrix))
                new_indices = np.arange(cand_start, cand_end)
                new_indices_tiled = np.tile(new_indices, (s1_end - s1_start, 1))
                combined_indices = np.hstack((current_best_indices, new_indices_tiled))
                
                # Get the top K from the combined
                top_k_rel_idx = np.argpartition(combined_scores, -actual_k, axis=1)[:, -actual_k:]
                
                for i in range(s1_end - s1_start):
                    best_scores[s1_start + i] = combined_scores[i, top_k_rel_idx[i]]
                    best_indices[s1_start + i] = combined_indices[i, top_k_rel_idx[i]]
        
        # Sort final results
        s1_ids = df_s1['entity_id'].values
        candidate_pairs = []
        for i, s1_id in enumerate(s1_ids):
            # Sort the final top K by score
            row_scores = best_scores[i]
            sorted_k = np.argsort(-row_scores)
            final_indices = best_indices[i][sorted_k]
            
            matched_cand_ids = cand_ids[final_indices]
            cand_str = ",".join(matched_cand_ids)
            candidate_pairs.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': cand_str
            })
            
        return pd.DataFrame(candidate_pairs)

    def generate_candidate_pairs(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
        df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        all_candidate_pairs = []
        s1_countries = df_s1['country'].unique()
        
        for country in s1_countries:
            logger.info(f"Processing country: {country}")
            s1_subset = df_s1[df_s1['country'] == country]
            cand_subset = df_candidates[df_candidates['country'] == country]
            
            if len(cand_subset) == 0:
                for s1_id in s1_subset['entity_id']:
                    all_candidate_pairs.append({
                        'source1_entity_id': s1_id,
                        'candidate_entity_ids': ""
                    })
                continue
                
            blocker = Blocker(top_k=self.top_k)
            pairs_df = blocker.fit_transform(s1_subset, cand_subset)
            all_candidate_pairs.extend(pairs_df.to_dict('records'))
            
        return pd.DataFrame(all_candidate_pairs)
