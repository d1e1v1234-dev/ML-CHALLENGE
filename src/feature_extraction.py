import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    def __init__(self):
        pass
        
    def _safe_str(self, val):
        if pd.isna(val) or val is None:
            return ""
        return str(val)

    def _extract_digits(self, text):
        return set(re.findall(r'\d+', self._safe_str(text)))

    def _ngram_jaccard(self, s1, s2, n=3):
        s1, s2 = self._safe_str(s1), self._safe_str(s2)
        if not s1 or not s2:
            return 0.0
        # Padding
        s1 = ' ' + s1 + ' '
        s2 = ' ' + s2 + ' '
        ngrams1 = set([s1[i:i+n] for i in range(len(s1)-n+1)])
        ngrams2 = set([s2[i:i+n] for i in range(len(s2)-n+1)])
        if not ngrams1 and not ngrams2:
            return 1.0
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        return len(intersection) / len(union) if union else 0.0

    def extract_features(self, df_pairs: pd.DataFrame, df_s1: pd.DataFrame, df_candidates: pd.DataFrame):
        """
        df_pairs: DataFrame with 'source1_entity_id' and 'candidate_entity_ids' (comma-separated)
        We need to explode this to (s1_id, cand_id) pairs first.
        """
        # 1. Explode candidate pairs using pandas vectorized operations
        df_exploded = df_pairs.copy()
        
        # Handle NA and empty strings
        df_exploded['candidate_entity_ids'] = df_exploded['candidate_entity_ids'].fillna("")
        df_exploded = df_exploded[df_exploded['candidate_entity_ids'] != ""]
        
        # Split and explode
        df_exploded['candidate_entity_id'] = df_exploded['candidate_entity_ids'].str.split(',')
        df_exploded = df_exploded.explode('candidate_entity_id')
        df_exploded = df_exploded.drop(columns=['candidate_entity_ids'])
        
        if len(df_exploded) == 0:
            return pd.DataFrame()
            
        # 2. Join with actual text data
        df_s1_indexed = df_s1.set_index('entity_id')
        df_cand_indexed = df_candidates.set_index('entity_id')
        
        # Merge S1 data
        df_feat = df_exploded.join(df_s1_indexed, on='source1_entity_id', rsuffix='_s1')
        # Merge Cand data
        df_feat = df_feat.join(df_cand_indexed, on='candidate_entity_id', rsuffix='_cand')
        
        # Rename S1 columns since they don't have suffix
        df_feat.rename(columns={
            'business_name': 'business_name_s1',
            'business_address': 'business_address_s1',
            'country': 'country_s1'
        }, inplace=True)
        
        # 3. Compute Features (using list comprehension instead of iterrows for speed)
        # We extract columns as lists for faster iteration
        s1_ids = df_feat['source1_entity_id'].tolist()
        cand_ids = df_feat['candidate_entity_id'].tolist()
        names1 = df_feat['business_name_s1'].tolist()
        names2 = df_feat['business_name_cand'].tolist()
        addrs1 = df_feat['business_address_s1'].tolist()
        addrs2 = df_feat['business_address_cand'].tolist()
        countries1 = df_feat['country_s1'].tolist()
        countries2 = df_feat['country_cand'].tolist()
        
        features = []
        for i in range(len(s1_ids)):
            f = {}
            f['source1_entity_id'] = s1_ids[i]
            f['candidate_entity_id'] = cand_ids[i]
            
            name1 = self._safe_str(names1[i])
            name2 = self._safe_str(names2[i])
            
            addr1 = self._safe_str(addrs1[i])
            addr2 = self._safe_str(addrs2[i])
            
            # --- Name Metrics ---
            f['name_levenshtein'] = fuzz.ratio(name1, name2)
            f['name_token_sort'] = fuzz.token_sort_ratio(name1, name2)
            f['name_jaro_winkler'] = distance.JaroWinkler.normalized_similarity(name1, name2)
            f['name_3gram_jaccard'] = self._ngram_jaccard(name1, name2, 3)
            f['name_4gram_jaccard'] = self._ngram_jaccard(name1, name2, 4)
            
            f['name_prefix_match'] = 1 if (name1 and name2 and (name1.startswith(name2) or name2.startswith(name1))) else 0
            f['name_suffix_match'] = 1 if (name1 and name2 and (name1.endswith(name2) or name2.endswith(name1))) else 0
            
            # --- Address Metrics ---
            f['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2)
            f['addr_lcs'] = distance.LCSseq.normalized_similarity(addr1, addr2)
            
            digits1 = self._extract_digits(addr1)
            digits2 = self._extract_digits(addr2)
            if not digits1 and not digits2:
                f['addr_digit_match'] = 1.0  # Both have no digits
            elif not digits1 or not digits2:
                f['addr_digit_match'] = 0.0
            else:
                intersection = digits1.intersection(digits2)
                union = digits1.union(digits2)
                f['addr_digit_match'] = len(intersection) / len(union)
                
            # --- Combined & Categorical Metrics ---
            # Country Exact Match
            c1 = self._safe_str(countries1[i])
            c2 = self._safe_str(countries2[i])
            f['country_match'] = 1 if c1 == c2 else 0
            
            # Simple TF-IDF cosine approximation (Jaccard on words) for speed
            words1 = set((name1 + " " + addr1).split())
            words2 = set((name2 + " " + addr2).split())
            if not words1 and not words2:
                f['combined_word_jaccard'] = 1.0
            elif not words1 or not words2:
                f['combined_word_jaccard'] = 0.0
            else:
                inter = words1.intersection(words2)
                uni = words1.union(words2)
                f['combined_word_jaccard'] = len(inter) / len(uni)
                
            features.append(f)
            
        return pd.DataFrame(features)
