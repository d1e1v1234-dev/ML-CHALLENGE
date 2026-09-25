import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Blocker:
    """
    Two-stage blocking:
      1. Cheap key-based inverted index (prefixes of name/address tokens) narrows
         each S1 record down to a small bucket of plausible candidates.
      2. TF-IDF cosine similarity is computed ONLY inside that bucket to pick top_k.

    This avoids ever materializing an N_s1 x N_candidates dense similarity matrix,
    which is what made the previous implementation effectively brute-force and
    infeasible once a country's candidate pool reached millions of rows.
    """

    def __init__(self, top_k=15, max_bucket_size=2000):
        self.top_k = top_k
        self.max_bucket_size = max_bucket_size
        self.vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(2, 4), min_df=2, max_df=0.8, max_features=50000
        )

    def _make_keys(self, name: str, addr: str):
        """Generate a handful of cheap blocking keys for one record."""
        name_tokens = name.split()
        addr_tokens = addr.split()
        keys = set()
        if name_tokens:
            keys.add("N:" + name_tokens[0][:4])
            if len(name_tokens) > 1:
                keys.add("N2:" + name_tokens[0][:2] + name_tokens[1][:2])
        if addr_tokens:
            keys.add("A:" + addr_tokens[0][:4])
            if len(addr_tokens) > 1:
                keys.add("A2:" + addr_tokens[1][:4])
        return keys or {"__NOKEY__"}

    def _build_inverted_index(self, df: pd.DataFrame):
        index = defaultdict(list)
        names = df['business_name'].fillna('').values
        addrs = df['business_address'].fillna('').values
        for i in range(len(df)):
            for k in self._make_keys(names[i], addrs[i]):
                index[k].append(i)
        return index

    def fit_transform(self, df_s1: pd.DataFrame, df_candidates: pd.DataFrame):
        s1_texts = (df_s1['business_name'].fillna('') + " " + df_s1['business_address'].fillna(''))
        cand_texts = (df_candidates['business_name'].fillna('') + " " + df_candidates['business_address'].fillna(''))

        # Fit on both sides so candidate-only vocabulary isn't dropped.
        self.vectorizer.fit(pd.concat([s1_texts, cand_texts], ignore_index=True))
        X_s1_all = self.vectorizer.transform(s1_texts)
        X_cand_all = self.vectorizer.transform(cand_texts)

        cand_index = self._build_inverted_index(df_candidates)
        cand_ids = df_candidates['entity_id'].values
        s1_ids = df_s1['entity_id'].values
        s1_names = df_s1['business_name'].fillna('').values
        s1_addrs = df_s1['business_address'].fillna('').values

        candidate_pairs = []
        n_no_bucket = 0
        n_capped = 0

        for i in tqdm(range(len(df_s1)), desc="Blocking S1 entities"):
            keys = self._make_keys(s1_names[i], s1_addrs[i])
            cand_idx_set = set()
            for k in keys:
                cand_idx_set.update(cand_index.get(k, []))

            if not cand_idx_set:
                n_no_bucket += 1
                candidate_pairs.append({'source1_entity_id': s1_ids[i], 'candidate_entity_ids': ""})
                continue

            cand_idx = np.fromiter(cand_idx_set, dtype=int)

            # Guard against pathologically common keys (e.g. very generic prefixes)
            # blowing the bucket back up to near-brute-force size.
            if len(cand_idx) > self.max_bucket_size:
                n_capped += 1
                sims_bucket = X_s1_all[i].dot(X_cand_all[cand_idx].T).toarray().ravel()
                keep = np.argpartition(-sims_bucket, self.max_bucket_size - 1)[:self.max_bucket_size]
                cand_idx = cand_idx[keep]

            sims = X_s1_all[i].dot(X_cand_all[cand_idx].T).toarray().ravel()
            k = min(self.top_k, len(cand_idx))
            top_rel = np.argpartition(-sims, k - 1)[:k]
            top_rel = top_rel[np.argsort(-sims[top_rel])]

            matched_ids = cand_ids[cand_idx[top_rel]]
            candidate_pairs.append({
                'source1_entity_id': s1_ids[i],
                'candidate_entity_ids': ",".join(matched_ids)
            })

        if n_no_bucket:
            logger.info(f"{n_no_bucket}/{len(df_s1)} S1 records had no candidate bucket (will be singletons).")
        if n_capped:
            logger.info(f"{n_capped}/{len(df_s1)} S1 records hit an oversized bucket and were capped/pre-filtered.")

        return pd.DataFrame(candidate_pairs)

    def generate_candidate_pairs(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
        df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        all_candidate_pairs = []
        s1_countries = df_s1['country'].unique()

        for country in s1_countries:
            logger.info(f"Processing country: {country}")
            s1_subset = df_s1[df_s1['country'] == country].reset_index(drop=True)
            cand_subset = df_candidates[df_candidates['country'] == country].reset_index(drop=True)

            if len(cand_subset) == 0:
                for s1_id in s1_subset['entity_id']:
                    all_candidate_pairs.append({
                        'source1_entity_id': s1_id,
                        'candidate_entity_ids': ""
                    })
                continue

            logger.info(f"  {len(s1_subset)} S1 records vs {len(cand_subset)} candidates in '{country}'")
            blocker = Blocker(top_k=self.top_k, max_bucket_size=self.max_bucket_size)
            pairs_df = blocker.fit_transform(s1_subset, cand_subset)
            all_candidate_pairs.extend(pairs_df.to_dict('records'))

        return pd.DataFrame(all_candidate_pairs)