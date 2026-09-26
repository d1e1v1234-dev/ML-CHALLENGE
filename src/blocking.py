import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Blocker:
    """
    Memory-safe blocking:
      1. Cheap key-based inverted index on RAW TEXT (no TF-IDF needed) buckets each
         S1 record with a small set of plausible candidates. Oversized key buckets
         are capped by random subsampling on raw text, before any vectorization.
      2. The vectorizer is fit on a SAMPLE of texts (not the full corpus) to avoid
         the vocabulary-building memory spike on tens of millions of documents.
      3. Candidates are TF-IDF transformed in CHUNKS (one chunk resident in memory
         at a time), and each chunk is only scored against the S1 rows whose bucket
         actually references a candidate in that chunk. The full N_s1 x N_cand
         matrix is never materialized.
    """

    def __init__(self, top_k=15, max_key_bucket=300, cand_chunk_size=200_000,
                 fit_sample_size=200_000, max_features=20_000):
        self.top_k = top_k
        self.max_key_bucket = max_key_bucket
        self.cand_chunk_size = cand_chunk_size
        self.fit_sample_size = fit_sample_size
        self.vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(2, 4), min_df=2, max_df=0.8,
            max_features=max_features, dtype=np.float32
        )

    def _make_keys(self, name: str, addr: str):
        name_tokens = name.split()
        addr_tokens = addr.split()
        keys = set()
        if name_tokens:
            keys.add("N:" + name_tokens[0][:4])
            if len(name_tokens) > 1:
                keys.add("N2:" + name_tokens[0][:2] + name_tokens[1][:2])
        if addr_tokens:
            keys.add("A:" + addr_tokens[0][:4])
        return keys or {"__NOKEY__"}

    def _build_capped_inverted_index(self, names, addrs, rng):
        index = defaultdict(list)
        for i in range(len(names)):
            for k in self._make_keys(names[i], addrs[i]):
                index[k].append(i)
        # Cap oversized buckets by random subsampling on RAW TEXT indices,
        # before any TF-IDF work is done -- this is what keeps memory bounded.
        for k in list(index.keys()):
            idxs = index[k]
            if len(idxs) > self.max_key_bucket:
                keep = rng.choice(len(idxs), size=self.max_key_bucket, replace=False)
                index[k] = [idxs[j] for j in keep]
        return index

    def fit_transform(self, df_s1: pd.DataFrame, df_candidates: pd.DataFrame):
        rng = np.random.default_rng(42)

        s1_names = df_s1['business_name'].fillna('').values
        s1_addrs = df_s1['business_address'].fillna('').values
        cand_names = df_candidates['business_name'].fillna('').values
        cand_addrs = df_candidates['business_address'].fillna('').values
        cand_ids = df_candidates['entity_id'].values
        s1_ids = df_s1['entity_id'].values

        s1_texts = pd.Series(s1_names) + " " + pd.Series(s1_addrs)
        cand_texts = pd.Series(cand_names) + " " + pd.Series(cand_addrs)

        # --- Fit on a SAMPLE only: avoids the .fit() vocabulary-build memory spike ---
        n_total = len(s1_texts) + len(cand_texts)
        sample_n = min(self.fit_sample_size, n_total)
        sample_texts = pd.concat([s1_texts, cand_texts], ignore_index=True).sample(
            n=sample_n, random_state=42
        )
        self.vectorizer.fit(sample_texts)

        # --- Transform S1 fully (the much smaller side) ---
        X_s1 = self.vectorizer.transform(s1_texts)

        # --- Cheap inverted index on raw text, capped BEFORE any vectorization ---
        cand_index = self._build_capped_inverted_index(cand_names, cand_addrs, rng)

        n_cand = len(df_candidates)
        n_chunks = int(np.ceil(n_cand / self.cand_chunk_size))

        # Resolve each S1 row's (capped) candidate bucket, then pre-split by chunk
        chunk_work = defaultdict(list)  # chunk_id -> [(s1_row_idx, local_indices), ...]
        n_no_bucket = 0
        for i in range(len(df_s1)):
            keys = self._make_keys(s1_names[i], s1_addrs[i])
            bucket = set()
            for k in keys:
                bucket.update(cand_index.get(k, []))
            if not bucket:
                n_no_bucket += 1
                continue
            bucket_arr = np.fromiter(bucket, dtype=np.int64)
            chunk_ids = bucket_arr // self.cand_chunk_size
            for c in np.unique(chunk_ids):
                mask = chunk_ids == c
                local_idx = (bucket_arr[mask] - c * self.cand_chunk_size).astype(np.int64)
                chunk_work[int(c)].append((i, local_idx))

        if n_no_bucket:
            logger.info(f"{n_no_bucket}/{len(df_s1)} S1 records had no candidate bucket (will be singletons).")

        # --- Process candidates chunk by chunk; only one chunk's matrix in memory at a time ---
        s1_scores = [[] for _ in range(len(df_s1))]
        s1_cand_idx = [[] for _ in range(len(df_s1))]

        for c in tqdm(range(n_chunks), desc="Candidate chunks"):
            if c not in chunk_work:
                continue
            start = c * self.cand_chunk_size
            end = min(start + self.cand_chunk_size, n_cand)
            X_chunk = self.vectorizer.transform(cand_texts.iloc[start:end])

            for s1_row_idx, local_idx in chunk_work[c]:
                sims = X_s1[s1_row_idx].dot(X_chunk[local_idx].T).toarray().ravel()
                s1_scores[s1_row_idx].append(sims)
                s1_cand_idx[s1_row_idx].append(local_idx + start)

            del X_chunk  # free before the next chunk loads

        candidate_pairs = []
        for i in range(len(df_s1)):
            if not s1_scores[i]:
                candidate_pairs.append({'source1_entity_id': s1_ids[i], 'candidate_entity_ids': ""})
                continue
            scores = np.concatenate(s1_scores[i])
            idxs = np.concatenate(s1_cand_idx[i])
            k = min(self.top_k, len(scores))
            top_rel = np.argpartition(-scores, k - 1)[:k]
            top_rel = top_rel[np.argsort(-scores[top_rel])]
            matched_ids = cand_ids[idxs[top_rel]]
            candidate_pairs.append({
                'source1_entity_id': s1_ids[i],
                'candidate_entity_ids': ",".join(matched_ids)
            })

        return pd.DataFrame(candidate_pairs)

    def generate_candidate_pairs(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
        df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        all_candidate_pairs = []

        for country in df_s1['country'].unique():
            logger.info(f"Processing country: {country}")
            s1_subset = df_s1[df_s1['country'] == country].reset_index(drop=True)
            cand_subset = df_candidates[df_candidates['country'] == country].reset_index(drop=True)
            logger.info(f"  {len(s1_subset)} S1 records vs {len(cand_subset)} candidates in '{country}'")

            if len(cand_subset) == 0:
                for s1_id in s1_subset['entity_id']:
                    all_candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ""})
                continue

            blocker = Blocker(top_k=self.top_k)
            pairs_df = blocker.fit_transform(s1_subset, cand_subset)
            all_candidate_pairs.extend(pairs_df.to_dict('records'))
            del blocker

        return pd.DataFrame(all_candidate_pairs)