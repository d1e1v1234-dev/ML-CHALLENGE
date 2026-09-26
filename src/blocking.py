import gc
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def _log_step(msg: str):
    """Timestamped, immediately-flushed progress line -- visible live in Kaggle's
    cell output even during stages that have no tqdm bar of their own."""
    logger.info(msg)
    print(f"[blocking] {time.strftime('%H:%M:%S')} - {msg}", flush=True)


class Blocker:
    """
    Correct, memory-safe blocking:
      1. Cheap key-based inverted index on RAW TEXT buckets each S1 record with a
         small set of plausible candidates. Both per-key AND per-row (post-union)
         caps bound how many candidates any single S1 row can ever reference.
      2. The vectorizer is fit on a SAMPLE of texts (avoids the vocabulary-build
         memory spike on millions of documents).
      3. ONLY the candidates that appear in at least one bucket are transformed
         (via np.unique on the referenced indices) -- not the full candidate pool,
         and NOT via any duplicated fancy-row-indexing trick. This avoids the
         integer-overflow / duplication bug in the previous chunked-multiply design,
         where selecting a sparse matrix by a repeated-index array silently
         duplicated each row's nonzeros once per repetition.
      4. Each S1 row then does exactly ONE small, cheap dot product against just
         its own (<= max_row_bucket) candidate vectors -- no huge intermediate
         matrices at any point.
    """

    def __init__(self, top_k=15, max_key_bucket=40, max_row_bucket=60,
                 fit_sample_size=200_000, max_features=20_000):
        self.top_k = top_k
        self.max_key_bucket = max_key_bucket
        self.max_row_bucket = max_row_bucket
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
        for i in tqdm(range(len(names)), desc="Indexing candidates", mininterval=1.0):
            for k in self._make_keys(names[i], addrs[i]):
                index[k].append(i)

        n_capped = 0
        for k in list(index.keys()):
            idxs = index[k]
            if len(idxs) > self.max_key_bucket:
                n_capped += 1
                keep = rng.choice(len(idxs), size=self.max_key_bucket, replace=False)
                index[k] = [idxs[j] for j in keep]
        _log_step(f"Inverted index built: {len(index):,} keys, {n_capped:,} capped at {self.max_key_bucket}")
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

        # --- Fit on a SAMPLE only ---
        sample_n = min(self.fit_sample_size, len(s1_texts) + len(cand_texts))
        _log_step(f"Fitting vectorizer on {sample_n:,} sampled docs "
                   f"(S1={len(s1_texts):,}, candidates={len(cand_texts):,} total)...")
        t0 = time.time()
        sample_texts = pd.concat([s1_texts, cand_texts], ignore_index=True).sample(
            n=sample_n, random_state=42
        )
        self.vectorizer.fit(sample_texts)
        _log_step(f"Vectorizer fit done in {time.time() - t0:.1f}s "
                   f"(vocab size={len(self.vectorizer.vocabulary_):,})")

        t0 = time.time()
        X_s1 = self.vectorizer.transform(s1_texts)
        _log_step(f"Transformed {X_s1.shape[0]:,} S1 rows in {time.time() - t0:.1f}s (nnz={X_s1.nnz:,})")

        cand_index = self._build_capped_inverted_index(cand_names, cand_addrs, rng)

        # --- Resolve each S1 row's (per-key AND per-row capped) bucket ---
        s1_buckets = [None] * len(df_s1)
        n_no_bucket = 0
        n_row_capped = 0
        for i in tqdm(range(len(df_s1)), desc="Resolving S1 buckets", mininterval=1.0):
            keys = self._make_keys(s1_names[i], s1_addrs[i])
            bucket = set()
            for k in keys:
                bucket.update(cand_index.get(k, []))
            if not bucket:
                n_no_bucket += 1
                continue
            arr = np.fromiter(bucket, dtype=np.int32)
            if len(arr) > self.max_row_bucket:
                n_row_capped += 1
                keep = rng.choice(len(arr), size=self.max_row_bucket, replace=False)
                arr = arr[keep]
            s1_buckets[i] = arr

        if n_no_bucket:
            _log_step(f"{n_no_bucket:,}/{len(df_s1):,} S1 records had no candidate bucket (will be singletons).")
        if n_row_capped:
            _log_step(f"{n_row_capped:,}/{len(df_s1):,} S1 rows had their union bucket capped at {self.max_row_bucket}.")

        # --- Transform ONLY the candidates actually referenced by some bucket ---
        non_empty = [b for b in s1_buckets if b is not None]
        if not non_empty:
            return pd.DataFrame({
                'source1_entity_id': s1_ids,
                'candidate_entity_ids': [""] * len(s1_ids)
            })

        referenced = np.unique(np.concatenate(non_empty))
        _log_step(f"Transforming {len(referenced):,} uniquely-referenced candidates "
                  f"(out of {len(df_candidates):,} total in this country)...")
        t0 = time.time()
        X_cand_ref = self.vectorizer.transform(cand_texts.iloc[referenced])
        _log_step(f"Candidate transform done in {time.time() - t0:.1f}s (nnz={X_cand_ref.nnz:,})")

        # Map global candidate index -> local row position in X_cand_ref
        local_pos = np.full(len(df_candidates), -1, dtype=np.int64)
        local_pos[referenced] = np.arange(len(referenced))
        del referenced, non_empty
        gc.collect()

        # --- One small dot product per S1 row: no duplicated matrices, no chunking needed ---
        candidate_pairs = []
        t0 = time.time()
        for i in tqdm(range(len(df_s1)), desc="Scoring S1 buckets", mininterval=1.0):
            bucket = s1_buckets[i]
            if bucket is None:
                candidate_pairs.append({'source1_entity_id': s1_ids[i], 'candidate_entity_ids': ""})
                continue

            local_idx = local_pos[bucket]
            sims = X_s1[i].dot(X_cand_ref[local_idx].T).toarray().ravel()

            k = min(self.top_k, len(sims))
            top_rel = np.argpartition(-sims, k - 1)[:k]
            top_rel = top_rel[np.argsort(-sims[top_rel])]
            matched_ids = cand_ids[bucket[top_rel]]
            candidate_pairs.append({
                'source1_entity_id': s1_ids[i],
                'candidate_entity_ids': ",".join(matched_ids)
            })
        _log_step(f"Scored all S1 rows in {time.time() - t0:.1f}s")

        return pd.DataFrame(candidate_pairs)

    def generate_candidate_pairs(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
        df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        all_candidate_pairs = []

        countries = list(df_s1['country'].unique())
        for country_num, country in enumerate(countries, 1):
            _log_step(f"[Country {country_num}/{len(countries)}] Processing '{country}'")
            s1_subset = df_s1[df_s1['country'] == country].reset_index(drop=True)
            cand_subset = df_candidates[df_candidates['country'] == country].reset_index(drop=True)
            _log_step(f"  {len(s1_subset):,} S1 records vs {len(cand_subset):,} candidates in '{country}'")

            if len(cand_subset) == 0:
                for s1_id in s1_subset['entity_id']:
                    all_candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ""})
                continue

            t0 = time.time()
            blocker = Blocker(top_k=self.top_k)
            pairs_df = blocker.fit_transform(s1_subset, cand_subset)
            all_candidate_pairs.extend(pairs_df.to_dict('records'))
            del blocker
            gc.collect()
            _log_step(f"[Country {country_num}/{len(countries)}] '{country}' done in {time.time() - t0:.1f}s")

        return pd.DataFrame(all_candidate_pairs)