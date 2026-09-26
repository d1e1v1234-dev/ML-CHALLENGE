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
    Fully vectorized, memory-safe blocking:
      1. Cheap key-based inverted index on RAW TEXT buckets each S1 record with a
         small set of plausible candidates. Oversized key buckets are capped by
         random subsampling BEFORE any vectorization.
      2. The vectorizer is fit on a SAMPLE of texts to avoid the vocabulary-build
         memory spike on tens of millions of documents.
      3. All (s1_row, candidate) bucket pairs are flattened into ONE pair of numpy
         arrays and sorted by candidate index -- no per-chunk Python dict/list of
         tuples is ever built (that was the bug in the previous version: it
         materialized a near-full row x chunk structure since buckets scatter
         across all chunks, not just a few).
      4. Candidates are transformed in chunks; each chunk's similarity scores are
         computed in ONE vectorized sparse operation (row-aligned multiply + sum),
         not a per-row loop.
      5. Top-k selection per S1 row is done with a single vectorized
         sort + groupby().cumcount(), not a Python loop over rows.
    """

    def __init__(self, top_k=15, max_key_bucket=100, cand_chunk_size=300_000,
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
        _log_step(f"Inverted index built: {len(index)} keys, {n_capped} capped at {self.max_key_bucket}")
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
        _log_step(f"Transformed {X_s1.shape[0]:,} S1 rows in {time.time() - t0:.1f}s "
                   f"(nnz={X_s1.nnz:,})")

        cand_index = self._build_capped_inverted_index(cand_names, cand_addrs, rng)

        # --- Build ONE flat (s1_row_idx, global_cand_idx) pair array ---
        row_parts, cand_parts = [], []
        n_no_bucket = 0
        for i in tqdm(range(len(df_s1)), desc="Resolving S1 buckets", mininterval=1.0):
            keys = self._make_keys(s1_names[i], s1_addrs[i])
            bucket = set()
            for k in keys:
                bucket.update(cand_index.get(k, []))
            if bucket:
                arr = np.fromiter(bucket, dtype=np.int64)
                cand_parts.append(arr)
                row_parts.append(np.full(len(arr), i, dtype=np.int64))
            else:
                n_no_bucket += 1

        if n_no_bucket:
            _log_step(f"{n_no_bucket:,}/{len(df_s1):,} S1 records had no candidate bucket (will be singletons).")

        if not cand_parts:
            return pd.DataFrame({
                'source1_entity_id': s1_ids,
                'candidate_entity_ids': [""] * len(s1_ids)
            })

        flat_rows = np.concatenate(row_parts)
        flat_cands = np.concatenate(cand_parts)
        del row_parts, cand_parts
        _log_step(f"Total (S1, candidate) pairs to score: {len(flat_rows):,} "
                   f"(~{len(flat_rows) / max(len(df_s1) - n_no_bucket, 1):.0f} avg per matched S1 row)")

        # Sort by candidate global index -> pairs touching the same chunk are contiguous
        order = np.argsort(flat_cands, kind='stable')
        flat_rows = flat_rows[order]
        flat_cands = flat_cands[order]
        del order

        n_cand = len(df_candidates)
        n_chunks = int(np.ceil(n_cand / self.cand_chunk_size))
        boundaries = np.searchsorted(
            flat_cands, np.arange(0, n_cand + self.cand_chunk_size, self.cand_chunk_size)
        )

        all_scores = np.empty(len(flat_cands), dtype=np.float32)
        total_pairs = len(flat_cands)
        pairs_done = 0

        chunk_bar = tqdm(range(n_chunks), desc="Candidate chunks")
        for c in chunk_bar:
            lo, hi = boundaries[c], boundaries[c + 1]
            if hi <= lo:
                continue
            start = c * self.cand_chunk_size
            end = min(start + self.cand_chunk_size, n_cand)
            X_chunk = self.vectorizer.transform(cand_texts.iloc[start:end])

            rows_slice = flat_rows[lo:hi]
            local_idx = flat_cands[lo:hi] - start

            X_s1_sel = X_s1[rows_slice]
            X_chunk_sel = X_chunk[local_idx]
            sims = np.asarray(X_s1_sel.multiply(X_chunk_sel).sum(axis=1)).ravel()
            all_scores[lo:hi] = sims

            pairs_done += (hi - lo)
            chunk_bar.set_postfix({
                'pairs': f"{pairs_done:,}/{total_pairs:,}",
                'pairs_in_chunk': hi - lo
            })

            del X_chunk, X_s1_sel, X_chunk_sel
        _log_step(f"Finished scoring all {total_pairs:,} candidate pairs across {n_chunks} chunks")

        # --- Vectorized top-k per S1 row: no Python loop over rows ---
        _log_step("Selecting top-k candidates per S1 row (sort + groupby)...")
        t0 = time.time()
        df_pairs = pd.DataFrame({'s1_row': flat_rows, 'cand_idx': flat_cands, 'score': all_scores})
        del flat_rows, flat_cands, all_scores

        df_pairs.sort_values(['s1_row', 'score'], ascending=[True, False], inplace=True)
        df_pairs['rank'] = df_pairs.groupby('s1_row', sort=False).cumcount()
        df_top = df_pairs[df_pairs['rank'] < self.top_k].copy()
        del df_pairs

        df_top['cand_entity_id'] = cand_ids[df_top['cand_idx'].values]
        grouped = df_top.groupby('s1_row', sort=False)['cand_entity_id'].apply(lambda x: ",".join(x))
        result_map = grouped.to_dict()
        _log_step(f"Top-k selection done in {time.time() - t0:.1f}s")

        candidate_pairs = [
            {'source1_entity_id': s1_ids[i], 'candidate_entity_ids': result_map.get(i, "")}
            for i in range(len(df_s1))
        ]
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
            _log_step(f"[Country {country_num}/{len(countries)}] '{country}' done in {time.time() - t0:.1f}s "
                       f"({country_num}/{len(countries)} countries complete)")

        return pd.DataFrame(all_candidate_pairs)