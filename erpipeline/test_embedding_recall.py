"""
Embedding-based retrieval recall test on the processed sample.

Builds country-partitioned FAISS indexes on sample S2/S3, retrieves top-k
per S1, unions with blocking candidates, and measures recall lift.
"""
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from src.matching.biencoder import BiEncoderMatcher

TOP_K = 50


def _as_id_list(value):
    if isinstance(value, list):
        return [v for v in value if v]
    if isinstance(value, str):
        return [v for v in value.split(',') if v]
    return []


def main():
    config = load_config()
    processed_dir = Path(config['paths']['processed_dir']) / 'sample' / 'train'
    candidates_dir = Path(config['paths']['candidates_dir']).parent / 'sample_candidates'

    print("=== Loading sample data ===")
    s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
    s23_df = pl.concat([s2_df, s3_df])
    print(f"S1: {len(s1_df)}, S2+S3: {len(s23_df)}")

    # Ground truth (sample only).
    gt_path = Path(config['paths']['raw_dir']).parent / 'sample' / 'train' / 'train_ground_truth.tsv'
    gt_sets = defaultdict(set)
    with open(gt_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                gt_sets[parts[0]].update(parts[1].split(','))
    print(f"GT S1 entities: {len(gt_sets)}")

    # Existing blocking candidates.
    blocking_df = pl.read_parquet(candidates_dir / 'candidates_train.parquet')
    blocking_cands = {}
    for row in blocking_df.iter_rows(named=True):
        blocking_cands[row['source1_entity_id']] = set(_as_id_list(row['candidate_entity_ids']))

    # Build embedding index on S2/S3.
    print("\n=== Building embedding index ===")
    s23_records = [row for row in s23_df.iter_rows(named=True)]
    matcher = BiEncoderMatcher(config['matching']['biencoder_model'],
                               config['matching']['biencoder_batch_size'])
    matcher.build_index(s23_records)

    # Retrieve top-k per S1.
    print(f"\n=== Retrieving top-{TOP_K} per S1 ===")
    s1_records = [row for row in s1_df.iter_rows(named=True)]
    emb_cands = matcher.retrieve_candidates(s1_records, k=TOP_K)
    emb_cands = {k: set(v) for k, v in emb_cands.items()}

    # Evaluate blocking-only vs blocking+embedding.
    def _recall(cand_map):
        total, hit, per_entity = 0, 0, []
        for s1_id, gt in gt_sets.items():
            if not gt:
                continue
            cands = cand_map.get(s1_id, set())
            recalled = len(gt & cands)
            total += len(gt)
            hit += recalled
            per_entity.append(recalled / len(gt))
        import numpy as np
        return (hit / total if total else 0.0,
                float(np.mean(per_entity)) if per_entity else 0.0,
                total, hit)

    b_overall, b_macro, b_total, b_hit = _recall(blocking_cands)
    print(f"\nBlocking only: overall={b_overall:.4f} macro={b_macro:.4f} "
          f"({b_hit}/{b_total})")

    combined = {s1: blocking_cands.get(s1, set()) | emb_cands.get(s1, set())
                for s1 in set(list(blocking_cands) + list(emb_cands))}
    c_overall, c_macro, c_total, c_hit = _recall(combined)
    print(f"Blocking+embedding: overall={c_overall:.4f} macro={c_macro:.4f} "
          f"({c_hit}/{c_total})")
    print(f"Recall lift: +{c_overall - b_overall:.4f} overall, +{c_macro - b_macro:.4f} macro")

    # Show newly recalled pairs (what embeddings fixed).
    print("\nNewly recalled by embeddings:")
    shown = 0
    for s1_id, gt in sorted(gt_sets.items()):
        newly = (gt & emb_cands.get(s1_id, set())) - blocking_cands.get(s1_id, set())
        if newly and shown < 10:
            print(f"  {s1_id}: {sorted(newly)}")
            shown += 1
    if shown == 0:
        print("  (none)")

    # Still missed after both.
    still_missed = sum(len(gt - combined.get(s1, set())) for s1, gt in gt_sets.items())
    print(f"\nStill missed after both: {still_missed}/{c_total}")


if __name__ == '__main__':
    main()
