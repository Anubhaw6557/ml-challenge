"""Debug: embedding-only vs blocking-only recall on test split (NEW data)."""
import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from config import load_config
from src.matching.biencoder import BiEncoderMatcher

config = load_config()
processed_dir = Path(config['paths']['processed_dir']) / 'sample' / 'train'

s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
s23_df = pl.concat([s2_df, s3_df])

# GT for test split (last 200 S1 of sample order)
gt_path = Path(config['paths']['raw_dir']).parent / 'sample' / 'train' / 'train_ground_truth.tsv'
gt_df = pl.read_csv(gt_path, separator='\t')
sample_s1 = gt_df['source1_entity_id'].to_list()
test_s1 = sample_s1[-200:]
gt_map = {}
for row in gt_df.iter_rows(named=True):
    if row['source1_entity_id'] in set(test_s1):
        gt_map[row['source1_entity_id']] = set(row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else [])

print(f"Test S1: {len(test_s1)}")

# Embedding-only retrieval
s23_records = [r for r in s23_df.iter_rows(named=True)]
matcher = BiEncoderMatcher(config['matching']['biencoder_model'],
                           config['matching']['biencoder_batch_size'])
matcher.build_index(s23_records)

test_records = [r for r in s1_df.filter(pl.col('entity_id').is_in(test_s1)).iter_rows(named=True)]
emb = matcher.retrieve_candidates(test_records, k=50)
emb = {k: set(v) for k, v in emb.items()}

total = sum(len(gt_map.get(s, set())) for s in test_s1)
hit = sum(len(gt_map.get(s, set()) & emb.get(s, set())) for s in test_s1)
print(f"Embedding-only recall on test: {hit}/{total} = {hit/total:.4f}")

# Show a Devanagari miss example
for s1 in test_s1[:200]:
    g = gt_map.get(s1, set())
    e = emb.get(s1, set())
    missed = g - e
    if missed:
        print(f"MISS {s1}: GT={sorted(g)[:4]} emb_got={len(e)} missed={sorted(missed)[:3]}")
        break
