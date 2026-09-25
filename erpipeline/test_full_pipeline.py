"""
End-to-end pipeline test on a small TRAIN sample with a held-out test split.

Flow (sample-only, no full-dataset leakage):
  1. Create a matched sample of ~50 S1 entities from TRAIN with ground truth.
  2. Preprocess the SAMPLE, generate candidates, evaluate blocking recall.
  3. Extract pairwise features for all candidate pairs.
  4. Split S1 entities: 40 for training, 10 held out for testing.
  5. Fit TF-IDF on TRAIN names only, train LightGBM, calibrate thresholds.
  6. Run inference on the held-out 10 S1 entities, print detailed per-entity
     results, and write sample-only output TSVs.
"""
import copy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.blocking.candidate_gen import generate_candidates, evaluate_recall
from src.matching.features import FeatureExtractor
from src.matching.lgbm_model import LightGBMMatcher
from src.matching.biencoder import BiEncoderMatcher
from src.matching.threshold import ThresholdCalibrator, macro_f05
from src.postprocess.output import generate_outputs


SAMPLE_SIZE = 2000
TEST_SIZE = 200
NON_FEATURE_COLS = {'source1_entity_id', 'candidate_entity_id', 'label',
                    'country', 'is_test'}


def _as_id_list(value):
    """Normalize parquet candidate values (list or comma-separated string)."""
    if isinstance(value, list):
        return [v for v in value if v]
    if isinstance(value, str):
        return [v for v in value.split(',') if v]
    return []


def create_matched_sample(config: dict, sample_size: int = SAMPLE_SIZE):
    """Create a sample of S1 entities (with GT matches) plus their S2/S3 matches."""
    raw_dir = Path(config['paths']['raw_dir'])
    sample_raw_dir = Path(config['paths']['raw_dir']).parent / 'sample' / 'train'
    sample_raw_dir.mkdir(parents=True, exist_ok=True)

    gt_path = raw_dir / 'train' / 'train_ground_truth.tsv'
    print(f"Loading ground truth from {gt_path}...")
    gt_df = pl.read_csv(gt_path, separator='\t',
                        columns=['source1_entity_id', 'matched_entity_ids'])

    gt_with_matches = gt_df.filter(pl.col('matched_entity_ids') != '')
    print(f"S1 entities with matches: {len(gt_with_matches)}")

    sample_s1_ids = gt_with_matches['source1_entity_id'].head(sample_size).to_list()
    print(f"Sampling {len(sample_s1_ids)} S1 entities with matches")

    matched_ids = set()
    for matches in gt_df.filter(
            pl.col('source1_entity_id').is_in(sample_s1_ids))['matched_entity_ids']:
        matched_ids.update(matches.split(','))
    print(f"Total matched S2/S3 IDs: {len(matched_ids)}")

    for src, prefix in [('source1', 'S1-'), ('source2', 'S2-'), ('source3', 'S3-')]:
        needed = [eid for eid in (sample_s1_ids if src == 'source1' else matched_ids)
                  if eid.startswith(prefix)]
        if not needed:
            print(f"No rows needed for {src}")
            continue
        df = pl.read_csv(raw_dir / 'train' / f'train_{src}.tsv', separator='\t')
        filtered = df.filter(pl.col('entity_id').is_in(needed))
        if len(filtered) == 0:
            print(f"No rows found for {src}")
            continue
        out_path = sample_raw_dir / f'train_{src}.tsv'
        filtered.write_csv(out_path, separator='\t')
        print(f"Created {src}: {len(filtered)} rows")

    gt_out = sample_raw_dir.parent / 'train' / 'train_ground_truth.tsv'
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    gt_df.filter(pl.col('source1_entity_id').is_in(sample_s1_ids)).write_csv(
        gt_out, separator='\t')
    print(f"Created sample ground truth: {gt_out}")

    return sample_s1_ids


def _format_entity(entity: dict, label: str) -> str:
    """Format one entity compactly for debugging misses/false positives."""
    lines = [f"  {label}:"]
    lines.append(f"    ID: {entity.get('entity_id', 'N/A')}")
    lines.append(f"    Country: {entity.get('country', 'N/A')}")
    lines.append(f"    Raw Name: {entity.get('name_raw', 'N/A')}")
    lines.append(f"    Clean Name: {entity.get('name_clean', 'N/A')}")
    lines.append(f"    Raw Addr: {entity.get('addr_raw', 'N/A')}")
    lines.append(f"    Clean Addr: {entity.get('addr_clean', 'N/A')}")
    lines.append(f"    PINs: {entity.get('postal_codes', [])}")
    lines.append(f"    City: {entity.get('city', '')} | State: {entity.get('state', '')} | "
                 f"House#: {entity.get('house_number', '')}")
    lines.append(f"    Metaphone: {entity.get('metaphone', '')} | "
                 f"Soundex: {entity.get('soundex', '')} | NYSIIS: {entity.get('nysiis', '')}")
    return "\n".join(lines)


def run_full_pipeline(config: dict, sample_size: int = SAMPLE_SIZE,
                      test_size: int = TEST_SIZE):
    print("=== Creating matched sample ===")
    sample_s1_ids = create_matched_sample(config, sample_size)

    # Redirect every downstream stage at the SAMPLE directories.
    test_config = copy.deepcopy(config)
    test_config['paths']['raw_dir'] = str(Path(config['paths']['raw_dir']).parent / 'sample')
    test_config['paths']['processed_dir'] = str(Path(config['paths']['processed_dir']) / 'sample')
    test_config['paths']['candidates_dir'] = str(
        Path(config['paths']['candidates_dir']).parent / 'sample_candidates')
    test_config['paths']['models_dir'] = str(Path(config['paths']['models_dir']) / 'sample')
    test_config['paths']['output_dir'] = str(Path(config['paths']['output_dir']) / 'sample')
    test_config['blocking']['max_candidates_per_s1'] = 50

    # 1. Preprocessing (sample only).
    print("\n=== Stage 1: Preprocessing (sample) ===")
    PreprocessingPipeline(test_config).run()

    # 2. Blocking + candidates + recall (sample only).
    print("\n=== Stage 2: Blocking + recall (sample) ===")
    evaluate_recall(test_config, split='train')
    generate_candidates(test_config, split='train')

    # 3. Feature extraction (sample only).
    print("\n=== Stage 3a: Feature extraction (sample) ===")
    processed_dir = Path(test_config['paths']['processed_dir']) / 'train'
    candidates_dir = Path(test_config['paths']['candidates_dir'])

    s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
    s23_df = pl.concat([s2_df, s3_df])
    s23_lookup = {row['entity_id']: row for row in s23_df.iter_rows(named=True)}
    s1_lookup = {row['entity_id']: row for row in s1_df.iter_rows(named=True)}
    candidates_df = pl.read_parquet(candidates_dir / 'candidates_train.parquet')

    # Stage 2b: embedding retrieval (country-partitioned FAISS), unioned
    # with blocking candidates for higher recall.
    print("\n=== Stage 2b: Embedding retrieval (sample) ===")
    s23_records = [row for row in s23_df.iter_rows(named=True)]
    s1_records = [row for row in s1_df.iter_rows(named=True)]
    emb_matcher = BiEncoderMatcher(test_config['matching']['biencoder_model'],
                                   test_config['matching']['biencoder_batch_size'])
    emb_matcher.build_index(s23_records)
    emb_hits = emb_matcher.retrieve_candidates(
        s1_records, k=test_config['blocking'].get('biencoder_top_k', 50))
    merged_rows = []
    for row in candidates_df.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        merged = set(_as_id_list(row['candidate_entity_ids']))
        merged.update(emb_hits.get(s1_id, []))
        max_cand = test_config['blocking']['max_candidates_per_s1']
        if len(merged) > max_cand:
            import random
            merged = set(random.sample(sorted(merged), max_cand))
        merged_rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': sorted(merged),
            'num_candidates': len(merged),
        })
    candidates_df = pl.DataFrame(merged_rows)
    candidates_df.write_parquet(candidates_dir / 'candidates_train.parquet')
    print(f"Merged blocking + embedding candidates for {len(candidates_df)} S1 entities")

    gt_path = Path(test_config['paths']['raw_dir']) / 'train' / 'train_ground_truth.tsv'
    gt_labels = {}
    with open(gt_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                for match_id in parts[1].split(','):
                    gt_labels[(parts[0], match_id)] = 1

    # Split SAMPLE S1 entities into train / held-out test by S1 id.
    s1_with_candidates = [row['source1_entity_id'] for row in
                          candidates_df.iter_rows(named=True)
                          if _as_id_list(row['candidate_entity_ids'])]
    ordered_test = [eid for eid in sample_s1_ids if eid in set(s1_with_candidates)]
    test_s1_ids = ordered_test[-test_size:] if len(ordered_test) >= test_size else ordered_test
    train_s1_ids = [eid for eid in sample_s1_ids if eid not in set(test_s1_ids)]
    print(f"Train S1: {len(train_s1_ids)}, held-out test S1: {len(test_s1_ids)}")

    # Fit TF-IDF on TRAIN records only (no test leakage).
    train_cand_ids = set()
    for row in candidates_df.filter(
            pl.col('source1_entity_id').is_in(train_s1_ids)).iter_rows(named=True):
        train_cand_ids.update(_as_id_list(row['candidate_entity_ids']))
    train_names = [s1_lookup[eid]['name_clean'] for eid in train_s1_ids if eid in s1_lookup]
    train_names += [s23_lookup[eid]['name_clean'] for eid in train_cand_ids if eid in s23_lookup]

    extractor = FeatureExtractor()
    extractor.fit([name for name in train_names if name])

    print("Extracting pairwise features...")
    train_rows, test_rows = [], []

    def _features_for(s1_id: str):
        s1_data = s1_lookup.get(s1_id)
        if not s1_data:
            return []
        cand_ids = _as_id_list(candidates_df.filter(
            pl.col('source1_entity_id') == s1_id
        )['candidate_entity_ids'].to_list()[0]
            if s1_id in candidates_df['source1_entity_id'].to_list() else [])
        rows = []
        for cand_id in cand_ids:
            cand_row = s23_lookup.get(cand_id)
            if not cand_row:
                continue
            features = extractor.extract(s1_data, cand_row)
            features['source1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            features['label'] = 1 if (s1_id, cand_id) in gt_labels else 0
            features['country'] = s1_data.get('country', 'US')
            rows.append(features)
        return rows

    for s1_id in train_s1_ids:
        train_rows.extend(_features_for(s1_id))
    for s1_id in test_s1_ids:
        test_rows.extend(_features_for(s1_id))

    train_features = pl.DataFrame(train_rows) if train_rows else pl.DataFrame()
    test_features = pl.DataFrame(test_rows) if test_rows else pl.DataFrame()
    print(f"Train pairs: {len(train_features)}, test pairs: {len(test_features)}")

    if len(train_features) == 0 or len(test_features) == 0:
        raise RuntimeError("Sample produced no train or no test pairs; increase SAMPLE_SIZE.")

    features_path = Path(test_config['paths']['candidates_dir']) / 'features_train.parquet'
    train_features.write_parquet(features_path)

    # Train/validation split on TRAIN pairs only.
    print("\n=== Stage 3b: Training LightGBM (40 S1 entities) ===")
    train_pd = train_features.to_pandas()
    try:
        from sklearn.model_selection import train_test_split
        model_idx, calib_idx = train_test_split(
            range(len(train_pd)), test_size=0.2, random_state=42,
            stratify=train_pd['label'])
    except ValueError:
        from sklearn.model_selection import train_test_split
        model_idx, calib_idx = train_test_split(
            range(len(train_pd)), test_size=0.2, random_state=42)
    model_pairs = train_features[list(model_idx)]
    calib_pairs = train_features[list(calib_idx)]

    feature_cols = [c for c in model_pairs.columns if c not in NON_FEATURE_COLS]
    lgbm = LightGBMMatcher(test_config['matching']['lgbm_params'])
    lgbm.train(model_pairs, calib_pairs, feature_cols)

    print("\n=== Stage 3c: Threshold calibration ===")
    calib_probs = lgbm.predict_proba(calib_pairs, feature_cols)
    calib_labels = calib_pairs['label'].to_numpy()
    calib_countries = (calib_pairs['country'].to_numpy()
                       if 'country' in calib_pairs.columns
                       else np.array(['US'] * len(calib_labels)))
    calibrator = ThresholdCalibrator(beta=0.5)
    calibrator.fit(calib_probs, calib_labels, calib_countries)

    models_dir = Path(test_config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    lgbm.save(str(models_dir / 'lgbm_model.pkl'))
    calibrator.save(str(models_dir / 'thresholds.pkl'))

    # Inference on the held-out 10 S1 entities.
    print("\n=== Stage 4: Test inference (10 held-out S1 entities) ===")
    test_probs = lgbm.predict_proba(test_features, feature_cols)
    test_countries = (test_features['country'].to_numpy()
                      if 'country' in test_features.columns
                      else np.array(['US'] * len(test_features)))
    calibrator = ThresholdCalibrator.load(str(models_dir / 'thresholds.pkl'))
    test_preds = calibrator.predict(test_probs, test_countries)

    prob_by_pair = {}
    for i, row in enumerate(test_features.iter_rows(named=True)):
        prob_by_pair[(row['source1_entity_id'], row['candidate_entity_id'])] = float(test_probs[i])

    predictions = defaultdict(list)
    for i, row in enumerate(test_features.iter_rows(named=True)):
        if int(test_preds[i]) == 1:
            predictions[row['source1_entity_id']].append(row['candidate_entity_id'])

    candidates_dict = {}
    for s1_id in test_s1_ids:
        match = candidates_df.filter(pl.col('source1_entity_id') == s1_id)
        candidates_dict[s1_id] = (_as_id_list(match['candidate_entity_ids'].to_list()[0])
                                  if len(match) else [])

    # Detailed per-entity test report against sample GT.
    print("\n=== Detailed held-out test results ===")
    per_entity_f05, all_labels, all_preds = [], [], []
    for idx, s1_id in enumerate(test_s1_ids, start=1):
        gt_set = {m for (s, m) in gt_labels if s == s1_id}
        pred_set = set(predictions.get(s1_id, []))
        cand_set = set(candidates_dict.get(s1_id, []))
        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 1.0
        recall = tp / len(gt_set) if gt_set else 1.0
        entity_f05 = ((1.25 * precision * recall) / (0.25 * precision + recall)
                      if (precision + recall) > 0 else 0.0)
        per_entity_f05.append(entity_f05)

        print(f"\n{'=' * 80}")
        print(f"[{idx:2d}] {s1_id}: GT={len(gt_set)} Predicted={len(pred_set)} "
              f"P={precision:.2f} R={recall:.2f} F0.5={entity_f05:.2f}")
        print(_format_entity(s1_lookup.get(s1_id, {}), "S1 Entity"))

        for mid in sorted(pred_set):
            status = "TP" if mid in gt_set else "FP"
            info = (f"{mid} prob={prob_by_pair.get((s1_id, mid), 0.0):.3f} | "
                    f"{s23_lookup.get(mid, {}).get('name_clean', 'N/A')} | "
                    f"{s23_lookup.get(mid, {}).get('addr_clean', 'N/A')}")
            print(f"  PREDICTED [{status}]: {info}")
        for mid in sorted(gt_set - pred_set):
            print(_format_entity(s23_lookup.get(mid, {}), f"  MISSED GT: {mid}"))
        for mid in sorted(pred_set - gt_set):
            print(_format_entity(s23_lookup.get(mid, {}), f"  FALSE POSITIVE: {mid}"))
        if not cand_set:
            print("  Note: blocking produced no candidates for this entity.")

        for mid in sorted(gt_set):
            all_labels.append(1 if mid in pred_set else 0)
            all_preds.append(1 if mid in pred_set else 0)

    from sklearn.metrics import precision_recall_fscore_support
    test_labels = test_features['label'].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='macro', zero_division=0)
    pair_f05 = ((1.25 * precision * recall) / (0.25 * precision + recall)
                if (precision + recall) > 0 else 0.0)
    print(f"\n{'=' * 80}")
    print("Held-out test summary (pair-level):")
    print(f"  Test pairs: {len(test_features)}, predicted positives: {int(test_preds.sum())}")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | "
          f"F1: {f1:.4f} | Macro F0.5: {pair_f05:.4f}")
    print(f"  Mean per-entity F0.5 over {len(test_s1_ids)} test S1: "
          f"{float(np.mean(per_entity_f05)) if per_entity_f05 else 0.0:.4f}")

    # Sample-only outputs.
    print("\n=== Stage 5: Sample outputs ===")
    output_dir = Path(test_config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path, candidate_path = generate_outputs(
        dict(predictions), candidates_dict,
        singleton_probs={}, singleton_threshold=0.5,
        output_dir=output_dir)
    print(f"Wrote {matching_path} and {candidate_path}")
    print("\n✅ Train/test sample pipeline complete!")


if __name__ == '__main__':
    config = load_config()
    print("=== Running train/test sample pipeline (40 train + 10 test S1) ===")
    run_full_pipeline(config, sample_size=SAMPLE_SIZE, test_size=TEST_SIZE)
