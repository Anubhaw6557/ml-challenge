"""
Test pipeline with train/test split from same sample
- Create sample of 50 S1 entities with matches
- Split into 40 train + 10 test S1 entities
- Train on 40, test on 10 using same blocking
"""
import polars as pl
import numpy as np
from pathlib import Path
import sys
import copy

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.blocking.candidate_gen import generate_candidates
from src.matching.features import FeatureExtractor
from src.matching.lgbm_model import LightGBMMatcher
from src.matching.threshold import ThresholdCalibrator
from src.postprocess.output import generate_outputs


SAMPLE_SIZE = 50
TEST_SIZE = 10


def create_matched_sample(config: dict, sample_size: int = 50):
    """Create sample that preserves ground truth matches"""
    raw_dir = Path(config['paths']['raw_dir'])
    sample_raw_dir = Path(config['paths']['raw_dir']).parent / 'sample' / 'train'
    sample_raw_dir.mkdir(parents=True, exist_ok=True)
    
    gt_path = raw_dir / 'train' / 'train_ground_truth.tsv'
    gt_df = pl.read_csv(gt_path, separator='\t', columns=['source1_entity_id', 'matched_entity_ids'])
    
    gt_with_matches = gt_df.filter(pl.col('matched_entity_ids') != '')
    sample_s1_ids = gt_with_matches['source1_entity_id'].head(sample_size).to_list()
    
    matched_ids = set()
    for matches in gt_df.filter(pl.col('source1_entity_id').is_in(sample_s1_ids))['matched_entity_ids']:
        matched_ids.update(matches.split(','))
    
    s1_needed = [eid for eid in sample_s1_ids]
    s2_needed = [eid for eid in matched_ids if eid.startswith('S2-')]
    s3_needed = [eid for eid in matched_ids if eid.startswith('S3-')]
    
    sample_raw_dir = Path(config['paths']['raw_dir']).parent / 'sample' / 'train'
    sample_raw_dir.mkdir(parents=True, exist_ok=True)
    
    for src, needed_ids in [('source1', [eid for eid in sample_s1_ids]), 
                             ('source2', [eid for eid in matched_ids if eid.startswith('S2-')]), 
                             ('source3', [eid for eid in matched_ids if eid.startswith('S3-')])]:
        if not needed_ids:
            continue
        in_path = raw_dir / 'train' / f'train_{src}.tsv'
        df = pl.read_csv(in_path, separator='\t')
        filtered = df.filter(pl.col('entity_id').is_in(needed_ids))
        if len(filtered) > 0:
            out_path = sample_raw_dir / f'train_{src}.tsv'
            sample_raw_dir.mkdir(parents=True, exist_ok=True)
            filtered.write_csv(out_path, separator='\t')
            print(f"Created {src}: {len(filtered)} rows")
    
    gt_out = sample_raw_dir.parent / 'train' / 'train_ground_truth.tsv'
    filtered_gt = gt_df.filter(pl.col('source1_entity_id').is_in(sample_s1_ids))
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    filtered_gt.write_csv(gt_out, separator='\t')
    
    return sample_s1_ids


def run_train_test_split(config: dict, sample_size: int = 50, test_size: int = 10):
    """Run train/test split evaluation"""
    # Create sample
    print("=== Creating matched sample ===")
    sample_s1_ids = create_matched_sample(config, sample_size)
    
    # Split into train and test
    train_s1_ids = sample_s1_ids[:sample_size - test_size]
    test_s1_ids = sample_s1_ids[sample_size - test_size:]
    print(f"Train S1: {len(train_s1_ids)}, Test S1: {len(test_s1_ids)}")
    
    # Override paths for sample
    test_config = copy.deepcopy(config)
    test_config['paths']['raw_dir'] = str(Path(config['paths']['raw_dir']).parent / 'sample')
    test_config['paths']['processed_dir'] = str(Path(config['paths']['processed_dir']) / 'sample')
    test_config['paths']['candidates_dir'] = str(Path(config['paths']['candidates_dir']).parent / 'sample_candidates')
    test_config['paths']['models_dir'] = str(Path(config['paths']['models_dir']) / 'sample')
    test_config['paths']['output_dir'] = str(Path(config['paths']['output_dir']) / 'sample')
    test_config['blocking']['max_candidates_per_s1'] = 50
    
    # 1. Preprocessing
    print("\n=== Stage 1: Preprocessing ===")
    pipeline = PreprocessingPipeline(config)
    pipeline.run()
    
    # 2. Blocking + Generate Candidates
    print("\n=== Stage 2: Blocking + Generate Candidates ===")
    generate_candidates(test_config, split='train')
    
    # 3. Feature Extraction for all
    print("\n=== Stage 3a: Feature Extraction ===")
    processed_dir = Path(test_config['paths']['processed_dir']) / 'train'
    candidates_dir = Path(test_config['paths']['candidates_dir'])
    
    s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
    s23_df = pl.concat([s2_df, s3_df])
    s23_lookup = {row['entity_id']: row for row in s23_df.iter_rows(named=True)}
    s1_lookup = {row['entity_id']: row for row in s1_df.iter_rows(named=True)}
    
    candidates_df = pl.read_parquet(candidates_dir / 'candidates_train.parquet')
    
    # Ground truth labels
    gt_path = Path(test_config['paths']['raw_dir']) / 'train' / 'train_ground_truth.tsv'
    gt_labels = {}
    with open(gt_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0]
                for m in parts[1].split(','):
                    gt_labels[(s1_id, m)] = 1
    
    # Feature extraction for all pairs
    extractor = FeatureExtractor()
    all_names = []
    for df in [s1_df, s2_df, s3_df]:
        all_names.extend(df['name_clean'].to_list())
    extractor.fit(all_names)
    
    print("Extracting features...")
    feature_rows = []
    for row in candidates_df.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        cand_ids = row['candidate_entity_ids']
        if isinstance(cand_ids, list):
            cand_ids = cand_ids
        elif isinstance(cand_ids, str):
            cand_ids = cand_ids.split(',') if cand_ids else []
        else:
            cand_ids = []
        
        s1_data = s1_lookup.get(s1_id)
        if not s1_data:
            continue
        for cand_id in cand_ids:
            cand_row = s23_lookup.get(cand_id)
            if not cand_row:
                continue
            label = 1 if (s1_id, cand_id) in gt_labels else 0
            features = extractor.extract(s1_data, cand_row)
            features['source1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            features['label'] = label
            features['country'] = s1_data.get('country', 'US')
            features['is_test'] = 1 if s1_id in test_s1_ids else 0
            feature_rows.append(features)
    
    features_df = pl.DataFrame(feature_rows)
    features_path = Path(test_config['paths']['candidates_dir']) / 'features_train.parquet'
    features_df.write_parquet(features_path)
    print(f"Extracted {len(features_df)} feature vectors")
    
    # Split features into train/test based on S1 IDs
    train_features = features_df.filter(~pl.col('source1_entity_id').is_in(test_s1_ids))
    test_features = features_df.filter(pl.col('source1_entity_id').is_in(test_s1_ids))
    
    print(f"Train features: {len(train_features)}, Test features: {len(test_features)}")
    
    # Train LightGBM on train split
    print("\n=== Stage 3b: Training LightGBM (on 40 entities) ===")
    train_df = train_features.filter(pl.col('label').is_not_null())
    
    from sklearn.model_selection import train_test_split
    train_pd = train_df.to_pandas()
    train_idx, val_idx = train_test_split(range(len(train_pd)), test_size=0.2, random_state=42, stratify=train_pd['label'])
    train_pairs = train_df[list(train_idx)]
    val_pairs = train_df[list(val_idx)]
    
    # Exclude non-feature columns
    exclude_cols = ['source1_entity_id', 'candidate_entity_id', 'label', 'country', 'is_test']
    feature_cols = [c for c in train_pairs.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label', 'country']]
    
    lgbm = LightGBMMatcher(test_config['matching']['lgbm_params'])
    lgbm.train(train_pairs, val_pairs, feature_cols)
    
    # Calibrate thresholds on validation
    print("\n=== Stage 3e: Threshold Calibration ===")
    calibrator = ThresholdCalibrator(beta=0.5)
    val_probs = lgbm.predict_proba(val_pairs, feature_cols)
    val_labels = val_pairs['label'].to_numpy()
    val_countries = val_pairs['country'].to_numpy() if 'country' in val_pairs.columns else np.array(['US'] * len(val_labels))
    calibrator.fit(val_probs, val_labels, val_countries)
    
    # Save models
    models_dir = Path(test_config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    lgbm.save(str(models_dir / 'lgbm_model.pkl'))
    calibrator.save(str(models_dir / 'thresholds.pkl'))
    
    # 4. Test Inference on held-out test entities (10 entities)
    print("\n=== Stage 4: Test Inference (on 10 held-out entities) ===")
    
    test_feature_cols = [c for c in test_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label', 'country']]
    
    if len(test_features) > 0:
        # Predict
        probs = lgbm.predict_proba(test_features, test_feature_cols)
        test_countries = test_features['country'].to_numpy()
        preds = calibrator.predict(probs, test_countries)
        
        # Build predictions dict
        from collections import defaultdict
        predictions = defaultdict(list)
        for i, row in enumerate(test_features.iter_rows(named=True)):
            if preds[i] == 1:
                predictions[row['source1_entity_id']].append(row['candidate_entity_id'])
        
        # Evaluate on test set (we have GT for test entities)
        test_labels = test_features['label'].to_numpy()
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='macro', zero_division=0)
        macro_f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        print(f"\n✅ Test Results (10 held-out entities):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  Macro F0.5: {macro_f05:.4f}")
        
        # Per-entity results
        print(f"\nPer-entity predictions (first 10):")
        for s1_id in list(test_s1_ids)[:10]:
            pred_matches = predictions.get(s1_id, [])
            print(f"  {s1_id}: {len(pred_matches)} predicted matches: {pred_matches[:5]}")
    else:
        print("No test features found!")
    
    # 5. Post-processing & Output
    print("\n=== Stage 5: Post-processing & Output ===")
    output_dir = Path(test_config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs for test entities
    candidates_dict = {}
    for row in candidates_df.filter(pl.col('source1_entity_id').is_in(test_s1_ids)).iter_rows(named=True):
        cand_ids = row['candidate_entity_ids']
        if isinstance(cand_ids, list):
            candidates_dict[row['source1_entity_id']] = cand_ids
        elif isinstance(cand_ids, str):
            candidates_dict[row['source1_entity_id']] = cand_ids.split(',') if cand_ids else []
        else:
            candidates_dict[row['source1_entity_id']] = []
    
    from src.postprocess.output import generate_outputs, validate_submission
    matching_path, candidate_path = generate_outputs(
        dict(predictions), candidates_dict, 
        singleton_probs={}, singleton_threshold=0.5,
        output_dir=output_dir
    )
    
    print("\n✅ Train/Test split pipeline complete!")


if __name__ == '__main__':
    config = load_config()
    print("=== Running Train/Test Split Pipeline (40 train + 10 test) ===")
    run_train_test_split(config, sample_size=50, test_size=10)
