"""
End-to-end pipeline test on small sample with detailed per-entity output
"""
import polars as pl
import numpy as np
from pathlib import Path
import sys
import copy
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.blocking.candidate_gen import generate_candidates, evaluate_recall
from src.matching.features import FeatureExtractor
from src.matching.lgbm_model import LightGBMMatcher
from src.matching.biencoder import BiEncoderMatcher
from src.matching.threshold import ThresholdCalibrator
from src.postprocess.output import generate_outputs, validate_submission
from collections import defaultdict
from sklearn.model_selection import train_test_split


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
    
    for src, needed_ids in [('source1', s1_needed), ('source2', s2_needed), ('source3', s3_needed)]:
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


def run_full_pipeline(config: dict, sample_size: int = 50):
    """Run complete pipeline on small sample"""
    # Create sample
    print("=== Creating matched sample ===")
    create_matched_sample(config, sample_size)
    
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
    pipeline = PreprocessingPipeline(test_config)
    pipeline.run()
    
    # 2. Blocking + Generate Candidates + Recall
    print("\n=== Stage 2: Blocking + Generate Candidates + Recall ===")
    evaluate_recall(test_config, split='train')
    generate_candidates(test_config, split='train')
    
    # 3. Feature Extraction
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
    
    # Feature extraction
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
            feature_rows.append(features)
    
    features_df = pl.DataFrame(feature_rows)
    features_path = Path(test_config['paths']['candidates_dir']) / 'features_train.parquet'
    features_df.write_parquet(features_path)
    print(f"Extracted {len(features_df)} feature vectors")
    
    # Train LightGBM
    print("\n=== Stage 3b: Training LightGBM ===")
    from sklearn.model_selection import train_test_split
    from src.matching.lgbm_model import LightGBMMatcher
    
    train_df = features_df.filter(pl.col('label').is_not_null())
    train_pd = train_df.to_pandas()
    train_idx, val_idx = train_test_split(range(len(train_pd)), test_size=0.2, random_state=42, stratify=train_pd['label'])
    train_pairs = train_df[list(train_idx)]
    val_pairs = train_df[list(val_idx)]
    
    feature_cols = [c for c in train_pairs.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label', 'country']]
    
    lgbm = LightGBMMatcher(test_config['matching']['lgbm_params'])
    lgbm.train(train_pairs, val_pairs, feature_cols)
    
    # Calibrate thresholds
    print("\n=== Stage 3e: Threshold Calibration ===")
    from src.matching.threshold import ThresholdCalibrator
    calibrator = ThresholdCalibrator(beta=0.5)
    val_probs = lgbm.predict_proba(val_pairs, feature_cols)
    val_labels = val_pairs['label'].to_numpy()
    val_countries = val_pairs['country'].to_numpy() if 'country' in val_pairs.columns else np.array(['US'] * len(val_labels))
    calibrator.fit(val_probs, val_labels, val_countries)
    
    # Save models
    models_dir = Path(test_config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    lgbm.save(str(models_dir / 'lgbm_model.pkl'))
    from src.matching.threshold import ThresholdCalibrator
    calibrator.save(str(models_dir / 'thresholds.pkl'))
    
    # 4. Bi-encoder
    print("\n=== Stage 3c: Bi-encoder ===")
    biencoder = BiEncoderMatcher(test_config['matching']['biencoder_model'],
                                  test_config['matching']['biencoder_batch_size'])
    
    all_records = []
    for df in [s1_df, s2_df, s3_df]:
        for row in df.iter_rows(named=True):
            all_records.append({
                'entity_id': row['entity_id'],
                'name_clean': row['name_clean'],
                'addr_clean': row['addr_clean'],
            })
    biencoder.build_index(all_records)
    biencoder.save(str(Path(test_config['paths']['models_dir']) / 'biencoder'))
    
    # 5. Test Inference
    print("\n=== Stage 4: Test Inference ===")
    test_config2 = copy.deepcopy(config)
    test_config2['paths']['raw_dir'] = str(Path(config['paths']['raw_dir']).parent / 'sample')
    test_config2['paths']['processed_dir'] = str(Path(config['paths']['processed_dir']) / 'sample')
    test_config2['paths']['candidates_dir'] = str(Path(config['paths']['candidates_dir']).parent / 'sample_candidates')
    test_config2['paths']['models_dir'] = str(Path(config['paths']['models_dir']) / 'sample')
    
    # Generate test candidates
    print("Generating test candidates...")
    generate_candidates(test_config2, split='test')
    
    # Load test data
    test_processed_dir = Path(test_config2['paths']['processed_dir']) / 'test'
    s1_test = pl.read_parquet(test_processed_dir / 'source1.parquet')
    s2_test = pl.read_parquet(test_processed_dir / 'source2.parquet')
    s3_test = pl.read_parquet(test_processed_dir / 'source3.parquet')
    s23_test = pl.concat([s2_test, s3_test])
    s23_test_lookup = {row['entity_id']: row for row in s23_test.iter_rows(named=True)}
    s1_test_lookup = {row['entity_id']: row for row in s1_test.iter_rows(named=True)}
    
    candidates_test_path = Path(test_config2['paths']['candidates_dir']) / 'candidates_test.parquet'
    candidates_test = pl.read_parquet(candidates_test_path)
    
    # Extract features for test
    print("Extracting test features...")
    test_feature_rows = []
    for row in candidates_test.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        cand_ids = row['candidate_entity_ids']
        if isinstance(cand_ids, list):
            cand_ids = cand_ids
        elif isinstance(cand_ids, str):
            cand_ids = cand_ids.split(',') if cand_ids else []
        else:
            cand_ids = []
        
        s1_data = s1_test_lookup.get(s1_id)
        if not s1_data:
            continue
        for cand_id in cand_ids:
            cand_row = s23_test_lookup.get(cand_id)
            if not cand_row:
                continue
            features = FeatureExtractor().extract(s1_data, cand_row)
            features['source1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            features['label'] = 0
            features['country'] = s1_data.get('country', 'US')
            test_feature_rows.append(features)
    
    test_features_df = pl.DataFrame(test_feature_rows)
    test_features_path = Path(test_config2['paths']['candidates_dir']) / 'features_test.parquet'
    test_features_df.write_parquet(test_features_path)
    print(f"Extracted {len(test_features_df)} test feature vectors")
    
    # Predict
    print("Predicting...")
    test_feature_cols = [c for c in test_features_df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label', 'country', 's1_name', 's2_name']]
    probs = lgbm.predict_proba(test_features_df, test_feature_cols)
    test_countries = test_features_df['country'].to_numpy()
    calibrator = ThresholdCalibrator.load(str(Path(test_config['paths']['models_dir']) / 'thresholds.pkl'))
    preds = calibrator.predict(probs, test_countries)
    
    # Build predictions dict
    predictions = defaultdict(list)
    for i, row in enumerate(test_features_df.iter_rows(named=True)):
        if preds[i] == 1:
            predictions[row['source1_entity_id']].append(row['candidate_entity_id'])
    
    # Load test candidates
    candidates_test_path = Path(test_config2['paths']['candidates_dir']) / 'candidates_test.parquet'
    candidates_test_df = pl.read_parquet(candidates_test_path)
    candidates_dict = {}
    for row in candidates_test_df.iter_rows(named=True):
        cand_ids = row['candidate_entity_ids']
        if isinstance(cand_ids, list):
            candidates_dict[row['source1_entity_id']] = cand_ids
        elif isinstance(cand_ids, str):
            candidates_dict[row['source1_entity_id']] = cand_ids.split(',') if cand_ids else []
        else:
            candidates_dict[row['source1_entity_id']] = []
    
    # Post-process and output
    print("\n=== Stage 5: Post-processing & Output ===")
    from src.postprocess.output import generate_outputs, validate_submission
    
    output_dir = Path(test_config2['paths']['output_dir'])
    matching_path, candidate_path = generate_outputs(
        predictions, candidates_dict, 
        singleton_probs={}, singleton_threshold=0.5,
        output_dir=output_dir
    )
    
    # Validate
    test_dir = Path(test_config['validation']['test_dir'])
    validate_submission(matching_path, candidate_path, test_dir)
    
    print("\n✅ Full pipeline test complete!")


if __name__ == '__main__':
    import numpy as np
    from collections import defaultdict
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    
    config = load_config()
    
    print("=== Running Full Pipeline on Small Sample (50 entities) ===")
    run_full_pipeline(config, sample_size=50)
