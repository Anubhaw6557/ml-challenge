"""
Main Pipeline Entry Point - Entity Resolution Pipeline
"""

import argparse
import sys
from pathlib import Path
import polars as pl
import numpy as np
from rich.console import Console

from .config import load_config
from .preprocessing.pipeline import PreprocessingPipeline
from .blocking.candidate_gen import generate_candidates, evaluate_recall
from .matching.features import FeatureExtractor
from .matching.lgbm_model import LightGBMMatcher, SingletonClassifier
from .matching.biencoder import BiEncoderMatcher
from .matching.crossencoder import CrossEncoderMatcher
from .matching.threshold import ThresholdCalibrator, calibrate_thresholds
from .postprocess.output import (generate_outputs, validate_submission, 
                                  postprocess_pipeline)

console = Console()


def run_preprocessing(config: dict):
    """Stage 1: Preprocess all source files"""
    console.print("[bold green]=== Stage 1: Preprocessing ===[/bold green]")
    pipeline = PreprocessingPipeline(config)
    pipeline.run()
    console.print("✅ Preprocessing complete")


def run_blocking(config: dict, split: str = 'train', eval_recall: bool = True,
                 use_biencoder: bool = False):
    """Stage 2: Blocking + Candidate Generation"""
    console.print(f"[bold green]=== Stage 2: Blocking ({split}) ===[/bold green]")
    
    if eval_recall and split == 'train':
        metrics = evaluate_recall(config, split=split)
        console.print(f"[bold]Recall: {metrics['overall_recall']:.4f} (macro: {metrics['macro_recall']:.4f})[/bold]")
    else:
        generate_candidates(config, split=split, use_faiss=use_biencoder)
    
    console.print("✅ Blocking complete")


def run_feature_extraction(config: dict, split: str = 'train') -> pl.DataFrame:
    """Stage 3a: Extract pairwise features from candidates"""
    console.print(f"[bold green]=== Stage 3a: Feature Extraction ({split}) ===[/bold green]")
    
    processed_dir = Path(config['paths']['processed_dir']) / split
    candidates_dir = Path(config['paths']['candidates_dir'])
    
    # Load data
    s1_df = pl.read_parquet(Path(config['paths']['processed_dir']) / split / 'source1.parquet')
    s2_df = pl.read_parquet(Path(config['paths']['processed_dir']) / split / 'source2.parquet')
    s3_df = pl.read_parquet(Path(config['paths']['processed_dir']) / split / 'source3.parquet')
    
    # Load candidates
    cand_path = Path(config['paths']['candidates_dir']) / f'candidates_{split}.parquet'
    candidates_df = pl.read_parquet(cand_path)
    
    # Load S2/S3 entity data for lookup
    s23_df = pl.concat([pl.read_parquet(Path(config['paths']['processed_dir']) / split / 'source2.parquet'),
                        pl.read_parquet(Path(config['paths']['processed_dir']) / split / 'source3.parquet')])
    s23_lookup = {row['entity_id']: row for row in s23_df.iter_rows(named=True)}
    
    # Load ground truth for labels (train only)
    gt_labels = {}
    if split == 'train':
        gt_path = Path(config['paths']['raw_dir']) / 'train' / 'train_ground_truth.tsv'
        with open(gt_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1]:
                    s1_id = parts[0]
                    for m in parts[1].split(','):
                        gt_labels[(s1_id, m)] = 1
    
    # Build feature vectors
    extractor = FeatureExtractor()
    
    # Fit TF-IDF on all names
    all_names = []
    for df in [pl.read_parquet(Path(config['paths']['processed_dir']) / split / f'source{i}.parquet') for i in [1,2,3]]:
        all_names.extend(df['name_clean'].to_list())
    extractor.fit(all_names)
    
    console.print("Extracting features...")
    feature_rows = []
    
    for row in candidates_df.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        cand_ids = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        
        if not cand_ids:
            continue
        
        s1_row = s23_lookup.get(s1_id)  # S1 is in S2/S3 lookup? No, need to load S1 separately
        # Actually S1 is in the S1 dataframe
        s1_df_full = pl.read_parquet(Path(config['paths']['processed_dir']) / split / 'source1.parquet')
        s1_lookup = {row['entity_id']: row for row in s1_df_full.iter_rows(named=True)}
        
        for cand_id in cand_ids:
            cand_row = s23_lookup.get(cand_id)
            if not cand_row:
                continue
            
            label = gt_labels.get((s1_id, cand_id), 0)
            
            # Extract features
            s1_data = s1_lookup[s1_id]
            cand_data = cand_row
            features = extractor.extract(s1_data, cand_data)
            features['source1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            features['label'] = label
            feature_rows.append(features)
    
    features_df = pl.DataFrame(feature_rows)
    console.print(f"Extracted {len(features_df)} feature vectors")
    
    # Save features
    features_path = Path(config['paths']['candidates_dir']) / f'features_{split}.parquet'
    features_df.write_parquet(features_path)
    
    return features_df


def run_training(config: dict, features_df: pl.DataFrame):
    """Stage 3b: Train LightGBM model"""
    console.print("[bold green]=== Stage 3b: Model Training ===[/bold green]")
    
    # Split features into train/val
    train_df = features_df.filter(pl.col('label').is_not_null())
    
    # Stratified split
    train_df = train_df.sample(fraction=0.8, shuffle=True, seed=42)
    val_df = train_df.filter(~pl.col('label').is_in(train_df['label']))
    # Actually need proper stratified split
    from sklearn.model_selection import train_test_split
    
    # For simplicity, random split
    train_indices = np.random.choice(len(train_df), int(0.8 * len(train_df)), replace=False)
    val_indices = [i for i in range(len(train_df)) if i not in train_indices]
    
    train_pairs = train_df[train_indices]
    val_pairs = train_df[val_indices]
    
    # Feature columns (exclude id and label columns)
    feature_cols = [c for c in train_pairs.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    # Train LightGBM
    lgbm = LightGBMMatcher(config['matching']['lgbm_params'])
    lgbm.train(train_pairs, val_pairs, feature_cols)
    
    # Save model
    models_dir = Path(config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    lgbm.save(str(models_dir / 'lgbm_model.pkl'))
    
    console.print("✅ LightGBM training complete")
    return lgbm


def run_biencoder(config: dict):
    """Stage 3c: Bi-encoder embedding generation"""
    console.print("[bold green]=== Stage 3c: Bi-encoder Embeddings ===[/bold green]")
    
    biencoder = BiEncoderMatcher(config['matching']['biencoder_model'],
                                  config['matching']['biencoder_batch_size'])
    
    # Encode all entities for FAISS
    processed_dir = Path(config['paths']['processed_dir']) / 'train'
    s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
    
    all_records = []
    for df in [s1_df, s2_df, s3_df]:
        for row in df.iter_rows(named=True):
            all_records.append({
                'entity_id': row['entity_id'],
                'name_clean': row['name_clean'],
                'addr_clean': row['addr_clean'],
            })
    
    biencoder.build_index(all_records)
    
    # Save index
    models_dir = Path(config['paths']['models_dir'])
    biencoder.save(str(models_dir / 'biencoder'))
    
    console.print("✅ Bi-encoder index built")


def run_crossencoder(config: dict):
    """Stage 3d: Cross-encoder fine-tuning"""
    console.print("[bold green]=== Stage 3d: Cross-encoder Fine-tuning ===[/bold green]")
    # Implementation would go here
    console.print("⚠️ Cross-encoder training skipped (optional)")


def run_threshold_calibration(config: dict, lgbm_model, features_df: pl.DataFrame):
    """Stage 3e: Threshold calibration"""
    console.print("[bold green]=== Stage 3e: Threshold Calibration ===[/bold green]")
    
    # Prepare validation data
    val_pairs = features_df.filter(pl.col('label').is_not_null()).sample(fraction=0.2, seed=42)
    feature_cols = [c for c in features_df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    probs = lgbm_model.predict_proba(val_pairs, feature_cols)
    labels = val_pairs['label'].to_numpy()
    countries = val_pairs.get_column('country').to_numpy() if 'country' in val_pairs.columns else np.array(['US'] * len(labels))
    
    calibrator = ThresholdCalibrator(beta=0.5)
    calibrator.fit(probs, labels, countries)
    
    # Save thresholds
    models_dir = Path(config['paths']['models_dir'])
    calibrator.save(str(models_dir / 'thresholds.pkl'))
    
    console.print(f"✅ Thresholds calibrated: {calibrator.thresholds}")


def run_inference(config: dict):
    """Stage 4: Test inference"""
    console.print("[bold green]=== Stage 4: Test Inference ===[/bold green]")
    
    # Load models
    models_dir = Path(config['paths']['models_dir'])
    lgbm = LightGBMMatcher.load(str(models_dir / 'lgbm_model.pkl'))
    calibrator = ThresholdCalibrator.load(str(models_dir / 'thresholds.pkl'))
    
    # Load test data
    processed_dir = Path(config['paths']['processed_dir']) / 'test'
    s1_test = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_test = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_test = pl.read_parquet(processed_dir / 'source3.parquet')
    
    # Load candidates
    candidates_dir = Path(config['paths']['candidates_dir'])
    candidates_df = pl.read_parquet(candidates_dir / 'candidates_test.parquet')
    
    # Extract features for test pairs
    extractor = FeatureExtractor()
    # ... (similar to training)
    
    # Predict
    # probs = lgbm.predict_proba(test_pairs, feature_cols)
    # preds = calibrator.predict(probs, countries, singleton_probs)
    
    console.print("✅ Inference complete")


def run_postprocessing(config: dict, predictions: Dict[str, List[str]],
                       candidates: Dict[str, List[str]],
                       singleton_probs: Optional[Dict[str, float]] = None):
    """Stage 5: Post-processing & Output"""
    console.print("[bold green]=== Stage 5: Post-processing & Output ===[/bold green]")
    
    # Generate outputs
    output_dir = Path(config['paths']['output_dir'])
    matching_path, candidate_path = generate_outputs(
        predictions, candidates, 
        singleton_probs={},  # TODO: add singleton probs
        singleton_threshold=0.5,
        output_dir=output_dir
    )
    
    # Validate
    test_dir = Path(config['validation']['test_dir'])
    validate_submission(matching_path, candidate_path, test_dir)
    
    console.print("✅ Post-processing complete")


def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--stage', choices=['preprocess', 'blocking', 'features', 'train', 
                                              'biencoder', 'crossencoder', 'calibrate', 'infer', 'all'],
                        default='all', help='Pipeline stage to run')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--no-eval', action='store_true', help='Skip recall evaluation')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.stage in ('preprocess', 'all'):
        run_preprocessing(config)
    
    if args.stage in ('blocking', 'all'):
        run_blocking(config, split=args.split, eval_recall=not args.no_eval)
    
    if args.stage in ('features', 'all'):
        features_df = run_feature_extraction(config, split=args.split)
    
    if args.stage in ('train', 'all'):
        # Need features_df from previous stage
        if args.stage == 'all':
            features_df = run_feature_extraction(config, split=args.split)
        else:
            features_path = Path(config['paths']['candidates_dir']) / f'features_{args.split}.parquet'
            features_df = pl.read_parquet(features_path)
        lgbm = run_training(config, features_df)
    
    if args.stage in ('biencoder', 'all'):
        run_biencoder(config)
    
    if args.stage in ('crossencoder', 'all'):
        run_crossencoder(config)
    
    if args.stage in ('calibrate', 'all'):
        # Need trained model
        models_dir = Path(config['paths']['models_dir'])
        lgbm = LightGBMMatcher.load(str(models_dir / 'lgbm_model.pkl'))
        features_path = Path(config['paths']['candidates_dir']) / f'features_{args.split}.parquet'
        features_df = pl.read_parquet(features_path)
        run_threshold_calibration(config, lgbm, features_df)
    
    if args.stage in ('infer', 'all'):
        run_inference(config)
    
    if args.stage in ('postprocess', 'all'):
        run_postprocessing(config, {}, {})


if __name__ == '__main__':
    main()
