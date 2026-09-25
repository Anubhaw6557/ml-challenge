"""
Test script to run pipeline on a sample that preserves ground truth matches
"""
import polars as pl
from pathlib import Path
import sys
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.blocking.candidate_gen import evaluate_recall


def create_small_sample(config: dict, sample_size: int = 100):
    """Create sample that preserves ground truth matches - OPTIMIZED"""
    raw_dir = Path(config['paths']['raw_dir'])
    sample_raw_dir = Path(config['paths']['raw_dir']).parent / 'sample' / 'train'
    sample_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load ground truth and pick S1 entities with matches
    gt_path = raw_dir / 'train' / 'train_ground_truth.tsv'
    print(f"Loading ground truth...")
    gt_df = pl.read_csv(gt_path, separator='\t', columns=['source1_entity_id', 'matched_entity_ids'])
    
    # Filter S1 entities that HAVE matches
    gt_with_matches = gt_df.filter(pl.col('matched_entity_ids') != '')
    print(f"S1 entities with matches: {len(gt_with_matches)}")
    
    sample_s1_ids = gt_with_matches['source1_entity_id'].head(sample_size).to_list()
    print(f"Sampling {len(sample_s1_ids)} S1 entities with matches")
    
    # Get matched S2/S3 IDs
    matched_ids = set()
    for matches in gt_df.filter(pl.col('source1_entity_id').is_in(sample_s1_ids))['matched_entity_ids']:
        matched_ids.update(matches.split(','))
    print(f"Total matched S2/S3 IDs: {len(matched_ids)}")
    
    # 3. ONLY load the needed entities using polars filter (much faster)
    sample_raw_dir = Path(config['paths']['raw_dir']).parent / 'sample' / 'train'
    sample_raw_dir.mkdir(parents=True, exist_ok=True)
    
    all_needed_ids = set(sample_s1_ids) | matched_ids
    s1_needed = [eid for eid in all_needed_ids if eid.startswith('S1-')]
    s2_needed = [eid for eid in all_needed_ids if eid.startswith('S2-')]
    s3_needed = [eid for eid in all_needed_ids if eid.startswith('S3-')]
    
    print(f"Loading {len(s1_needed)} S1, {len(s2_needed)} S2, {len(s3_needed)} S3 entities...")
    
    for src, needed_ids in [('source1', s1_needed), ('source2', s2_needed), ('source3', s3_needed)]:
        if not needed_ids:
            continue
        in_path = raw_dir / 'train' / f'train_{src}.tsv'
        # Use polars filter - much faster than loading all then filtering in Python
        df = pl.read_csv(in_path, separator='\t')
        filtered = df.filter(pl.col('entity_id').is_in(needed_ids))
        
        if len(filtered) > 0:
            out_path = sample_raw_dir / f'train_{src}.tsv'
            sample_raw_dir.mkdir(parents=True, exist_ok=True)
            filtered.write_csv(out_path, separator='\t')
            print(f"Created sample: {out_path} ({len(filtered)} rows)")
        else:
            print(f"No rows found for {src}")
    
    # Write filtered ground truth
    gt_out = sample_raw_dir.parent / 'train' / 'train_ground_truth.tsv'
    filtered_gt = gt_df.filter(pl.col('source1_entity_id').is_in(sample_s1_ids))
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    filtered_gt.write_csv(gt_out, separator='\t')
    print(f"Created sample ground truth: {gt_out} ({len(filtered_gt)} rows)")


def test_preprocessing(config: dict):
    """Test preprocessing on small sample"""
    test_config = copy.deepcopy(config)
    test_config['paths']['raw_dir'] = str(Path(config['paths']['raw_dir']).parent / 'sample')
    test_config['paths']['processed_dir'] = str(Path(config['paths']['processed_dir']) / 'sample')
    
    pipeline = PreprocessingPipeline(test_config)
    
    for src in ['source1', 'source2', 'source3']:
        in_path = Path(test_config['paths']['raw_dir']) / 'train' / f'train_{src}.tsv'
        out_path = Path(test_config['paths']['processed_dir']) / 'train' / f'{src}.parquet'
        
        if in_path.exists():
            pipeline.process_file(in_path, out_path)


def test_blocking(config: dict):
    """Test blocking on matched sample"""
    test_config = copy.deepcopy(config)
    test_config['paths']['processed_dir'] = str(Path(config['paths']['processed_dir']) / 'sample')
    test_config['paths']['candidates_dir'] = str(Path(config['paths']['candidates_dir']).parent / 'sample_candidates')
    
    test_config['blocking']['max_candidates_per_s1'] = 50
    
    evaluate_recall(test_config, split='train')


if __name__ == '__main__':
    import copy
    config = load_config()
    
    print("=== Creating matched sample (100 S1 entities with matches) ===")
    create_small_sample(config, sample_size=100)
    
    print("\n=== Testing preprocessing ===")
    test_preprocessing(config)
    
    print("\n=== Testing blocking & recall ===")
    test_blocking(config)
    
    print("\n✅ Sample test complete")
