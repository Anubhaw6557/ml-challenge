"""
Test script to run pipeline on a small sample (1000 rows per source)
"""
import polars as pl
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from preprocessing.pipeline import PreprocessingPipeline
from blocking.candidate_gen import generate_candidates, evaluate_recall


def create_small_sample(config: dict, sample_size: int = 1000):
    """Create small sample from raw TSV files for quick testing"""
    raw_dir = Path(config['paths']['raw_dir'])
    sample_dir = Path(config['paths']['processed_dir']).parent / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train']:
        for src in ['source1', 'source2', 'source3']:
            in_path = raw_dir / split / f'{split}_{src}.tsv'
            out_path = sample_dir / f'{src}.tsv'
            
            if in_path.exists():
                df = pl.read_csv(in_path, separator='\t')
                sample_df = df.head(sample_size)
                sample_df.write_csv(out_path, separator='\t')
                print(f"Created sample: {out_path} ({len(sample_df)} rows)")


def test_preprocessing(config: dict):
    """Test preprocessing on small sample"""
    sample_dir = Path(config['paths']['processed_dir']).parent / 'sample'
    processed_dir = Path(config['paths']['processed_dir']) / 'sample'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PreprocessingPipeline(config)
    
    for src in ['source1', 'source2', 'source3']:
        in_path = sample_dir / f'{src}.tsv'
        out_path = processed_dir / f'{src}.parquet'
        
        if in_path.exists():
            pipeline.process_file(in_path, out_path)


def test_blocking(config: dict):
    """Test blocking on small sample"""
    # Temporarily modify config to use sample data
    test_config = config.copy()
    test_config['paths']['processed_dir'] = str(
        Path(config['paths']['processed_dir']).parent / 'sample'
    )
    test_config['paths']['candidates_dir'] = str(
        Path(config['paths']['candidates_dir']).parent / 'sample_candidates'
    )
    
    evaluate_recall(test_config, split='sample')


if __name__ == '__main__':
    config = load_config()
    
    print("=== Creating small sample ===")
    create_small_sample(config, sample_size=1000)
    
    print("\n=== Testing preprocessing ===")
    test_preprocessing(config)
    
    print("\n=== Testing blocking & recall ===")
    test_blocking(config)
    
    print("\n✅ Small sample test complete")
