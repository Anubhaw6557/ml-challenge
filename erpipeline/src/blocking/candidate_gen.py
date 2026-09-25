"""
Candidate Generation - Blocking + Recall Evaluation
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm

from .index import BlockingIndex, BlockingConfig, evaluate_blocking_recall
from ..config import get_config
from ..preprocessing.pipeline import PreprocessingPipeline


def generate_candidates(config: Dict = None, 
                        split: str = 'train',
                        use_faiss: bool = False,
                        s1_embeddings: np.ndarray = None) -> pl.DataFrame:
    """
    Generate candidate pairs for all S1 entities in a split
    """
    config = config or get_config()
    
    # Load processed data
    processed_dir = Path(config['paths']['processed_dir']) / split
    s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
    
    # Combine S2 and S3
    s23_df = pl.concat([s2_df, s3_df])
    
    # Build blocking index
    bc = BlockingConfig(
        name_ngram_size=config['blocking']['name_ngram_size'],
        name_min_overlap=config['blocking']['name_min_overlap'],
        metaphone_enabled=config['blocking']['metaphone_enabled'],
        postal_exact=config['blocking']['postal_exact'],
        city_state_exact=config['blocking']['city_state_exact'],
        house_number_exact=config['blocking']['house_number_exact'],
        minhash_enabled=config['blocking']['minhash_enabled'],
        minhash_threshold=config['blocking']['minhash_threshold'],
        minhash_perm=config['blocking']['minhash_perm'],
        biencoder_enabled=use_faiss,
        biencoder_top_k=config['blocking']['biencoder_top_k'],
        max_candidates_per_s1=config['blocking']['max_candidates_per_s1'],
    )
    
    index = BlockingIndex(bc)
    index.build_from_dataframe(s23_df)
    
    # Save index
    candidates_dir = Path(config['paths']['candidates_dir'])
    candidates_dir.mkdir(parents=True, exist_ok=True)
    index.save(str(candidates_dir / 'blocking_index'))
    
    # Generate candidates for S1
    print(f"Generating candidates for {len(s1_df)} S1 entities...")
    candidates_df = index.get_candidates_for_s1(s1_df, s1_embeddings)
    
    # Save candidates
    candidates_df.write_parquet(candidates_dir / f'candidates_{split}.parquet')
    
    print(f"Saved candidates to {candidates_dir / f'candidates_{split}.parquet'}")
    return candidates_df


def evaluate_recall(config: Dict = None, split: str = 'train') -> Dict:
    """
    Evaluate blocking recall on ground truth
    """
    config = config or get_config()
    
    # Load processed data
    processed_dir = Path(config['paths']['processed_dir']) / split
    print(f"Loading processed data from: {processed_dir}")
    s1_df = pl.read_parquet(processed_dir / 'source1.parquet')
    s2_df = pl.read_parquet(processed_dir / 'source2.parquet')
    s3_df = pl.read_parquet(processed_dir / 'source3.parquet')
    s23_df = pl.concat([s2_df, s3_df])
    print(f"Loaded {len(s1_df)} S1, {len(s2_df)} S2, {len(s3_df)} S3")
    
    # Load ground truth
    gt_path = Path(config['paths']['raw_dir']) / split / f'{split}_ground_truth.tsv'
    print(f"Loading ground truth from: {gt_path}")
    
    # Build blocking index
    bc = BlockingConfig(
        name_ngram_size=config['blocking']['name_ngram_size'],
        name_min_overlap=config['blocking']['name_min_overlap'],
        metaphone_enabled=config['blocking']['metaphone_enabled'],
        postal_exact=config['blocking']['postal_exact'],
        city_state_exact=config['blocking']['city_state_exact'],
        house_number_exact=config['blocking']['house_number_exact'],
        minhash_enabled=config['blocking']['minhash_enabled'],
        minhash_threshold=config['blocking']['minhash_threshold'],
        minhash_perm=config['blocking']['minhash_perm'],
        biencoder_enabled=False,  # No FAISS for recall eval
        max_candidates_per_s1=config['blocking']['max_candidates_per_s1'],
    )
    
    print("Building blocking index...")
    index = BlockingIndex(bc)
    index.build_from_dataframe(s23_df)
    print("Blocking index built")
    
    # Evaluate
    print("Evaluating recall...")
    s23_entity_data = {r['entity_id']: r for r in s23_df.iter_rows(named=True)}
    metrics = evaluate_blocking_recall(index, s1_df, str(gt_path), s23_entity_data)
    
    print("\n=== Blocking Recall Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Per-country breakdown
    print("\nPer-country recall:")
    for country in ['US', 'India']:
        country_s1 = s1_df.filter(pl.col('country') == country)
        if len(country_s1) > 0:
            country_metrics = evaluate_blocking_recall(index, country_s1, str(gt_path), 
                                                        {r['entity_id']: r for r in s23_df.iter_rows(named=True)})
            print(f"  {country}: overall={country_metrics['overall_recall']:.4f}, "
                  f"macro={country_metrics['macro_recall']:.4f}, "
                  f"pairs={country_metrics['total_gt_pairs']}")
    
    return metrics


def analyze_candidate_distribution(candidates_df: pl.DataFrame) -> Dict:
    """Analyze candidate set statistics"""
    stats = {
        'total_s1': len(candidates_df),
        'avg_candidates': candidates_df['num_candidates'].mean(),
        'median_candidates': candidates_df['num_candidates'].median(),
        'max_candidates': candidates_df['num_candidates'].max(),
        'min_candidates': candidates_df['num_candidates'].min(),
        'empty_candidates': (candidates_df['num_candidates'] == 0).sum(),
        'capped_candidates': (candidates_df['num_candidates'] >= 
                             200).sum(),  # Assuming max=200
    }
    
    print("\n=== Candidate Distribution ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--eval-recall', action='store_true')
    parser.add_argument('--use-faiss', action='store_true')
    args = parser.parse_args()
    
    if args.eval_recall:
        evaluate_recall(split=args.split)
    else:
        generate_candidates(split=args.split, use_faiss=args.use_faiss)
