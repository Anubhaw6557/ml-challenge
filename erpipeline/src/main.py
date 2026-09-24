"""
Main Pipeline Entry Point
"""

import argparse
import sys
from pathlib import Path

import polars as pl
from rich.console import Console

from .config import load_config
from .preprocessing.pipeline import PreprocessingPipeline
from .blocking.candidate_gen import generate_candidates, evaluate_recall

console = Console()


def run_preprocessing(config: dict):
    """Stage 1: Preprocess all source files"""
    console.print("[bold green]=== Stage 1: Preprocessing ===[/bold green]")
    pipeline = PreprocessingPipeline(config)
    pipeline.run()
    console.print("✅ Preprocessing complete")


def run_blocking(config: dict, split: str = 'train', eval_recall: bool = True):
    """Stage 2: Blocking + Candidate Generation"""
    console.print(f"[bold green]=== Stage 2: Blocking ({split}) ===[/bold green]")
    
    if eval_recall and split == 'train':
        metrics = evaluate_recall(config, split=split)
        console.print(f"[bold]Recall: {metrics['overall_recall']:.4f} (macro: {metrics['macro_recall']:.4f})[/bold]")
    else:
        generate_candidates(config, split=split)
    
    console.print("✅ Blocking complete")


def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--stage', choices=['preprocess', 'blocking', 'all'], 
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


if __name__ == '__main__':
    main()
