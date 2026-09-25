"""
Post-processing and Output Generation
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def deduplicate_matches(matches: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove duplicate candidate IDs from matches"""
    result = {}
    for s1_id, cand_list in matches.items():
        seen = set()
        unique = []
        for cid in cand_list:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)
        result[s1_id] = unique
    return result


def filter_matches_by_candidates(matches: Dict[str, List[str]],
                                 candidates: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Ensure all matches are subset of candidates"""
    result = {}
    for s1_id, match_list in matches.items():
        cand_set = set(candidates.get(s1_id, []))
        result[s1_id] = [m for m in match_list if m in cand_set]
    return result


def apply_singleton_filter(matches: Dict[str, List[str]],
                           singleton_probs: Dict[str, float],
                           threshold: float = 0.5) -> Dict[str, List[str]]:
    """Remove matches for predicted singletons"""
    result = {}
    for s1_id, match_list in matches.items():
        if singleton_probs.get(s1_id, 0) > threshold:
            result[s1_id] = []
        else:
            result[s1_id] = match_list
    return result


def write_matching_results(matches: Dict[str, List[str]], output_path: Path):
    """Write matching_results.tsv"""
    rows = []
    for s1_id in sorted(matches.keys()):
        matches_str = ','.join(matches[s1_id]) if matches[s1_id] else ''
        rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': matches_str})
    
    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path, separator='\t')
    print(f"Written {len(rows)} rows to {output_path}")


def write_candidate_pairs(candidates: Dict[str, List[str]], output_path: Path):
    """Write candidate_pairs.tsv"""
    rows = []
    for s1_id in sorted(candidates.keys()):
        cands_str = ','.join(candidates[s1_id]) if candidates[s1_id] else ''
        rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cands_str})
    
    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path, separator='\t')
    print(f"Written {len(rows)} rows to {output_path}")


def validate_submission(matching_path: Path, candidate_path: Path, test_dir: Path):
    """Run validation script"""
    import subprocess
    result = subprocess.run([
        'python3', 'utils/validate_submission.py',
        '--matching', str(matching_path),
        '--candidate', str(candidate_path),
        '--test-dir', str(test_dir)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Validation PASSED")
    else:
        print("❌ Validation FAILED:")
        print(result.stdout)
        print(result.stderr)
    return result.returncode == 0


def postprocess_pipeline(predictions: Dict[str, List[str]],
                         candidates: Dict[str, List[str]],
                         singleton_probs: Optional[Dict[str, float]] = None,
                         singleton_threshold: float = 0.5) -> Dict[str, List[str]]:
    """Full post-processing pipeline"""
    
    # 1. Deduplicate
    matches = deduplicate_matches(predictions)
    
    # 2. Ensure matches are subset of candidates
    matches = filter_matches_by_candidates(matches, candidates)
    
    # 3. Singleton filter
    if singleton_probs:
        matches = apply_singleton_filter(matches, singleton_probs)
    
    return matches


def generate_outputs(predictions: Dict[str, List[str]],
                     candidates: Dict[str, List[str]],
                     singleton_probs: Optional[Dict[str, float]],
                     singleton_threshold: float,
                     output_dir: Path):
    """Generate both output files"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Post-process
    final_matches = postprocess_pipeline(predictions, candidates, singleton_probs, singleton_threshold)
    
    # Write outputs
    matching_path = output_dir / 'matching_results.tsv'
    candidate_path = output_dir / 'candidate_pairs.tsv'
    
    write_matching_results(final_matches, matching_path)
    write_candidate_pairs(candidates, candidate_path)
    
    return matching_path, candidate_path
