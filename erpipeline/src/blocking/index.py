"""
Blocking Index - Library-First Implementation
Uses: datasketch (MinHash LSH), faiss (ANN), exact hash maps
"""

import pickle
from collections import defaultdict
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import polars as pl
from datasketch import MinHash, MinHashLSH
import faiss
from tqdm import tqdm

from ..config import get_config


@dataclass
class BlockingConfig:
    # Name-based keys
    name_ngram_size: int = 3
    name_min_overlap: int = 2
    metaphone_enabled: bool = True
    
    # Address-based keys (NEW)
    addr_ngram_size: int = 3
    addr_min_overlap: int = 2
    addr_metaphone_enabled: bool = True
    addr_minhash_enabled: bool = True
    addr_minhash_threshold: float = 0.55
    addr_minhash_perm: int = 128
    addr_metaphone_sorted_enabled: bool = True
    
    # Exact match keys
    postal_exact: bool = True
    city_state_exact: bool = True
    house_number_exact: bool = True
    
    # Name-based approximate
    metaphone_enabled: bool = True
    minhash_enabled: bool = True
    minhash_threshold: float = 0.55
    minhash_perm: int = 128
    
    # Embedding-based
    biencoder_enabled: bool = True
    biencoder_top_k: int = 100
    
    # Candidate cap
    max_candidates_per_s1: int = 200


class BlockingIndex:
    def __init__(self, config: BlockingConfig = None):
        self.config = config or BlockingConfig()
        
        # Exact match indexes (hash maps)
        self.pin_index: Dict[str, Set[str]] = defaultdict(set)
        self.city_state_index: Dict[str, Set[str]] = defaultdict(set)
        self.house_number_index: Dict[str, Set[str]] = defaultdict(set)
        self.metaphone_index: Dict[str, Set[str]] = defaultdict(set)
        self.metaphone_sorted_index: Dict[str, Set[str]] = defaultdict(set)
        self.soundex_index: Dict[str, Set[str]] = defaultdict(set)
        self.nysiis_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Address-based indexes (NEW)
        self.addr_ngram_index: Dict[str, Set[str]] = defaultdict(set)
        self.addr_metaphone_index: Dict[str, Set[str]] = defaultdict(set)
        self.addr_soundex_index: Dict[str, Set[str]] = defaultdict(set)
        self.addr_nysiis_index: Dict[str, Set[str]] = defaultdict(set)
        self.addr_metaphone_sorted_index: Dict[str, Set[str]] = defaultdict(set)
        self.addr_minhash_lsh: Optional[MinHashLSH] = None
        
        # Approximate indexes
        self.minhash_lsh: Optional[MinHashLSH] = None
        if self.config.minhash_enabled:
            self.minhash_lsh = MinHashLSH(
                threshold=self.config.minhash_threshold,
                num_perm=self.config.minhash_perm
            )
        
        # Address-based MinHash LSH (NEW)
        if self.config.addr_minhash_enabled:
            self.addr_minhash_lsh = MinHashLSH(
                threshold=self.config.addr_minhash_threshold,
                num_perm=self.config.addr_minhash_perm
            )
        
        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_id_map: Dict[int, str] = {}  # faiss_idx -> entity_id
        
        # Entity data storage
        self.entity_data: Dict[str, Dict] = {}
    
    def _minhash_from_ngrams(self, ngrams: List[str]) -> MinHash:
        """Create MinHash from name n-grams"""
        m = MinHash(num_perm=self.config.minhash_perm)
        for ng in ngrams:
            m.update(ng.encode('utf-8'))
        return m
    
    def add_entity(self, record: Dict[str, Any]):
        """Add entity to all blocking indexes"""
        eid = record['entity_id']
        self.entity_data[eid] = record
        
        # Exact indexes
        if self.config.postal_exact:
            for pc in record.get('postal_codes', []):
                if pc:
                    self.pin_index[f'PIN:{pc}'].add(eid)
        
        if self.config.city_state_exact:
            city = record.get('city')
            state = record.get('state')
            if city and state:
                key = f'CITY_STATE:{city}:{state}'
                self.city_state_index[key].add(eid)
        
        if self.config.house_number_exact:
            hn = record.get('house_number_norm') or record.get('house_number')
            if hn:
                self.house_number_index[f'HOUSE:{hn}'].add(eid)
        
        if self.config.metaphone_enabled:
            meta = record.get('metaphone')
            if meta:
                self.metaphone_index[f'META:{meta}'].add(eid)

            meta_sorted = record.get('metaphone_sorted')
            if meta_sorted:
                self.metaphone_sorted_index[f'META_SORTED:{meta_sorted}'].add(eid)
            
            soundex = record.get('soundex')
            if soundex:
                self.soundex_index[f'SOUNDEX:{soundex}'].add(eid)
            
            nysiis = record.get('nysiis')
            if nysiis:
                self.nysiis_index[f'NYSIIS:{nysiis}'].add(eid)
        
        # MinHash LSH (name-based)
        if self.config.minhash_enabled:
            ngrams = record.get('name_ngrams', [])
            if ngrams:
                mh = self._minhash_from_ngrams(ngrams)
                self.minhash_lsh.insert(eid, mh)
        
        # Address-based indexes (NEW)
        if self.config.addr_metaphone_enabled:
            meta = record.get('addr_metaphone')
            if meta:
                self.addr_metaphone_index[f'ADDR_META:{meta}'].add(eid)
            
            meta_sorted = record.get('addr_metaphone_sorted')
            if meta_sorted:
                self.addr_metaphone_sorted_index[f'ADDR_META_SORTED:{meta_sorted}'].add(eid)
            
            soundex = record.get('addr_soundex')
            if soundex:
                self.addr_soundex_index[f'ADDR_SOUNDEX:{soundex}'].add(eid)
            
            nysiis = record.get('addr_nysiis')
            if nysiis:
                self.addr_nysiis_index[f'ADDR_NYSIIS:{nysiis}'].add(eid)
            
            meta_sorted = record.get('addr_metaphone_sorted')
            if meta_sorted:
                self.addr_metaphone_sorted_index[f'ADDR_META_SORTED:{meta_sorted}'].add(eid)
        
        # Address MinHash LSH
        if self.config.addr_minhash_enabled:
            addr_ngrams = record.get('addr_ngrams', [])
            if addr_ngrams:
                mh = self._minhash_from_ngrams(addr_ngrams)
                self.addr_minhash_lsh.insert(eid, mh)
    
    def build_faiss_index(self, embeddings: np.ndarray, entity_ids: List[str]):
        """Build FAISS index from bi-encoder embeddings"""
        if not self.config.biencoder_enabled:
            return
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Use IVF-Flat for fast search
        dim = embeddings.shape[1]
        nlist = min(4096, max(1, len(entity_ids) // 100))
        quantizer = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train and add
        self.faiss_index.train(embeddings)
        self.faiss_index.add(embeddings)
        
        # Store id mapping
        self.faiss_id_map = {i: eid for i, eid in enumerate(entity_ids)}
    
    def build_from_dataframe(self, df: pl.DataFrame):
        """Build all indexes from processed DataFrame"""
        print(f"Building blocking indexes for {len(df)} entities...")
        
        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            self.add_entity(row)
        
        print(f"  PIN index: {len(self.pin_index)} keys")
        print(f"  City-State index: {len(self.city_state_index)} keys")
        print(f"  House number index: {len(self.house_number_index)} keys")
        print(f"  Metaphone index: {len(self.metaphone_index)} keys")
        
        if self.config.minhash_enabled and self.minhash_lsh:
            print(f"  MinHash LSH (name): built")
        if self.config.addr_minhash_enabled and self.addr_minhash_lsh:
            print(f"  MinHash LSH (address): built")
        print(f"  Address metaphone index: {len(self.addr_metaphone_index)} keys")
    
    def query_exact(self, record: Dict[str, Any]) -> Set[str]:
        """Query exact match indexes"""
        candidates = set()
        
        if self.config.postal_exact:
            for pc in record.get('postal_codes', []):
                if pc:
                    candidates.update(self.pin_index.get(f'PIN:{pc}', set()))
        
        if self.config.city_state_exact:
            city = record.get('city')
            state = record.get('state')
            if city and state:
                candidates.update(self.city_state_index.get(f'CITY_STATE:{city}:{state}', set()))
        
        if self.config.house_number_exact:
            hn = record.get('house_number_norm') or record.get('house_number')
            if hn:
                candidates.update(self.house_number_index.get(f'HOUSE:{hn}', set()))
        
        if self.config.metaphone_enabled:
            meta = record.get('metaphone')
            if meta:
                candidates.update(self.metaphone_index.get(f'META:{meta}', set()))

            meta_sorted = record.get('metaphone_sorted')
            if meta_sorted:
                candidates.update(self.metaphone_sorted_index.get(
                    f'META_SORTED:{meta_sorted}', set()))
            
            soundex = record.get('soundex')
            if soundex:
                candidates.update(self.soundex_index.get(f'SOUNDEX:{soundex}', set()))
            
            nysiis = record.get('nysiis')
            if nysiis:
                candidates.update(self.nysiis_index.get(f'NYSIIS:{nysiis}', set()))

        # Address exact matches (mirrors name matching on addr_clean)
        if self.config.addr_metaphone_enabled:
            addr_meta = record.get('addr_metaphone')
            if addr_meta:
                candidates.update(self.addr_metaphone_index.get(f'ADDR_META:{addr_meta}', set()))

            addr_meta_sorted = record.get('addr_metaphone_sorted')
            if addr_meta_sorted:
                candidates.update(self.addr_metaphone_sorted_index.get(
                    f'ADDR_META_SORTED:{addr_meta_sorted}', set()))

            addr_soundex = record.get('addr_soundex')
            if addr_soundex:
                candidates.update(self.addr_soundex_index.get(f'ADDR_SOUNDEX:{addr_soundex}', set()))

            addr_nysiis = record.get('addr_nysiis')
            if addr_nysiis:
                candidates.update(self.addr_nysiis_index.get(f'ADDR_NYSIIS:{addr_nysiis}', set()))

        return candidates
    
    def query_minhash(self, record: Dict[str, Any]) -> Set[str]:
        """Query MinHash LSH for approximate name similarity"""
        if not self.config.minhash_enabled or not self.minhash_lsh:
            return set()
        
        ngrams = record.get('name_ngrams', [])
        if not ngrams:
            return set()
        
        mh = self._minhash_from_ngrams(ngrams)
        return set(self.minhash_lsh.query(mh))

    def query_addr_minhash(self, record: Dict[str, Any]) -> Set[str]:
        """Query address MinHash LSH for approximate address similarity"""
        if not self.config.addr_minhash_enabled or not self.addr_minhash_lsh:
            return set()

        ngrams = record.get('addr_ngrams', [])
        if not ngrams:
            return set()

        mh = self._minhash_from_ngrams(ngrams)
        return set(self.addr_minhash_lsh.query(mh))

    def query_faiss(self, query_embedding: np.ndarray, k: int = None) -> List[str]:
        """Query FAISS index for embedding similarity"""
        if not self.config.biencoder_enabled or not self.faiss_index:
            return []
        
        k = k or self.config.biencoder_top_k
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        return [self.faiss_id_map.get(int(idx)) for idx in indices[0] if idx >= 0]
    
    def get_candidates(self, record: Dict[str, Any], 
                       query_embedding: np.ndarray = None) -> Set[str]:
        """Get all candidates for a query record.

        Country is a 100%-purity blocking key (zero cross-country matches in
        ground truth), so candidates are restricted to the query's country.
        """
        candidates = set()

        # Exact matches
        candidates.update(self.query_exact(record))

        # MinHash LSH (name)
        candidates.update(self.query_minhash(record))

        # MinHash LSH (address)
        candidates.update(self.query_addr_minhash(record))

        # FAISS (bi-encoder)
        if query_embedding is not None:
            candidates.update(self.query_faiss(query_embedding))

        # Country partition: drop any cross-country candidates.
        query_country = record.get('country')
        if query_country:
            candidates = {eid for eid in candidates
                          if self.entity_data.get(eid, {}).get('country') in (None, '', query_country)}

        return candidates
    
    def get_candidates_split(self, record: Dict[str, Any],
                               query_embedding: np.ndarray = None
                               ) -> Tuple[Set[str], Set[str]]:
        """Get candidates split into (exact, fuzzy) sets.

        Exact = PIN / city-state / house-number / phonetic indexes.
        Fuzzy = name and address MinHash LSH (+ FAISS).
        """
        exact = self.query_exact(record)
        fuzzy = self.query_minhash(record)
        fuzzy.update(self.query_addr_minhash(record))
        if query_embedding is not None:
            fuzzy.update(self.query_faiss(query_embedding))
        fuzzy -= exact
        return exact, fuzzy

    def get_candidates_for_s1(self, s1_df: pl.DataFrame,
                              s1_embeddings: np.ndarray = None) -> pl.DataFrame:
        """Generate candidates for all S1 entities.

        Prioritized cap: keep every exact/phonetic match, sample only the
        fuzzy overflow. A random cap over the union can drop true matches
        when a loose fuzzy key floods the pool.
        """
        import random
        results = []

        for i, row in enumerate(s1_df.iter_rows(named=True)):
            s1_id = row['entity_id']

            emb = s1_embeddings[i] if s1_embeddings is not None else None
            exact, fuzzy = self.get_candidates_split(row, emb)
            candidates = set(exact)
            room = self.config.max_candidates_per_s1 - len(candidates)
            if room > 0 and fuzzy:
                if len(fuzzy) > room:
                    candidates.update(random.sample(sorted(fuzzy), room))
                else:
                    candidates.update(fuzzy)

            results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': sorted(candidates),
                'num_candidates': len(candidates)
            })

        return pl.DataFrame(results)
    
    def save(self, path: str):
        """Save blocking index to disk"""
        data = {
            'pin_index': dict(self.pin_index),
            'city_state_index': dict(self.city_state_index),
            'house_number_index': dict(self.house_number_index),
            'metaphone_index': dict(self.metaphone_index),
            'metaphone_sorted_index': dict(self.metaphone_sorted_index),
            'soundex_index': dict(self.soundex_index),
            'nysiis_index': dict(self.nysiis_index),
            'addr_metaphone_index': dict(self.addr_metaphone_index),
            'addr_metaphone_sorted_index': dict(self.addr_metaphone_sorted_index),
            'addr_soundex_index': dict(self.addr_soundex_index),
            'addr_nysiis_index': dict(self.addr_nysiis_index),
            'entity_data': self.entity_data,
            'config': self.config.__dict__,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately
        if self.faiss_index:
            faiss.write_index(self.faiss_index, path + '.faiss')
            with open(path + '.faiss_map.pkl', 'wb') as f:
                pickle.dump(self.faiss_id_map, f)
        
        print(f"Saved blocking index to {path}")
    
    @classmethod
    def load(cls, path: str, config: BlockingConfig = None) -> 'BlockingIndex':
        """Load blocking index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(config or BlockingConfig(**data['config']))
        obj.pin_index = defaultdict(set, data['pin_index'])
        obj.city_state_index = defaultdict(set, data['city_state_index'])
        obj.house_number_index = defaultdict(set, data['house_number_index'])
        obj.metaphone_index = defaultdict(set, data['metaphone_index'])
        obj.metaphone_sorted_index = defaultdict(
            set, data.get('metaphone_sorted_index', {}))
        obj.soundex_index = defaultdict(set, data['soundex_index'])
        obj.nysiis_index = defaultdict(set, data['nysiis_index'])
        obj.addr_metaphone_index = defaultdict(set, data.get('addr_metaphone_index', {}))
        obj.addr_metaphone_sorted_index = defaultdict(
            set, data.get('addr_metaphone_sorted_index', {}))
        obj.addr_soundex_index = defaultdict(set, data.get('addr_soundex_index', {}))
        obj.addr_nysiis_index = defaultdict(set, data.get('addr_nysiis_index', {}))
        obj.entity_data = data['entity_data']
        
        # Load FAISS
        try:
            obj.faiss_index = faiss.read_index(path + '.faiss')
            with open(path + '.faiss_map.pkl', 'rb') as f:
                obj.faiss_id_map = pickle.load(f)
        except Exception:
            pass
        
        # Rebuild MinHash LSH
        if obj.config.minhash_enabled:
            obj.minhash_lsh = MinHashLSH(
                threshold=obj.config.minhash_threshold,
                num_perm=obj.config.minhash_perm
            )
            # Note: MinHash LSH needs re-insertion of entities
            # For now, we skip this and rely on exact + FAISS
        
        print(f"Loaded blocking index from {path}")
        return obj


def _format_entity_info(entity_data: Dict, label: str) -> str:
    """Format entity info for detailed output"""
    lines = [f"  {label}:"]
    lines.append(f"    ID: {entity_data.get('entity_id', 'N/A')}")
    lines.append(f"    Country: {entity_data.get('country', 'N/A')}")
    lines.append(f"    Raw Name: {entity_data.get('name_raw', 'N/A')}")
    lines.append(f"    Clean Name: {entity_data.get('name_clean', 'N/A')}")
    lines.append(f"    Raw Addr: {entity_data.get('addr_raw', 'N/A')}")
    lines.append(f"    Clean Addr: {entity_data.get('addr_clean', 'N/A')}")
    lines.append(f"    PINs: {entity_data.get('postal_codes', [])}")
    lines.append(f"    City: {entity_data.get('city', 'N/A')}")
    lines.append(f"    State: {entity_data.get('state', 'N/A')}")
    lines.append(f"    District: {entity_data.get('district', 'N/A')}")
    lines.append(f"    House#: {entity_data.get('house_number', 'N/A')}")
    lines.append(f"    Road: {entity_data.get('road', 'N/A')}")
    lines.append(f"    Unit: {entity_data.get('unit', 'N/A')}")
    lines.append(f"    PO Box: {entity_data.get('po_box', 'N/A')}")
    lines.append(f"    Metaphone: {entity_data.get('metaphone', 'N/A')}")
    lines.append(f"    Soundex: {entity_data.get('soundex', 'N/A')}")
    lines.append(f"    NYSIIS: {entity_data.get('nysiis', 'N/A')}")
    lines.append(f"    N-grams: {entity_data.get('name_ngrams', [])[:10]}...")
    return "\n".join(lines)


def evaluate_blocking_recall(index: BlockingIndex, 
                             s1_df: pl.DataFrame,
                             gt_path: str,
                             s23_entity_data: Dict) -> Dict[str, float]:
    """Evaluate blocking recall on ground truth with detailed per-entity output"""
    print(f"  Loading ground truth from: {gt_path}")
    
    # Load ground truth
    gt_pairs = set()
    with open(gt_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0]
                for match_id in parts[1].split(','):
                    gt_pairs.add((s1_id, match_id))
    
    print(f"  Loaded {len(gt_pairs)} ground truth pairs")
    
    # Evaluate per S1 entity
    total_gt = 0
    recalled_gt = 0
    per_entity_recall = []
    candidate_counts = []
    
    print(f"  Evaluating {len(s1_df)} S1 entities...")
    for idx, row in enumerate(s1_df.iter_rows(named=True)):
        s1_id = row['entity_id']
        
        # Get ground truth matches for this S1
        s1_gt = {m for s1, m in gt_pairs if s1 == s1_id}
        if not s1_gt:
            print(f"\n{'='*80}")
            print(f"[{idx+1:3d}] {s1_id}: NO GROUND TRUTH (singleton)")
            print(_format_entity_info(row, "S1 Entity (NO GT)"))
            continue
        
        # Get candidates
        candidates = index.get_candidates(row)
        
        # Count recalled
        recalled = len(s1_gt & candidates)
        total_gt += len(s1_gt)
        recalled_gt += recalled
        
        entity_recall = recalled / len(s1_gt) if s1_gt else 1.0
        per_entity_recall.append(entity_recall)
        candidate_counts.append(len(candidates))
        
        missed = s1_gt - candidates
        matched = s1_gt & candidates
        
        # Detailed output for EVERY entity
        print(f"\n{'='*80}")
        print(f"[{idx+1:3d}] {s1_id}: GT={len(s1_gt)} Candidates={len(candidates)} Recall={entity_recall:.2f}")
        
        # Print S1 entity details
        print(_format_entity_info(row, "S1 Entity"))
        
        # Print matched candidates with details
        if matched:
            print(f"  Matched ({len(matched)}):")
            for mid in sorted(matched):
                if mid in s23_entity_data:
                    print(_format_entity_info(s23_entity_data[mid], f"  ✓ MATCH: {mid}"))
                else:
                    print(f"    ✓ MATCH: {mid} (data not in index)")
        
        # Print missed candidates with details
        if missed:
            print(f"  Missed ({len(missed)}):")
            for mid in sorted(missed):
                if mid in s23_entity_data:
                    print(_format_entity_info(s23_entity_data[mid], f"  ✗ MISSED: {mid}"))
                else:
                    print(f"    ✗ MISSED: {mid} (data not in index)")
        
        # Print candidate list
        if candidates:
            print(f"  All Candidates ({len(candidates)}): {sorted(candidates)}")
        
        # Break early for testing (remove this for full eval)
        if idx >= 49:  # Test first 50 entities
            print(f"\n  [TEST MODE] Stopping after 50 entities")
            break
    
    overall_recall = recalled_gt / total_gt if total_gt > 0 else 0.0
    macro_recall = np.mean(per_entity_recall) if per_entity_recall else 0.0
    avg_candidates = np.mean(candidate_counts) if candidate_counts else 0
    
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Total GT pairs: {total_gt}, Recalled: {recalled_gt}")
    print(f"  Overall recall: {overall_recall:.4f}")
    print(f"  Macro recall: {macro_recall:.4f}")
    print(f"  Avg candidates per S1: {avg_candidates:.1f}")
    
    return {
        'overall_recall': overall_recall,
        'macro_recall': macro_recall,
        'total_gt_pairs': total_gt,
        'recalled_pairs': recalled_gt,
        'avg_candidates_per_s1': avg_candidates,
    }


if __name__ == '__main__':
    # Quick test
    config = BlockingConfig()
    index = BlockingIndex(config)
    print("BlockingIndex created successfully")
