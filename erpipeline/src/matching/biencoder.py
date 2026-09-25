"""
Bi-encoder for Embedding-based Candidate Retrieval.

Country-partitioned FAISS indexes: one index per country, so retrieval
never returns cross-country candidates. Uses exact IndexFlatIP for small
pools and IVF-Flat for large pools.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class BiEncoderMatcher:
    """Bi-encoder for fast candidate retrieval using embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 512):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        # country -> {'index': faiss.Index, 'ids': List[str]}
        self.indexes: Dict[str, Dict] = {}

    def _prepare_text(self, record: Dict) -> str:
        """Prepare text for encoding: cleaned name + address."""
        name = record.get('name_clean', '') or ''
        addr = record.get('addr_clean', '') or ''
        return f"{name} [SEP] {addr}".strip()

    def encode_records(self, records: List[Dict]) -> np.ndarray:
        """Encode records to L2-normalized embeddings."""
        texts = [self._prepare_text(r) for r in records]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype('float32')

    def build_index(self, records: List[Dict]) -> Dict[str, int]:
        """Build one FAISS index per country. Returns {country: count}."""
        by_country: Dict[str, List[Dict]] = {}
        for record in records:
            country = record.get('country', '') or 'UNKNOWN'
            by_country.setdefault(country, []).append(record)

        counts = {}
        for country, country_records in by_country.items():
            print(f"Encoding {len(country_records)} records for {country}...")
            embeddings = self.encode_records(country_records)
            dim = embeddings.shape[1]

            # Exact search for small pools, IVF for large pools.
            if len(country_records) < 5000:
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
            else:
                nlist = min(4096, max(1, len(country_records) // 100))
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                           faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings)
                index.add(embeddings)

            self.indexes[country] = {
                'index': index,
                'ids': [r.get('entity_id', '') for r in country_records],
            }
            counts[country] = len(country_records)
            print(f"  Built {country} index: {len(country_records)} vectors, dim={dim}")
        return counts

    def search(self, query_records: List[Dict], k: int = 50
               ) -> List[List[Tuple[str, float]]]:
        """Search top-k candidates per query, restricted to query's country."""
        results: List[List[Tuple[str, float]]] = []
        # Group queries by country for batched search.
        by_country: Dict[str, List[int]] = {}
        for i, record in enumerate(query_records):
            country = record.get('country', '') or 'UNKNOWN'
            by_country.setdefault(country, []).append(i)
            results.append([])

        for country, positions in by_country.items():
            entry = self.indexes.get(country)
            if not entry:
                continue
            batch = [query_records[i] for i in positions]
            query_embeddings = self.encode_records(batch)
            distances, indices = entry['index'].search(query_embeddings, k)
            for pos, dists, idxs in zip(positions, distances, indices):
                results[pos] = [
                    (entry['ids'][int(idx)], float(dist))
                    for dist, idx in zip(dists, idxs)
                    if 0 <= int(idx) < len(entry['ids'])
                ]
        return results

    def retrieve_candidates(self, query_records: List[Dict], k: int = 50,
                            min_score: float = 0.0) -> Dict[str, List[str]]:
        """Return {entity_id: [candidate_ids]} for query records."""
        hits = self.search(query_records, k=k)
        out: Dict[str, List[str]] = {}
        for record, scored in zip(query_records, hits):
            out[record.get('entity_id', '')] = [
                eid for eid, score in scored if eid and score >= min_score
            ]
        return out

    def save(self, path: str):
        """Save FAISS indexes and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {}
        for country, entry in self.indexes.items():
            faiss.write_index(entry['index'], str(path / f'index_{country}.faiss'))
            meta[country] = entry['ids']
        with open(path / 'idmap.pkl', 'wb') as f:
            pickle.dump(meta, f)
        print(f"Saved bi-encoder indexes to {path}")

    def load(self, path: str):
        """Load FAISS indexes and metadata."""
        path = Path(path)
        with open(path / 'idmap.pkl', 'rb') as f:
            meta = pickle.load(f)
        self.indexes = {}
        for country, ids in meta.items():
            self.indexes[country] = {
                'index': faiss.read_index(str(path / f'index_{country}.faiss')),
                'ids': ids,
            }
        print(f"Loaded bi-encoder indexes from {path}")
