"""
Bi-encoder for Embedding-based Candidate Retrieval
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


class BiEncoderMatcher:
    """Bi-encoder for fast candidate retrieval using embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 512):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_id_map: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None
    
    def encode_batch(self, records: List[Dict], text_field: str = 'combined') -> np.ndarray:
        """Encode records to embeddings"""
        texts = [self._prepare_text(r, text_field) for r in records]
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def _prepare_text(self, record: Dict, field: str) -> str:
        """Prepare text for encoding"""
        if field == 'combined':
            name = record.get('name_clean', '')
            addr = record.get('addr_clean', '')
            return f"{name} [SEP] {addr}"
        return record.get(field, '')
    
    def build_index(self, records: List[Dict], text_field: str = 'combined') -> faiss.Index:
        """Build FAISS index from records"""
        print(f"Encoding {len(records)} records...")
        embeddings = self.encode_batch(records)
        self.embeddings = embeddings
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Use IVF-Flat for fast search
        dim = embeddings.shape[1]
        nlist = min(4096, max(1, len(records) // 100))
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        index = faiss.IndexIVFFlat(quantizer, dim, faiss.METRIC_INNER_PRODUCT)
        
        # Train and add
        index.train(embeddings)
        index.add(embeddings)
        
        self.faiss_index = index
        return index
    
    def search(self, query_records: List[Dict], k: int = 50, 
               text_field: str = 'combined') -> List[List[Tuple[str, float]]]:
        """Search for top-k candidates for each query"""
        if not self.faiss_index:
            raise ValueError("Index not built")
        
        query_embeddings = self.encode_batch(query_records, text_field)
        query_embeddings = query_embeddings.astype('float32')
        
        # FAISS expects (n_queries, dim)
        distances, indices = self.faiss_index.search(query_embeddings, k)
        
        results = []
        for i in range(len(query_records)):
            results.append([
                (self.faiss_id_map.get(int(idx), ''), float(dist))
                for dist, idx in zip(distances[i], indices[i])
                if idx >= 0
            ])
        
        return results
    
    def save(self, path: str):
        """Save FAISS index and metadata"""
        if self.faiss_index:
            faiss.write_index(self.faiss_index, f"{path}.faiss")
            np.save(f"{path}_embeddings.npy", self.embeddings)
            with open(f"{path}_idmap.pkl", 'wb') as f:
                pickle.dump(self.faiss_id_map, f)
    
    def load(self, path: str):
        """Load FAISS index and metadata"""
        self.faiss_index = faiss.read_index(f"{path}.faiss")
        self.embeddings = np.load(f"{path}_embeddings.npy")
        with open(f"{path}_idmap.pkl", 'rb') as f:
            self.faiss_id_map = pickle.load(f)
