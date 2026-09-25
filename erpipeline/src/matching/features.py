from typing import Dict, List, Set, Tuple, Any
"""
Pairwise Feature Extraction for Entity Matching
"""

import re
from typing import Dict, List, Set, Tuple
import numpy as np
import rapidfuzz.fuzz as fuzz
import rapidfuzz.distance as distance
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    """Extract pairwise features for entity matching"""
    
    def __init__(self, tfidf_ngram_range: Tuple[int, int] = (3, 3)):
        # TF-IDF on char n-grams
        self.tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=tfidf_ngram_range,
            lowercase=True,
            min_df=1,
            max_df=1.0,
        )
        self.fitted = False
        self._name_vocab: List[str] = []
    
    def fit(self, all_names: List[str]):
        """Fit TF-IDF on all names"""
        self._name_vocab = [n for n in all_names if n]
        if self._name_vocab:
            self.tfidf.fit(self._name_vocab)
            self.fitted = True
    
    def _safe_str(self, text: Any) -> str:
        return str(text) if text is not None else ""
    
    def _name_features(self, name1: str, name2: str) -> Dict[str, float]:
        """Extract name similarity features"""
        name1 = self._safe_str(name1).lower().strip()
        name2 = self._safe_str(name2).lower().strip()
        
        if not name1 or not name2:
            return self._empty_name_features()
        
        features = {}
        
        # rapidfuzz metrics
        features['fuzz_wratio'] = fuzz.WRatio(name1, name2) / 100.0
        features['fuzz_token_set_ratio'] = fuzz.token_set_ratio(name1, name2) / 100.0
        features['fuzz_token_sort_ratio'] = fuzz.token_sort_ratio(name1, name2) / 100.0
        features['fuzz_partial_ratio'] = fuzz.partial_ratio(name1, name2) / 100.0
        features['fuzz_qratio'] = fuzz.QRatio(name1, name2) / 100.0
        
        # Levenshtein normalized
        lev_dist = distance.Levenshtein.distance(name1, name2)
        max_len = max(len(name1), len(name2))
        features['levenshtein_norm'] = 1.0 - (lev_dist / max_len if max_len > 0 else 0)
        
        # Jaro-Winkler
        features['jaro_winkler'] = fuzz.WRatio(name1, name2) / 100.0  # WRatio includes JW
        
        # Jaccard on tokens
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())
        if tokens1 or tokens2:
            inter = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            features['token_jaccard'] = inter / union if union > 0 else 0.0
        else:
            features['token_jaccard'] = 0.0
        
        # First token match
        first1 = name1.split()[0] if name1.split() else ""
        first2 = name2.split()[0] if name2.split() else ""
        features['first_token_match'] = 1.0 if first1 and first1 == first2 else 0.0
        
        # TF-IDF Cosine
        features['tfidf_cosine'] = self._tfidf_cosine(name1, name2)
        
        return features
    
    def _tfidf_cosine(self, name1: str, name2: str) -> float:
        if not self.fitted or not name1 or not name2:
            return 0.0
        try:
            vecs = self.tfidf.transform([name1, name2])
            return float(cosine_similarity(vecs[0], vecs[1])[0, 0])
        except:
            return 0.0
    
    def _empty_name_features(self) -> Dict[str, float]:
        return {
            'fuzz_wratio': 0.0, 'fuzz_token_set_ratio': 0.0,
            'fuzz_token_sort_ratio': 0.0, 'fuzz_partial_ratio': 0.0,
            'fuzz_qratio': 0.0, 'levenshtein_norm': 0.0,
            'jaro_winkler': 0.0, 'token_jaccard': 0.0,
            'first_token_match': 0.0, 'tfidf_cosine': 0.0,
        }
    
    def _phonetic_features(self, name1: str, name2: str) -> Dict[str, float]:
        """Extract phonetic similarity features"""
        name1 = self._safe_str(name1).lower().strip()
        name2 = self._safe_str(name2).lower().strip()
        
        if not name1 or not name2:
            return {'metaphone_match': 0.0, 'soundex_match': 0.0, 'nysiis_match': 0.0}
        
        return {
            'metaphone_match': 1.0 if jellyfish.metaphone(name1) == jellyfish.metaphone(name2) else 0.0,
            'soundex_match': 1.0 if jellyfish.soundex(name1) == jellyfish.soundex(name2) else 0.0,
            'nysiis_match': 1.0 if jellyfish.nysiis(name1) == jellyfish.nysiis(name2) else 0.0,
        }
    
    def _address_features(self, addr1: str, addr2: str, 
                           pc1: List[str], pc2: List[str],
                           city1: str, city2: str,
                           state1: str, state2: str,
                           district1: str, district2: str,
                           hn1: str, hn2: str,
                           road1: str, road2: str,
                           unit1: str, unit2: str,
                           po_box1: str, po_box2: str) -> Dict[str, float]:
        """Extract address similarity features"""
        addr1 = addr1.lower().strip() if addr1 else ""
        addr2 = addr2.lower().strip() if addr2 else ""
        
        features = {}
        
        # Postal code exact match
        pc_set1 = set(pc1) if pc1 else set()
        pc_set2 = set(pc2) if pc2 else set()
        features['pin_exact'] = 1.0 if pc_set1 & pc_set2 else 0.0
        
        # Admin exact matches
        features['city_exact'] = 1.0 if city1 and city2 and city1.lower() == city2.lower() else 0.0
        features['state_exact'] = 1.0 if state1 and state2 and state1.lower() == state2.lower() else 0.0
        features['district_exact'] = 1.0 if district1 and district2 and district1.lower() == district2.lower() else 0.0
        
        # House number match
        features['house_number_match'] = 1.0 if hn1 and hn2 and hn1 == hn2 else 0.0
        
        # Road match
        features['road_match'] = 1.0 if road1 and road2 and road1 == road2 else 0.0
        
        # Unit match
        features['unit_match'] = 1.0 if unit1 and unit2 and unit1 == unit2 else 0.0
        
        # PO Box match
        features['po_box_match'] = 1.0 if po_box1 and po_box2 and po_box1 == po_box2 else 0.0
        
        # Token Jaccard on full address
        if addr1 and addr2:
            tokens1 = set(addr1.split())
            tokens2 = set(addr2.split())
            inter = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            features['addr_token_jaccard'] = inter / union if union > 0 else 0.0
        else:
            features['addr_token_jaccard'] = 0.0
        
        # Numeric token Jaccard
        nums1 = set(re.findall(r'\b\d+[A-Z/-]*\b', addr1))
        nums2 = set(re.findall(r'\b\d+[A-Z/-]*\b', addr2))
        if nums1 or nums2:
            features['numeric_jaccard'] = len(nums1 & nums2) / len(nums1 | nums2)
        else:
            features['numeric_jaccard'] = 0.0
        
        return features
    
    def _cross_field_features(self, name1: str, addr1: str, name2: str, addr2: str) -> Dict[str, float]:
        """Cross-field features: name tokens in address, address tokens in name"""
        name1 = self._safe_str(name1).lower()
        addr1 = self._safe_str(addr1).lower()
        name2 = self._safe_str(name2).lower()
        addr2 = self._safe_str(addr2).lower()
        
        if not name1 or not name2:
            return {'name1_in_addr2': 0.0, 'name2_in_addr1': 0.0}
        
        name1_tokens = set(name1.split())
        name2_tokens = set(name2.split())
        addr1_tokens = set(addr1.split())
        addr2_tokens = set(addr2.split())
        
        features = {}
        
        # Name tokens in other's address
        if name1_tokens:
            features['name1_in_addr2'] = len(name1_tokens & addr2_tokens) / len(name1_tokens)
        else:
            features['name1_in_addr2'] = 0.0
        
        if name2_tokens:
            features['name2_in_addr1'] = len(name2_tokens & addr1_tokens) / len(name2_tokens)
        else:
            features['name2_in_addr1'] = 0.0
        
        return features
    
    def extract(self, s1: Dict, s2: Dict) -> Dict[str, float]:
        """Extract all features for a pair"""
        features = {}
        
        # Name features
        features.update(self._name_features(s1.get('name_clean', ''), s2.get('name_clean', '')))
        
        # Phonetic
        features.update(self._phonetic_features(
            s1.get('name_clean', ''), s2.get('name_clean', '')
        ))
        
        # Address
        features.update(self._address_features(
            s1.get('addr_clean', ''), s2.get('addr_clean', ''),
            s1.get('postal_codes', []), s2.get('postal_codes', []),
            s1.get('city', ''), s2.get('city', ''),
            s1.get('state', ''), s2.get('state', ''),
            s1.get('district', ''), s2.get('district', ''),
            s1.get('house_number', ''), s2.get('house_number', ''),
            s1.get('road', ''), s2.get('road', ''),
            s1.get('unit', ''), s2.get('unit', ''),
            s1.get('po_box', ''), s2.get('po_box', ''),
        ))
        
        # Cross-field
        features.update(self._cross_field_features(
            s1.get('name_clean', ''), s1.get('addr_clean', ''),
            s2.get('name_clean', ''), s2.get('addr_clean', '')
        ))
        
        # Country match
        features['country_match'] = 1.0 if s1.get('country') == s2.get('country') else 0.0
        
        return features
    
    def extract_batch(self, pairs: List[Tuple[Dict, Dict]]) -> List[Dict[str, float]]:
        """Extract features for multiple pairs"""
        return [self.extract(s1, s2) for s1, s2 in pairs]
