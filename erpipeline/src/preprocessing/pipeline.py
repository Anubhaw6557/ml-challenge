"""
Preprocessing Pipeline - Library-First with Optional pypostal
Uses: pypostal-multiarch (if available), aksharamukha, unidecode, 
cleanco, pycountry, geotext, jellyfish, rapidfuzz
"""

import unicodedata
import re
import importlib
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass

import polars as pl
from tqdm import tqdm

import pycountry
from geotext import GeoText
from aksharamukha import transliterate
from unidecode import unidecode
import jellyfish
from cleanco import basename

from ..config import get_config


def _import_pypostal() -> Tuple[bool, Any, Any]:
    """Try to import pypostal dynamically, return (has_postal, parse_func, expand_func)"""
    try:
        postal_parser = importlib.import_module('postal.parser')
        postal_expand = importlib.import_module('postal.expand')
        parse_address = postal_parser.parse_address
        expand_address = postal_expand.expand_address
        
        # Test if libpostal data directory is available
        try:
            parse_address("test")
            return True, parse_address, expand_address
        except Exception:
            return False, None, None
    except ImportError:
        return False, None, None
    except Exception:
        return False, None, None


@dataclass
class ProcessedRecord:
    entity_id: str
    country: str
    name_raw: str
    addr_raw: str
    name_clean: str
    addr_clean: str
    postal_codes: List[str]
    city: Optional[str]
    state: Optional[str]
    district: Optional[str]
    house_number: Optional[str]
    road: Optional[str]
    unit: Optional[str]
    po_box: Optional[str]
    name_ngrams: List[str]
    metaphone: str
    soundex: str
    nysiis: str
    addr_ngrams: List[str]
    addr_metaphone: str
    addr_soundex: str
    addr_nysiis: str


def _safe_str(text: Any) -> str:
    """Safely convert to string, handling None"""
    if text is None:
        return ""
    return str(text)


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.countries = self.config['preprocessing']['countries']
        self.ngram_size = self.config['blocking']['name_ngram_size']
        self.has_postal, self._parse_address_libpostal_fn, self._expand_address_libpostal_fn = _import_pypostal()
        
        if self.has_postal:
            print("✅ pypostal available - using libpostal for address parsing")
        else:
            print("⚠️ pypostal not available - using regex fallback for address parsing")
        
        # Pre-compile regex patterns (fallback)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for address parsing (fallback)"""
        # Postal codes by country
        self.postal_patterns = {
            'India': re.compile(r'\b\d{6}\b'),
            'US': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            'France': re.compile(r'\b\d{5}\b'),
        }
        
        # House/building number patterns
        self.house_number_pattern = re.compile(
            r'\b(?:h\.?no\.?|house\s+no\.?|door\s+no\.?|plot\s+no\.?|building\s+no\.?|'
            r'flat\s+no\.?|unit\s+no\.?)\.?\s*[:\-]?\s*([A-Z0-9\-\/]+)',
            re.IGNORECASE
        )
        
        # Street/road patterns
        self.street_pattern = re.compile(
            r'\b(?:road|street|avenue|drive|lane|boulevard|circle|court|place|'
            r'highway|parkway|rd|st|ave|dr|ln|blvd|cir|ct|pl|hwy|pkwy)\.?\b',
            re.IGNORECASE
        )
        
        # Unit/floor patterns
        self.unit_pattern = re.compile(
            r'\b(?:unit|flat|apartment|apt|suite|ste|floor|fl|level)\.?\s*[:\-]?\s*([A-Z0-9\-\/]+)',
            re.IGNORECASE
        )
        
        # PO Box
        self.po_box_pattern = re.compile(
            r'\b(?:p\.?o\.?\s*box|post\s*office\s*box)\.?\s*[:\-]?\s*(\d+)',
            re.IGNORECASE
        )
    
    def _safe_str(self, text: Any) -> str:
        """Safely convert to string, handling None"""
        if text is None:
            return ""
        return str(text)
    
    def _strip_accents(self, text: str) -> str:
        """Remove combining marks (accent noise in US/India)"""
        return ''.join(c for c in unicodedata.normalize('NFD', text) 
                       if unicodedata.category(c) != 'Mn')
    
    def _normalize_french(self, text: str) -> str:
        """Normalize French accents to ASCII"""
        return unidecode(text)
    
    # Mapping from Unicode script names (unicodedata) to aksharamukha
    # script names. Covers all major Indian scripts, not just Hindi/Tamil.
    INDIC_SCRIPT_MAP = {
        'DEVANAGARI': 'Devanagari',   # Hindi, Marathi, Sanskrit, Nepali
        'BENGALI': 'Bengali',         # Bengali, Assamese
        'GURMUKHI': 'Gurmukhi',       # Punjabi
        'GUJARATI': 'Gujarati',       # Gujarati
        'ORIYA': 'Oriya',             # Odia
        'TAMIL': 'Tamil',             # Tamil
        'TELUGU': 'Telugu',           # Telugu
        'KANNADA': 'Kannada',         # Kannada
        'MALAYALAM': 'Malayalam',     # Malayalam
        'SINHALA': 'Sinhala',         # Sinhala
        'TIBETAN': 'Tibetan',         # Ladakhi, Sikkimese
    }

    def _detect_script(self, text: str) -> str:
        """Detect the primary non-Latin script in text.

        Returns an aksharamukha script name, or 'Latin' if no Indic
        script is found.
        """
        scripts = set()
        for c in text:
            try:
                name = unicodedata.name(c)
            except ValueError:
                continue
            for script_key in self.INDIC_SCRIPT_MAP:
                if script_key in name:
                    scripts.add(script_key)
                    break
        # Prefer any Indic script over Latin.
        for script_key in self.INDIC_SCRIPT_MAP:
            if script_key in scripts:
                return self.INDIC_SCRIPT_MAP[script_key]
        return 'Latin'

    def _transliterate_indic(self, text: str) -> str:
        """Transliterate any Indic script to plain ASCII Latin.

        Must run BEFORE accent stripping: NFD-based accent removal would
        destroy Indic vowel signs (matras). ISO output is further normalized
        with unidecode so downstream char n-grams and phonetic encoders see
        a single ASCII alphabet.
        """
        script = self._detect_script(text)
        if script == 'Latin':
            return text
        try:
            text = transliterate.process(script, 'ISO', text)
        except Exception:
            return text
        return unidecode(text)
    
    def _normalize_name(self, name: str, country: str) -> str:
        """Normalize business name using libraries.

        Order: transliterate first, then normalize. NFD-based accent
        stripping would destroy Indic vowel signs, so any Indic script must
        be converted to Latin before accent handling.
        """
        name = self._safe_str(name)
        name = unicodedata.normalize('NFKC', name)

        # 1. Transliterate any Indic script to ASCII Latin first.
        name = self._transliterate_indic(name)

        # 2. Normalize accents / noise on the (now Latin) text.
        if country == 'US' or country == 'India':
            name = self._strip_accents(name)
        elif country == 'France':
            name = self._normalize_french(name)

        # Legal suffix normalization via cleanco
        name = basename(name)

        return name.lower().strip()

    def _normalize_address(self, addr: str, country: str) -> str:
        addr = self._safe_str(addr)
        addr = unicodedata.normalize('NFKC', addr)

        if addr == "":
            return ""

        # 1. Transliterate any Indic script to ASCII Latin first.
        addr = self._transliterate_indic(addr)

        # 2. Normalize accents / noise on the (now Latin) text.
        if country == 'US' or country == 'India':
            addr = self._strip_accents(addr)
        elif country == 'France':
            addr = self._normalize_french(addr)

        return addr.lower().strip()
    
    def _parse_address(self, addr: str, country: str) -> Dict[str, Any]:
        """Parse address using pypostal if available, else regex fallback"""
        if self.has_postal:
            return self._parse_address_libpostal(addr)
        else:
            return self._parse_address_regex(addr, country)
    
    def _parse_address_libpostal(self, addr: str) -> Dict[str, Any]:
        """Parse address using libpostal (pypostal)"""
        result = {
            'house_number': None,
            'road': None,
            'unit': None,
            'po_box': None,
            'postal_codes': [],
        }
        
        try:
            # Expand abbreviations first
            expanded = self._expand_address_libpostal_fn(addr)
            if expanded:
                addr = expanded[0]
        except Exception:
            pass
        
        try:
            parsed = self._parse_address_libpostal_fn(addr)
            parsed_dict = {label: value for value, label in parsed}
            
            result['house_number'] = parsed_dict.get('house_number')
            result['road'] = parsed_dict.get('road')
            result['unit'] = parsed_dict.get('unit')
            result['po_box'] = parsed_dict.get('po_box')
            result['postal_codes'] = [parsed_dict['postcode']] if parsed_dict.get('postcode') else []
            
        except Exception:
            pass
        
        return result
    
    def _parse_address_regex(self, addr: str, country: str) -> Dict[str, Any]:
        """Parse address using regex patterns (fallback)"""
        result = {
            'house_number': None,
            'road': None,
            'unit': None,
            'po_box': None,
            'postal_codes': [],
        }
        
        if not addr:
            return result
        
        # Postal codes
        pattern = self.postal_patterns.get(country)
        if pattern:
            result['postal_codes'] = pattern.findall(addr)
        
        # House number
        match = self.house_number_pattern.search(addr)
        if match:
            result['house_number'] = match.group(1).strip()
        
        # Road
        street_match = self.street_pattern.search(addr)
        if street_match:
            start = max(0, street_match.start() - 50)
            end = min(len(addr), street_match.end() + 20)
            result['road'] = addr[start:end].strip()
        
        # Unit
        match = self.unit_pattern.search(addr)
        if match:
            result['unit'] = match.group(1).strip()
        
        # PO Box
        match = self.po_box_pattern.search(addr)
        if match:
            result['po_box'] = match.group(1).strip()
        
        return result
    
    def _normalize_admin(self, addr: str, country: str, parsed: Dict) -> Dict[str, Optional[str]]:
        """Normalize administrative divisions using pycountry + geotext"""
        result = {'city': None, 'state': None, 'district': None}
        
        if addr:
            places = GeoText(addr)
            if places.cities:
                result['city'] = places.cities[0].lower()
        
        # Normalize state using pycountry
        country_codes = {'US': 'US', 'India': 'IN', 'France': 'FR'}
        cc = country_codes.get(country)
        
        if cc:
            for subdiv in pycountry.subdivisions.get(country_code=cc):
                if subdiv.name.lower() in addr:
                    result['state'] = subdiv.code.split('-')[-1]
                    break
        
        return result
    
    def _normalize_house_number(self, house_number: Any) -> str:
        """Normalize house numbers: '6267B' -> '6267', '123-A' -> '123'.

        Suffix letters usually denote sub-units of the same building, so the
        numeric base is the reliable blocking signal.
        """
        hn = self._safe_str(house_number).strip()
        if not hn:
            return ''
        match = re.match(r'^(\d+)', hn)
        return match.group(1) if match else hn.lower()

    def _char_ngrams(self, text: str, n: int = 3) -> Set[str]:
        text = text.lower()
        if len(text) < n:
            return set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def _get_phonetic_keys(self, name: str) -> Dict[str, str]:
        keys = {
            'metaphone': jellyfish.metaphone(name) if name else '',
            'soundex': jellyfish.soundex(name) if name else '',
            'nysiis': jellyfish.nysiis(name) if name else '',
        }
        # Sorted-token metaphone is robust to word-order permutations
        # e.g. "Aimei Waters & Millar" vs "Waters & Millar Aimei".
        if name:
            tokens = sorted(name.split())
            keys['metaphone_sorted'] = ' '.join(
                jellyfish.metaphone(t) for t in tokens if t)
        else:
            keys['metaphone_sorted'] = ''
        return keys
    
    def process_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        eid = self._safe_str(row['entity_id'])
        country = self._safe_str(row['country'])
        name = row['business_name']
        addr = row['business_address']
        
        name_clean = self._normalize_name(name, country)
        addr_clean = self._normalize_address(addr, country)
        
        parsed = self._parse_address(addr_clean, country)
        admin = self._normalize_admin(addr_clean, country, parsed)
        phonetic = self._get_phonetic_keys(name_clean)
        name_ngrams = self._char_ngrams(name_clean, self.ngram_size)
        # Address-level matching signals (mirrors name matching on addr_clean).
        addr_phonetic = self._get_phonetic_keys(addr_clean)
        addr_ngrams = self._char_ngrams(addr_clean, self.ngram_size)
        
        return {
            'entity_id': eid,
            'country': country,
            'name_raw': self._safe_str(row['business_name']),
            'addr_raw': self._safe_str(row['business_address']),
            'name_clean': name_clean,
            'addr_clean': addr_clean,
            'postal_codes': parsed['postal_codes'],
            'city': self._safe_str(admin['city']),
            'state': self._safe_str(admin['state']),
            'district': self._safe_str(admin['district']),
            'house_number': self._safe_str(parsed['house_number']),
            'road': self._safe_str(parsed['road']),
            'unit': self._safe_str(parsed['unit']),
            'po_box': self._safe_str(parsed['po_box']),
            'name_ngrams': list(name_ngrams),
            'metaphone': self._safe_str(phonetic['metaphone']),
            'metaphone_sorted': self._safe_str(phonetic['metaphone_sorted']),
            'house_number_norm': self._normalize_house_number(parsed['house_number']),
            'soundex': self._safe_str(phonetic['soundex']),
            'nysiis': self._safe_str(phonetic['nysiis']),
            'addr_ngrams': list(addr_ngrams),
            'addr_metaphone': self._safe_str(addr_phonetic['metaphone']),
            'addr_metaphone_sorted': self._safe_str(addr_phonetic['metaphone_sorted']),
            'addr_soundex': self._safe_str(addr_phonetic['soundex']),
            'addr_nysiis': self._safe_str(addr_phonetic['nysiis']),
        }
    
    def process_file(self, input_path: Path, output_path: Path) -> pl.DataFrame:
        print(f"Processing {input_path}...")
        
        df = pl.read_csv(input_path, separator='\t')
        results = []
        
        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            results.append(self.process_record(row))
        
        out_df = pl.DataFrame(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(output_path)
        
        print(f"  Saved {len(out_df)} records to {output_path}")
        return out_df
    
    def run(self):
        raw_dir = Path(self.config['paths']['raw_dir'])
        processed_dir = Path(self.config['paths']['processed_dir'])
        
        for split in ['train', 'test']:
            for src in ['source1', 'source2', 'source3']:
                in_path = raw_dir / split / f'{split}_{src}.tsv'
                out_path = processed_dir / split / f'{src}.parquet'
                
                if in_path.exists():
                    self.process_file(in_path, out_path)
                else:
                    print(f"  WARNING: {in_path} not found")


if __name__ == '__main__':
    pipeline = PreprocessingPipeline()
    pipeline.run()
