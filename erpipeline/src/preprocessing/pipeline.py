"""
Preprocessing Pipeline - Library-First, No System Dependencies
Uses: aksharamukha, unidecode, company-name-match, pycountry, geotext, jellyfish, rapidfuzz
"""

import unicodedata
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass

import polars as pl
from tqdm import tqdm

import pycountry
from geotext import GeoText
from aksharamukha import transliterate
from unidecode import unidecode
import jellyfish
from company_name_match import clean_company_name

from ..config import get_config


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


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.countries = self.config['preprocessing']['countries']
        self.ngram_size = self.config['blocking']['name_ngram_size']
        
        # Pre-compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for address parsing"""
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
        
        # Administrative keywords (India)
        self.admin_keywords_india = {
            'district', 'taluk', 'tehsil', 'block', 'mandal', 'zone',
            'north', 'south', 'east', 'west', 'central'
        }
    
    def _strip_accents(self, text: str) -> str:
        """Remove combining marks (accent noise in US/India)"""
        return ''.join(c for c in unicodedata.normalize('NFD', text) 
                       if unicodedata.category(c) != 'Mn')
    
    def _normalize_french(self, text: str) -> str:
        """Normalize French accents to ASCII"""
        return unidecode(text)
    
    def _detect_script(self, text: str) -> str:
        """Detect primary script: Devanagari, Tamil, Latin"""
        scripts = set()
        for c in text:
            try:
                name = unicodedata.name(c)
                if 'DEVANAGARI' in name:
                    scripts.add('Devanagari')
                elif 'TAMIL' in name:
                    scripts.add('Tamil')
                elif 'LATIN' in name:
                    scripts.add('Latin')
            except ValueError:
                pass
        if 'Devanagari' in scripts:
            return 'Devanagari'
        if 'Tamil' in scripts:
            return 'Tamil'
        return 'Latin'
    
    def _transliterate_indic(self, text: str) -> str:
        """Auto-detect and transliterate Indic scripts to ISO Latin"""
        script = self._detect_script(text)
        if script == 'Devanagari':
            return transliterate.process('Devanagari', 'ISO', text)
        elif script == 'Tamil':
            return transliterate.process('Tamil', 'ISO', text)
        return text
    
    def _normalize_name(self, name: str, country: str) -> str:
        """Normalize business name using libraries"""
        # Unicode normalize
        name = unicodedata.normalize('NFKC', name)
        
        # Country-specific accent handling
        if country == 'US' or country == 'India':
            name = self._strip_accents(name)  # Remove noise accents
        elif country == 'France':
            name = self._normalize_french(name)  # Normalize French accents
        
        # Transliterate Indic scripts (India)
        if country == 'India':
            name = self._transliterate_indic(name)
        
        # Legal suffix normalization via library
        name = clean_company_name(name)
        
        return name.lower().strip()
    
    def _normalize_address(self, addr: str, country: str) -> str:
        """Normalize address text"""
        addr = unicodedata.normalize('NFKC', addr)
        
        if country == 'US' or country == 'India':
            addr = self._strip_accents(addr)
        elif country == 'France':
            addr = self._normalize_french(addr)
        
        if country == 'India':
            addr = self._transliterate_indic(addr)
        
        return addr.lower().strip()
    
    def _parse_address_regex(self, addr: str, country: str) -> Dict[str, Any]:
        """Parse address using regex patterns (pure Python fallback)"""
        result = {
            'house_number': None,
            'road': None,
            'unit': None,
            'po_box': None,
            'postal_codes': [],
        }
        
        # Extract postal codes
        pattern = self.postal_patterns.get(country)
        if pattern:
            result['postal_codes'] = pattern.findall(addr)
        
        # Extract house number
        match = self.house_number_pattern.search(addr)
        if match:
            result['house_number'] = match.group(1).strip()
        
        # Extract road/street (first street-type word + preceding words)
        street_match = self.street_pattern.search(addr)
        if street_match:
            # Get context around the street word
            start = max(0, street_match.start() - 50)
            end = min(len(addr), street_match.end() + 20)
            result['road'] = addr[start:end].strip()
        
        # Extract unit
        match = self.unit_pattern.search(addr)
        if match:
            result['unit'] = match.group(1).strip()
        
        # Extract PO Box
        match = self.po_box_pattern.search(addr)
        if match:
            result['po_box'] = match.group(1).strip()
        
        return result
    
    def _normalize_admin(self, addr: str, country: str, parsed: Dict) -> Dict[str, Optional[str]]:
        """Normalize administrative divisions using pycountry + geotext"""
        result = {
            'city': None,
            'state': None,
            'district': None,
        }
        
        # Use geotext for city extraction
        places = GeoText(addr)
        if places.cities:
            result['city'] = places.cities[0].lower()
        
        # Normalize state using pycountry
        if country == 'US':
            # Try to match US state names/abbreviations
            for subdiv in pycountry.subdivisions.get(country_code='US'):
                if subdiv.name.lower() in addr or subdiv.code.lower() in addr:
                    result['state'] = subdiv.code
                    break
        
        elif country == 'India':
            # Try to match Indian state names/abbreviations
            for subdiv in pycountry.subdivisions.get(country_code='IN'):
                if subdiv.name.lower() in addr:
                    result['state'] = subdiv.code.split('-')[-1]
                    break
        
        elif country == 'France':
            # Try to match French regions/departments
            for subdiv in pycountry.subdivisions.get(country_code='FR'):
                if subdiv.name.lower() in addr:
                    result['state'] = subdiv.code.split('-')[-1]
                    break
        
        return result
    
    def _char_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams"""
        text = text.lower()
        if len(text) < n:
            return set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def _get_phonetic_keys(self, name: str) -> Dict[str, str]:
        """Get phonetic encodings using jellyfish"""
        return {
            'metaphone': jellyfish.metaphone(name) if name else '',
            'soundex': jellyfish.soundex(name) if name else '',
            'nysiis': jellyfish.nysiis(name) if name else '',
        }
    
    def process_record(self, row: Dict[str, Any]) -> ProcessedRecord:
        eid = row['entity_id']
        country = row['country']
        name = row['business_name']
        addr = row['business_address']
        
        # Normalize name
        name_clean = self._normalize_name(name, country)
        
        # Normalize address
        addr_clean = self._normalize_address(addr, country)
        
        # Parse address with regex fallback
        parsed = self._parse_address_regex(addr_clean, country)
        
        # Extract address components
        admin = self._normalize_admin(addr_clean, country, parsed)
        
        # Phonetic keys
        phonetic = self._get_phonetic_keys(name_clean)
        
        # Name n-grams for blocking
        name_ngrams = self._char_ngrams(name_clean, self.ngram_size)
        
        return ProcessedRecord(
            entity_id=eid,
            country=country,
            name_raw=name,
            addr_raw=addr,
            name_clean=name_clean,
            addr_clean=addr_clean,
            postal_codes=parsed['postal_codes'],
            city=admin['city'],
            state=admin['state'],
            district=admin['district'],
            house_number=parsed['house_number'],
            road=parsed['road'],
            unit=parsed['unit'],
            po_box=parsed['po_box'],
            name_ngrams=list(name_ngrams),
            metaphone=phonetic['metaphone'],
            soundex=phonetic['soundex'],
            nysiis=phonetic['nysiis'],
        )
    
    def process_file(self, input_path: Path, output_path: Path) -> pl.DataFrame:
        """Process a single TSV file to Parquet"""
        print(f"Processing {input_path}...")
        
        # Read TSV with polars (streaming)
        df = pl.read_csv(input_path, separator='\t')
        
        # Process records
        results = []
        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            processed = self.process_record(row)
            results.append({
                'entity_id': processed.entity_id,
                'country': processed.country,
                'name_raw': processed.name_raw,
                'addr_raw': processed.addr_raw,
                'name_clean': processed.name_clean,
                'addr_clean': processed.addr_clean,
                'postal_codes': processed.postal_codes,
                'city': processed.city,
                'state': processed.state,
                'district': processed.district,
                'house_number': processed.house_number,
                'road': processed.road,
                'unit': processed.unit,
                'po_box': processed.po_box,
                'name_ngrams': processed.name_ngrams,
                'metaphone': processed.metaphone,
                'soundex': processed.soundex,
                'nysiis': processed.nysiis,
            })
        
        out_df = pl.DataFrame(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(output_path)
        
        print(f"  Saved {len(out_df)} records to {output_path}")
        return out_df
    
    def run(self):
        """Process all source files for train and test"""
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
