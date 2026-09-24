"""
Quick test to verify all libraries are working
"""

def test_imports():
    """Test all library imports"""
    print("Testing imports...")
    
    try:
        import polars as pl
        print(f"  ✅ polars {pl.__version__}")
    except Exception as e:
        print(f"  ❌ polars: {e}")
    
    try:
        import numpy as np
        print(f"  ✅ numpy {np.__version__}")
    except Exception as e:
        print(f"  ❌ numpy: {e}")
    
    try:
        from postal.parser import parse_address
        from postal.expand import expand_address
        print(f"  ✅ pypostal (libpostal)")
    except Exception as e:
        print(f"  ❌ pypostal: {e}")
    
    try:
        import pycountry
        print(f"  ✅ pycountry")
    except Exception as e:
        print(f"  ❌ pycountry: {e}")
    
    try:
        from aksharamukha import transliterate
        print(f"  ✅ aksharamukha")
    except Exception as e:
        print(f"  ❌ aksharamukha: {e}")
    
    try:
        from unidecode import unidecode
        print(f"  ✅ unidecode")
    except Exception as e:
        print(f"  ❌ unidecode: {e}")
    
    try:
        from rapidfuzz import fuzz
        print(f"  ✅ rapidfuzz")
    except Exception as e:
        print(f"  ❌ rapidfuzz: {e}")
    
    try:
        import jellyfish
        print(f"  ✅ jellyfish")
    except Exception as e:
        print(f"  ❌ jellyfish: {e}")
    
    try:
        from company_name_match import clean_company_name
        print(f"  ✅ company-name-match")
    except Exception as e:
        print(f"  ❌ company-name-match: {e}")
    
    try:
        import lightgbm as lgb
        print(f"  ✅ lightgbm")
    except Exception as e:
        print(f"  ❌ lightgbm: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  ✅ sentence-transformers")
    except Exception as e:
        print(f"  ❌ sentence-transformers: {e}")
    
    try:
        import faiss
        print(f"  ✅ faiss")
    except Exception as e:
        print(f"  ❌ faiss: {e}")
    
    try:
        from datasketch import MinHash, MinHashLSH
        print(f"  ✅ datasketch")
    except Exception as e:
        print(f"  ❌ datasketch: {e}")
    
    try:
        import splink
        print(f"  ✅ splink")
    except Exception as e:
        print(f"  ❌ splink: {e}")


def test_basic_functionality():
    """Test basic library functionality"""
    print("\nTesting basic functionality...")
    
    # Test unidecode
    from unidecode import unidecode
    test = "Café Résumé Piñata"
    result = unidecode(test)
    print(f"  unidecode: '{test}' → '{result}'")
    
    # Test rapidfuzz
    from rapidfuzz import fuzz
    sim = fuzz.ratio("Apple Inc", "Apple Incorporated")
    print(f"  rapidfuzz: 'Apple Inc' vs 'Apple Incorporated' = {sim}")
    
    # Test jellyfish
    import jellyfish
    meta = jellyfish.metaphone("Smith")
    print(f"  jellyfish metaphone: 'Smith' → '{meta}'")
    
    # Test company-name-match
    from company_name_match import clean_company_name
    cleaned = clean_company_name("Apple Inc.")
    print(f"  company-name-match: 'Apple Inc.' → '{cleaned}'")
    
    # Test aksharamukha
    from aksharamukha import transliterate
    deva = "एसएस फूड"
    latin = transliterate.process('Devanagari', 'ISO', deva)
    print(f"  aksharamukha: '{deva}' → '{latin}'")
    
    # Test pycountry
    import pycountry
    us = pycountry.countries.get(alpha_2='US')
    print(f"  pycountry: US = {us.name}")
    
    # Test pypostal
    try:
        from postal.parser import parse_address
        parsed = parse_address("123 Main St, New York, NY 10001")
        print(f"  pypostal: parsed {len(parsed)} components")
        for v, k in parsed:
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  pypostal: {e}")


if __name__ == '__main__':
    test_imports()
    test_basic_functionality()
