"""
LightGBM Matching Model with Macro F0.5 Calibration
"""

import numpy as np
import polars as pl
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pickle
import warnings
warnings.filterwarnings('ignore')


class LightGBMMatcher:
    """LightGBM-based entity matching model with F0.5 calibration"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
            'verbose': -1,
            'random_state': 42,
        }
        self.model = None
        self.feature_names = None
        self.thresholds = {}  # per-country thresholds
        self.beta = 0.5  # F0.5
        
    def prepare_data(self, pairs_df: pl.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from pairs DataFrame"""
        X = pairs_df.select(feature_cols).to_numpy()
        y = pairs_df['label'].to_numpy()
        return X, y
    
    def train(self, train_pairs: pl.DataFrame, val_pairs: pl.DataFrame, 
              feature_cols: List[str], country_col: str = 'country') -> 'LightGBMMatcher':
        """Train LightGBM model with validation"""
        
        X_train, y_train = self.prepare_data(train_pairs, feature_cols)
        X_val, y_val = self.prepare_data(val_pairs, feature_cols)
        
        self.feature_names = feature_cols
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(self.params['early_stopping_rounds']),
                lgb.log_evaluation(50)
            ]
        )
        
        # Calibrate thresholds per country on validation set
        self._calibrate_thresholds(val_pairs, feature_cols, country_col)
        
        return self
    
    def _calibrate_thresholds(self, val_pairs: pl.DataFrame, feature_cols: List[str], country_col: str):
        """Calibrate decision threshold per country for macro F0.5"""
        
        X_val, y_val = self.prepare_data(val_pairs, feature_cols)
        countries = val_pairs[country_col].to_numpy()
        
        val_probs = self.predict_proba(val_pairs, feature_cols)
        
        unique_countries = np.unique(countries)
        self.thresholds = {}
        
        for country in unique_countries:
            mask = countries == country
            if mask.sum() == 0:
                self.thresholds[country] = 0.5
                continue
            
            best_thresh = 0.5
            best_score = 0.0
            
            # Grid search for optimal threshold
            for thresh in np.linspace(0.1, 0.9, 81):
                preds = (val_probs[mask] > thresh).astype(int)
                score = self._macro_f05(y_val[mask], preds)
                
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            
            self.thresholds[country] = best_thresh
            print(f"  Country {country}: threshold={best_thresh:.3f}, F0.5={best_score:.4f}")
    
    def _macro_f05(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute macro F0.5 score"""
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        if precision + recall == 0:
            return 0.0
        beta = 0.5
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    def predict_proba(self, pairs_df: pl.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Predict match probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = pairs_df.select(feature_cols).to_numpy()
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, pairs_df: pl.DataFrame, feature_cols: List[str], 
                country_col: str = 'country') -> np.ndarray:
        """Predict match labels using per-country thresholds"""
        probs = self.predict_proba(pairs_df, feature_cols)
        countries = pairs_df['country'].to_numpy()
        
        preds = np.zeros(len(probs), dtype=int)
        for country in np.unique(countries):
            mask = countries == country
            thresh = self.thresholds.get(country, 0.5)
            preds[mask] = (probs[mask] > thresh).astype(int)
        
        return preds
    
    def save(self, path: str):
        """Save model and thresholds"""
        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'thresholds': self.thresholds,
            'params': self.params,
            'beta': self.beta,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMMatcher':
        """Load model and thresholds"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(data['params'])
        obj.model = data['model']
        obj.feature_names = data['feature_names']
        obj.thresholds = data['thresholds']
        obj.beta = data.get('beta', 0.5)
        return obj


class SingletonClassifier:
    """Separate classifier for singleton (no-match) detection"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def extract_features(self, s1: Dict, candidates: List[Dict]) -> np.ndarray:
        """Extract singleton features"""
        if not candidates:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Strong singleton signal
        
        max_name_sim = max(c.get('fuzz_wratio', 0) for c in candidates)
        max_addr_sim = max(c.get('addr_token_jaccard', 0) for c in candidates)
        max_token_jaccard = max(c.get('token_jaccard', 0) for c in candidates)
        max_pin = max(c.get('pin_exact', 0) for c in candidates)
        max_city = max(c.get('city_exact', 0) for c in candidates)
        max_house = max(c.get('house_number_match', 0) for c in candidates)
        num_candidates = len(candidates)
        
        return np.array([
            max_name_sim, max_addr_sim, max_token_jaccard,
            max_pin, max_city, max_house, num_candidates
        ])
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train singleton classifier"""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'
        )
        self.model.fit(X, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict singleton probability"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model}, f)
    
    @classmethod
    def load(cls, path: str) -> 'SingletonClassifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data['model']
        return obj
