"""
Threshold Calibration for Macro F0.5 Optimization
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support
import pickle


def macro_f05(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    """Compute macro F-beta score"""
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def calibrate_thresholds(probs: np.ndarray, labels: np.ndarray,
                         countries: np.ndarray,
                         beta: float = 0.5) -> Dict[str, float]:
    """Find optimal threshold per country for macro F-beta"""
    thresholds = {}
    
    for country in np.unique(countries):
        mask = countries == country
        if mask.sum() == 0:
            thresholds[country] = 0.5
            continue
        
        best_thresh = 0.5
        best_score = 0.0
        
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (probs[mask] > thresh).astype(int)
            score = macro_f05(labels[mask], preds, beta)
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        thresholds[country] = best_thresh
    
    return thresholds


def apply_thresholds(probs: np.ndarray, countries: np.ndarray,
                     thresholds: Dict[str, float],
                     default_thresh: float = 0.5) -> np.ndarray:
    """Apply per-country thresholds to probabilities"""
    preds = np.zeros(len(probs), dtype=int)
    for country in np.unique(countries):
        mask = countries == country
        thresh = thresholds.get(country, 0.5)
        preds[mask] = (probs[mask] > thresh).astype(int)
    return preds


def calibrate_global_threshold(probs: np.ndarray, labels: np.ndarray,
                                beta: float = 0.5) -> float:
    """Find single global threshold for macro F-beta"""
    best_thresh = 0.5
    best_score = 0.0
    
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (probs > thresh).astype(int)
        score = macro_f05(labels, preds, beta)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh


class ThresholdCalibrator:
    """Calibrate per-country and per-entity thresholds"""
    
    def __init__(self, beta: float = 0.5):
        self.beta = beta
        self.thresholds = {}
        self.global_threshold = 0.5
        self.singleton_threshold = 0.5
    
    def fit(self, probs: np.ndarray, labels: np.ndarray,
            countries: np.ndarray,
            singleton_probs: Optional[np.ndarray] = None,
            singleton_labels: Optional[np.ndarray] = None):
        """Calibrate all thresholds"""
        
        # Per-country thresholds
        self.thresholds = calibrate_thresholds(probs, labels, countries, self.beta)
        
        # Global threshold
        self.global_threshold = calibrate_global_threshold(probs, labels, self.beta)
        
        # Singleton threshold (if data provided)
        if singleton_probs is not None and singleton_labels is not None:
            best_t = 0.5
            best_s = 0.0
            for t in np.linspace(0.1, 0.9, 81):
                preds = (singleton_probs > t).astype(int)
                s = macro_f05(singleton_labels, preds, self.beta)
                if s > best_s:
                    best_s = s
                    best_t = t
            self.singleton_threshold = best_t
    
    def predict(self, probs: np.ndarray, countries: np.ndarray,
                singleton_probs: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply thresholds"""
        preds = apply_thresholds(probs, countries, self.thresholds)
        
        # Singleton override
        if singleton_probs is not None:
            singleton_mask = singleton_probs > self.singleton_threshold
            preds[singleton_mask] = 0  # Force no-match for predicted singletons
        
        return preds
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'thresholds': self.thresholds,
                'global_threshold': self.global_threshold,
                'singleton_threshold': self.singleton_threshold,
                'beta': self.beta,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'ThresholdCalibrator':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(data['beta'])
        obj.thresholds = data['thresholds']
        obj.global_threshold = data['global_threshold']
        obj.singleton_threshold = data['singleton_threshold']
        return obj
