from typing import List, Dict, Tuple, Optional
"""
Cross-encoder for Pairwise Reranking
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm


class CrossEncoderDataset(Dataset):
    """Dataset for cross-encoder training"""
    
    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Prepare text: [CLS] name1 [SEP] addr1 [SEP] name2 [SEP] addr2 [SEP]
        text1 = f"{pair['name1']} [SEP] {pair['addr1']}"
        text2 = f"{pair['name2']} [SEP] {pair['addr2']}"
        
        encoding = self.tokenizer(
            text1, text2,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(pair['label'], dtype=torch.float)
        }


class CrossEncoderMatcher:
    """Cross-encoder for high-accuracy pairwise reranking"""
    
    def __init__(self, model_name: str = "xlm-roberta-base",
                 max_length: int = 256,
                 batch_size: int = 32,
                 device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        ).to(self.device)
    
    def _prepare_pairs(self, s1_records: List[Dict], s2_records: List[Dict],
                       labels: List[int]) -> List[Dict]:
        """Prepare pairs for training"""
        pairs = []
        for s1, s2, label in zip(s1_records, s2_records, labels):
            pairs.append({
                'name1': s1.get('name_clean', ''),
                'addr1': s1.get('addr_clean', ''),
                'name2': s2.get('name_clean', ''),
                'addr2': s2.get('addr_clean', ''),
                'label': label
            })
        return pairs
    
    def train(self, train_pairs: List[Dict], val_pairs: List[Dict] = None,
              epochs: int = 3, learning_rate: float = 2e-5,
              warmup_steps: int = 100) -> 'CrossEncoderMatcher':
        """Fine-tune cross-encoder on labeled pairs"""
        
        # Prepare datasets
        train_dataset = CrossEncoderDataset(train_pairs, self.tokenizer, self.max_length)
        val_dataset = CrossEncoderDataset(val_pairs, self.tokenizer, self.max_length) if val_pairs else None
        
        training_args = TrainingArguments(
            output_dir='./crossencoder_output',
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="steps" if val_pairs else "no",
            eval_steps=100 if val_pairs else None,
            save_steps=500,
            load_best_model_at_end=val_pairs is not None,
            metric_for_best_model="eval_loss" if val_pairs else None,
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        return self
    
    def predict_proba(self, s1_records: List[Dict], s2_records: List[Dict]) -> np.ndarray:
        """Predict match probabilities for pairs"""
        self.model.eval()
        probs = []
        
        pairs = self._prepare_pairs(s1_records, s2_records, [0] * len(s1_records))
        dataset = CrossEncoderDataset(pairs, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Cross-encoder inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs.extend(probs)
        
        return np.array(probs)
    
    def predict_batch(self, s1_records: List[Dict], s2_records: List[Dict]) -> np.ndarray:
        """Alias for predict_proba"""
        return self.predict_proba(s1_records, s2_records)
    
    def save(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'CrossEncoderMatcher':
        """Load model and tokenizer"""
        obj = cls.__new__(cls)
        obj.model_name = None
        obj.max_length = 256
        obj.batch_size = 32
        obj.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(obj.device)
        return obj
