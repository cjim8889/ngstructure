import os
from typing import Optional

import torch
import numpy as np
import pandas as pd
import lightning as L
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class MolecularDataset(Dataset):
    """Dataset for molecular diffusion training that loads SMILES and dreams_embedding."""
    
    def __init__(self, meta_file: str, arrays_file: str, vocab_file: str, max_length: int = 500):
        self.meta_df = pd.read_csv(meta_file)
        # self.arrays_data = np.load(arrays_file)
        self.tokenizer = SmilesTokenizer(vocab_file=vocab_file)
        self.max_length = max_length
        
        # Extract dreams_embedding from NPZ file

        with np.load(arrays_file) as arrays_data:
            self.dreams_embeddings = arrays_data['dreams_embedding'].copy()
        
        # Encode categorical variables for conditioning
        self.instrument_encoder = LabelEncoder()
        self.adduct_encoder = LabelEncoder()
        
        # Fit encoders on all unique values
        self.meta_df['instrument_encoded'] = self.instrument_encoder.fit_transform(
            self.meta_df['instrument_type'].fillna('unknown')
        )
        self.meta_df['adduct_encoded'] = self.adduct_encoder.fit_transform(
            self.meta_df['adduct'].fillna('unknown')
        )
        
        print(f"Dataset loaded: {len(self.meta_df)} samples")
        print(f"Dreams embedding shape: {self.dreams_embeddings.shape}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        print(f"Unique instruments: {len(self.instrument_encoder.classes_)}")
        print(f"Unique adducts: {len(self.adduct_encoder.classes_)}")
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        
        # Get SMILES string and tokenize it
        smiles = row['smiles']
        
        # Tokenize SMILES string
        tokenized_smiles = self.tokenizer.encode(smiles)
        padded_tokens = self.tokenizer.add_padding_tokens(tokenized_smiles, length=self.max_length)
        
        # Convert tokens to tensor
        tokens = torch.tensor(padded_tokens, dtype=torch.long)

        mask = tokens != self.tokenizer.pad_token_id

        
        # Get dreams embedding for this sample
        dreams_emb = torch.from_numpy(self.dreams_embeddings[idx]).float()
         
        return {
            'smiles': smiles,
            'tokens': tokens,
            'token_mask': mask,
            'dreams_embedding': dreams_emb,
            'collision_energy': torch.tensor(row['collision_energy'], dtype=torch.float32),
            'instrument_type': torch.tensor(row['instrument_encoded'], dtype=torch.long),
            'adduct': torch.tensor(row['adduct_encoded'], dtype=torch.long),
            'parent_mass': torch.tensor(row['parent_mass'], dtype=torch.float32),
            'precursor_mz': torch.tensor(row['precursor_mz'], dtype=torch.float32),
        }


class MolecularDataModule(L.LightningDataModule):
    """Lightning data module for molecular data with SMILES and dreams embeddings."""
    
    def __init__(
        self,
        data_dir: str = "data",
        vocab_file: str = "data/vocab.txt",
        batch_size: int = 32,
        max_length: int = 64,
        num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        
        # Store dataset info for model initialization
        self.num_instruments = None
        self.num_adducts = None
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MolecularDataset(
                meta_file=os.path.join(self.data_dir, "train_meta.csv"),
                arrays_file=os.path.join(self.data_dir, "train_arrays.npz"),
                vocab_file=self.vocab_file,
                max_length=self.max_length
            )
            self.val_dataset = MolecularDataset(
                meta_file=os.path.join(self.data_dir, "val_meta.csv"),
                arrays_file=os.path.join(self.data_dir, "val_arrays.npz"),
                vocab_file=self.vocab_file,
                max_length=self.max_length
            )
            
            # Store dataset info
            self.num_instruments = len(self.train_dataset.instrument_encoder.classes_)
            self.num_adducts = len(self.train_dataset.adduct_encoder.classes_)
            
        if stage == "test" or stage is None:
            self.test_dataset = MolecularDataset(
                meta_file=os.path.join(self.data_dir, "test_meta.csv"),
                arrays_file=os.path.join(self.data_dir, "test_arrays.npz"),
                vocab_file=self.vocab_file,
                max_length=self.max_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_dataset_info(self):
        """Get information about the dataset for model initialization."""
        if self.num_instruments is None or self.num_adducts is None:
            # Setup if not already done
            self.setup("fit")
        
        return {
            'num_instruments': self.num_instruments,
            'num_adducts': self.num_adducts,
            'vocab_size': self.train_dataset.tokenizer.vocab_size,
            'dreams_emb_dim': self.train_dataset.dreams_embeddings.shape[1],
        } 
    
if __name__ == "__main__":
    data_module = MolecularDataModule()
    data_module.setup("fit")
    print(data_module.train_dataset[0])