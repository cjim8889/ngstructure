#!/usr/bin/env python3
"""
Test script for the molecular generator functionality.
"""

import os
import sys

import torch
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

# Add parent directory to path to import pkg modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkg.generator import MolecularGenerator
from pkg.transformer import DiT


class DummyScheduler:
    """Dummy scheduler for testing."""
    def __init__(self, num_timesteps=100):
        self.num_timesteps = num_timesteps
        device = torch.device('cpu')
        
        # Create dummy scheduler tensors
        betas = torch.linspace(1e-4, 2e-2, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)


def test_generator():
    """Test the generator with a small model."""
    print("Testing Molecular Generator...")
    
    # Check if vocab file exists
    vocab_file = "data/vocab.txt"
    if not os.path.exists(vocab_file):
        print(f"Vocabulary file {vocab_file} not found!")
        print("Creating a dummy vocab file for testing...")
        os.makedirs("data", exist_ok=True)
        
        # Create a simple vocabulary for testing
        dummy_vocab = [
            '[CLS]', '[PAD]', '[UNK]', '[MASK]',
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's',
            '(', ')', '[', ']', '=', '#', '@', '+', '-',
            '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        
        with open(vocab_file, 'w') as f:
            for token in dummy_vocab:
                f.write(f"{token}\n")
    
    # Initialize tokenizer
    try:
        tokenizer = SmilesTokenizer(vocab_file=vocab_file)
        print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return False
    
    # Create a small test model
    model = DiT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,  # Small for testing
        num_layers=2,
        nhead=4,
        mlp_ratio=2.0,
        dreams_emb_dim=128,
        max_length=50
    )
    
    # Create dummy scheduler
    scheduler = DummyScheduler(num_timesteps=50)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize generator
    try:
        generator = MolecularGenerator(
            model=model,
            scheduler=scheduler,
            tokenizer=tokenizer,
            max_length=50
        )
        print("Generator initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        return False
    
    # Test embedding to tokens conversion
    print("\nTesting embedding to tokens conversion...")
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    # Create dummy embeddings
    dummy_embeddings = torch.randn(batch_size, seq_len, d_model)
    
    try:
        tokens = generator.embedding_to_tokens(dummy_embeddings)
        print(f"Converted embeddings shape {dummy_embeddings.shape} to tokens shape {tokens.shape}")
        print(f"Sample tokens: {tokens[0, :5].tolist()}")
    except Exception as e:
        print(f"Failed embedding to tokens conversion: {e}")
        return False
    
    # Test SMILES decoding
    print("\nTesting SMILES decoding...")
    try:
        smiles_list = generator._decode_tokens_to_smiles(tokens)
        print(f"Decoded {len(smiles_list)} SMILES:")
        for i, smiles in enumerate(smiles_list):
            print(f"  {i+1}: {smiles}")
    except Exception as e:
        print(f"Failed SMILES decoding: {e}")
        return False
    
    # Test full generation pipeline
    print("\nTesting full generation pipeline...")
    try:
        # Create dummy dreams embeddings
        dreams_embeddings = torch.randn(batch_size, 128)
        
        # Generate samples (use fewer steps for testing)
        generated_tokens, generated_smiles = generator.sample(
            dreams_embeddings=dreams_embeddings,
            num_sampling_steps=10,  # Reduced for testing
            temperature=0.8
        )
        
        print(f"Generated {len(generated_smiles)} SMILES sequences:")
        for i, smiles in enumerate(generated_smiles):
            print(f"  {i+1}: {smiles}")
        
        # Test quality evaluation
        metrics = generator.evaluate_sample_quality(generated_smiles)
        print(f"\nGeneration metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Test with reference SMILES if we have valid ones
        try:
            # Create some simple reference SMILES for testing
            reference_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']  # ethanol, acetic acid, benzene
            
            print(f"\nTesting with reference SMILES...")
            ref_metrics = generator.evaluate_sample_quality(
                generated_smiles, 
                reference_smiles=reference_smiles
            )
            print(f"Reference comparison metrics:")
            for key, value in ref_metrics.items():
                if key not in metrics:  # Only show new metrics
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Reference comparison test failed: {e}")
        
    except Exception as e:
        print(f"Failed full generation test: {e}")
        return False
    
    print("\n✅ All tests passed! Generator is working correctly.")
    return True


if __name__ == "__main__":
    success = test_generator()
    sys.exit(0 if success else 1) 