#!/usr/bin/env python3
"""
Test the fixed data loader
"""

from datasets import load_dataset
from fixed_load_gpn import TGMSAFixed

def test_fixed_loader():
    print("=== Testing Fixed Data Loader ===")
    
    # Load a small subset of TraitGym data
    tg = load_dataset(
        "songlab/TraitGym",
        "complex_traits",  # Use smaller non-full version for testing
        split="test[:10]",  # Just first 10 samples
    )
    
    print(f"Loaded {len(tg)} samples")
    
    # Test the fixed loader
    dataset = TGMSAFixed(tg)
    
    # Get a few samples
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  ref shape: {sample['ref'].shape}, range: [{sample['ref'].min()}, {sample['ref'].max()}]")
            print(f"  alt shape: {sample['alt'].shape}, range: [{sample['alt'].min()}, {sample['alt'].max()}]")
            print(f"  attn shape: {sample['attn'].shape}")
            print(f"  label: {sample['label']}")
            print(f"  ref sample: {sample['ref'][:10]}")
            print(f"  alt sample: {sample['alt'][:10]}")
            
            # Check if tokens are in reasonable range (0-4 for ACGT-)
            if sample['ref'].max() <= 4 and sample['alt'].max() <= 4:
                print(f"  ✅ Tokens in valid range")
            else:
                print(f"  ❌ Tokens out of range")
                
        except Exception as e:
            print(f"  ❌ Error processing sample {i}: {e}")

if __name__ == "__main__":
    test_fixed_loader()