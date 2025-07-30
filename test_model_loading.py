#!/usr/bin/env python3
"""
Test model loading and basic forward pass
"""

import torch
from transformers import AutoModel
from fixed_load_gpn import TGMSAFixed
from datasets import load_dataset

def test_model_loading():
    print("=== Testing Model Loading ===")
    
    try:
        # Try to load the model
        enc = AutoModel.from_pretrained("songlab/gpn-msa-sapiens", trust_remote_code=True)
        print("✅ Model loaded successfully")
        
        # Print config
        print(f"Model config:")
        print(f"  vocab_size: {enc.config.vocab_size}")
        print(f"  hidden_size: {enc.config.hidden_size}")
        
        # Test with small batch
        print("\n=== Testing Forward Pass ===")
        tg = load_dataset("songlab/TraitGym", "complex_traits", split="test[:1]")
        dataset = TGMSAFixed(tg)
        sample = dataset[0]
        
        # Create mini batch
        ref = sample['ref'].unsqueeze(0)  # Add batch dimension
        alt = sample['alt'].unsqueeze(0)
        attn = sample['attn'].unsqueeze(0).bool()  # Convert to bool for attention mask
        
        print(f"Input shapes: ref={ref.shape}, alt={alt.shape}, attn={attn.shape}")
        print(f"Token ranges: ref=[{ref.min()}, {ref.max()}], alt=[{alt.min()}, {alt.max()}]")
        
        # Test model forward pass
        enc.eval()
        with torch.no_grad():
            try:
                out_ref = enc(ref, attention_mask=attn)
                print(f"✅ Forward pass successful")
                print(f"Output shape: {out_ref.last_hidden_state.shape}")
                
                # Test the center position (position 64)
                center_emb = out_ref.last_hidden_state[:, 64, :]
                print(f"Center embedding shape: {center_emb.shape}")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

if __name__ == "__main__":
    test_model_loading()