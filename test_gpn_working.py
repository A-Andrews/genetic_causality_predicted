#!/usr/bin/env python3
"""
Test working GPN model loading using direct imports
"""

import torch
from gpn.model import GPNRoFormerModel
from fixed_load_gpn import TGMSAFixed
from datasets import load_dataset

def test_gpn_working():
    print("=== Testing Working GPN Model ===")
    
    try:
        # Load model using direct GPN import
        enc = GPNRoFormerModel.from_pretrained("songlab/gpn-msa-sapiens")
        print("✅ Model loaded successfully with GPN direct import")
        
        # Print config
        print(f"Model config:")
        print(f"  vocab_size: {enc.config.vocab_size}")
        print(f"  hidden_size: {enc.config.hidden_size}")
        print(f"  max_position_embeddings: {enc.config.max_position_embeddings}")
        
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
        
        # Check token compatibility
        if ref.max() < enc.config.vocab_size and alt.max() < enc.config.vocab_size:
            print(f"✅ Tokens compatible with vocab_size ({enc.config.vocab_size})")
        else:
            print(f"❌ Tokens incompatible: max tokens {max(ref.max().item(), alt.max().item())} >= vocab_size {enc.config.vocab_size}")
            return
        
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
                
                # Test alt sequence
                out_alt = enc(alt, attention_mask=attn)
                center_emb_alt = out_alt.last_hidden_state[:, 64, :]
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(center_emb, center_emb_alt).item()
                print(f"Cosine similarity between ref and alt: {cos_sim:.4f}")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpn_working()