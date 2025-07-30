#!/usr/bin/env python3
"""
Minimal debug script for MSA encoding issues - login node safe
"""

import numpy as np
from gpn.data import GenomeMSA
from settings import ZARR_PATH

def debug_encoding_minimal():
    print("=== Minimal MSA Debug (CPU only) ===")
    
    # Load MSA
    try:
        msa = GenomeMSA(ZARR_PATH)
        print(f"✅ MSA loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load MSA: {e}")
        return
    
    # Check MSA properties
    print(f"MSA has encode method: {hasattr(msa, 'encode')}")
    print(f"MSA gap_token: {getattr(msa, 'gap_token', 'Not found')}")
    
    # Test a small MSA slice
    try:
        chrom = "17"
        pos0 = 43106463  # 0-based BRCA1 position
        start, end = pos0 - 5, pos0 + 5  # Very small window
        
        print(f"\nTesting small MSA slice: chr{chrom}:{start}-{end}")
        X = msa.get_msa(chrom, start, end)
        print(f"Raw MSA shape: {X.shape}")
        print(f"Raw MSA dtype: {X.dtype}")
        
        # Check data type and conversion
        VOCAB = {c: i for i, c in enumerate("ACGT-")}
        if np.issubdtype(X.dtype, np.bytes_):
            print("Converting byte strings...")
            X_converted = np.vectorize(lambda b: VOCAB[b.decode("ascii").upper()])(X)
            print(f"Converted shape: {X_converted.shape}, dtype: {X_converted.dtype}")
            print(f"Converted values: {X_converted.flatten()}")
            X = X_converted
        else:
            print(f"Raw values: {X.flatten()}")
        
        # Test encoding methods
        if hasattr(msa, "encode"):
            print("\nTesting MSA encode method...")
            try:
                tokens = msa.encode(X)
                print(f"Encoded successfully: shape {tokens.shape}, dtype {tokens.dtype}")
                print(f"Token range: [{np.min(tokens)}, {np.max(tokens)}]")
                print(f"Sample tokens: {tokens}")
            except Exception as e:
                print(f"MSA encode failed: {e}")
        
        print("\nTesting base-5 fallback...")
        try:
            base = np.power(5, np.arange(X.shape[1]), dtype=np.int64)
            tokens_fallback = (X.astype(np.int64) * base).sum(axis=1)
            print(f"Base-5 tokens: {tokens_fallback}")
            print(f"Base-5 range: [{np.min(tokens_fallback)}, {np.max(tokens_fallback)}]")
        except Exception as e:
            print(f"Base-5 fallback failed: {e}")
            
    except Exception as e:
        print(f"❌ MSA slice failed: {e}")
    
    # Check model config without loading weights
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("songlab/gpn-msa-sapiens")
        print(f"\n=== Model Config ===")
        print(f"vocab_size: {config.vocab_size}")
        print(f"hidden_size: {config.hidden_size}")
        
        # Compare with our token values
        if 'tokens' in locals():
            max_token = np.max(tokens)
            if max_token >= config.vocab_size:
                print(f"❌ ERROR: Max token ({max_token}) >= vocab_size ({config.vocab_size})")
            else:
                print(f"✅ Tokens within vocab_size")
                
    except Exception as e:
        print(f"❌ Config loading failed: {e}")

if __name__ == "__main__":
    debug_encoding_minimal()