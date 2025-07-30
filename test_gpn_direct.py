#!/usr/bin/env python3
"""
Test loading GPN model directly through the GPN library
"""

import torch
import gpn.model
from fixed_load_gpn import TGMSAFixed
from datasets import load_dataset

def test_gpn_direct():
    print("=== Testing GPN Direct Loading ===")
    
    try:
        # Try different ways to load the GPN model
        print("Attempting to load with gpn.model...")
        
        # Check what's available in gpn.model
        print(f"Available in gpn.model: {dir(gpn.model)}")
        
        # Try the original approach from the working gpn_test.py
        from transformers import AutoModel
        
        # Let's check if the model files contain the architecture
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id="songlab/gpn-msa-sapiens", filename="config.json")
        
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        
        print(f"Model config from hub: {config}")
        
    except Exception as e:
        print(f"❌ Direct loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpn_direct()