"""
capsule.py — State persistence module for MT-LNN (AwareLiquid).

This module enables serializing the O(1) recurrent states (h_prev) of the
MT-LNN model to portable physical files (.capsule). Unlike SessionMemory
which stores states in a local SQLite database, a capsule allows a session's
"consciousness" to be transferred across devices, backed up, or loaded
instantly without recomputing prompt tokens.
"""

import os
import torch
from typing import Optional
from .model import ModelCacheStruct


def save_capsule(cache: ModelCacheStruct, filepath: str) -> None:
    """
    Serialize the recurrent state (h_prev) of the MT-LNN model to a file.
    
    Args:
        cache: The ModelCacheStruct containing current conversational state.
        filepath: Save destination (e.g., 'dialogue_day_3.capsule')
    """
    # Extract only the recurrent hidden states (h_prev) and token count
    h_states = []
    for layer_cache in cache.layers:
        h_prev = layer_cache[1] if (layer_cache is not None and len(layer_cache) > 1) else None
        h_states.append(h_prev.cpu() if h_prev is not None else None)
        
    capsule_data = {
        "version": "1.0",
        "token_count": cache.token_count,
        "h_states": h_states
    }
    
    # Save the dictionary. Using weights_only=False since it's an internal dict
    torch.save(capsule_data, filepath)
    
    # Quick diagnostics
    num_layers = len([h for h in h_states if h is not None])
    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"[State Capsule] Successfully crystallized {num_layers} layers of state into {filepath} ({file_size_kb:.1f} KB).")


def load_capsule(filepath: str, device: str = "cpu") -> ModelCacheStruct:
    """
    Load the recurrent state (h_prev) from a file into a new ModelCacheStruct.
    
    Args:
        filepath: Path to the .capsule file.
        device: Target device for tensors (e.g., 'cuda', 'cpu').
        
    Returns:
        A new ModelCacheStruct seeded with the historical state.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Capsule file not found: {filepath}")
        
    capsule_data = torch.load(filepath, map_location=device, weights_only=False)
    h_states = capsule_data.get("h_states", [])
    token_count = capsule_data.get("token_count", 0)
    
    new_cache = ModelCacheStruct(token_count=token_count)
    for h in h_states:
        h_tensor = h.to(device) if h is not None else None
        # layer_cache tuple: (kv_state, h_prev, coherence_state)
        new_cache.layers.append((None, h_tensor, None))
        
    print(f"[State Capsule] Inherited awareness shape. Recovered {token_count} tokens of accumulated context from {filepath}.")
    return new_cache
