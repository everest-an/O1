"""
router.py — Cloud Oracle Router for Awareness Network.

This module acts as the "API Handshake" for the Hybrid Edge-State architecture,
allowing the local MT-LNN (AwareLiquid) to securely query a larger cloud model
(like Awareness Cloud, Gemini, or ChatGPT) strictly for factual knowledge when
it hits an internal cognitive blind spot.
"""

import time
import requests
import torch
from typing import Optional, Dict

class CloudOracleRouter:
    """
    Manages asymmetric queries to external Cloud LLMs.
    """
    def __init__(self, endpoint: str = "mock", api_key: str = "mock_key"):
        self.endpoint = endpoint
        self.api_key = api_key
        
    def query(self, topic: str) -> str:
        """
        Sends a surgical query to the Cloud Oracle and retrieves factual context.
        """
        print(f"\n[Cloud Oracle Router] Dispatching surgical query to cloud API: '{topic}'...")
        # Simulate network latency
        time.sleep(1.5)
        
        # MOCK IMPLEMENTATION FOR MVP
        mock_database = {
            "m-theory": "M-theory is a framework in physics that unifies all consistent versions of superstring theory. It was first conjectured by Edward Witten.",
            "awareliquid": "AwareLiquid is a revolutionary $O(1)$ recurrent state flow engine mimicking microtubule intelligence.",
            "tokyo": "Tokyo is the capital of Japan, known for its mix of modern and traditional architecture.",
            "quantum": "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
            "mamba": "Mamba is a simplified linear RNN architecture that historically struggles with information degradation over extreme long context, unlike MT-LNN."
        }
        
        # Find partial match
        for key, value in mock_database.items():
            if key.lower() in topic.lower():
                print(f"[Cloud Oracle Router] Retrieved {len(value)} bytes from Azure/Awareness Cloud.")
                return value
                
        # Default fallback
        fallback = f"Fact retrieval result for '{topic}': External API indicates this is a complex subject currently under research."
        print(f"[Cloud Oracle Router] Retrieved {len(fallback)} bytes from Azure/Awareness Cloud.")
        return fallback

    def inject_to_local_state(self, fact: str, tokenizer, model, cache, device="cpu"):
        """
        [Quiet Mode] Feeds factual text into the local engine without emitting tokens.
        Updates the 4.1KB h_prev state invisibly.
        """
        print(f"[Local Brain] Silently absorbing cloud facts into local awareness state...")
        
        # We need prefill_state_only
        from .streaming import prefill_state_only
        
        # Prepare the silent context (mock ASCII mapping for tiny sandbox)
        quiet_prompt = f"<|oracle|>\n{fact}\n<|endoracle|>\n"
        ids_list = [ord(c) % 200 for c in quiet_prompt]
        input_ids = torch.tensor([ids_list], dtype=torch.long).to(device)
        
        # Update cache invisibly
        _, updated_cache = prefill_state_only(
            model, 
            input_ids, 
            cache=cache, 
            use_lnn_recurrence=True
        )
        print("[Local Brain] Absorption complete. O(1) state matrix updated.")
        return updated_cache
