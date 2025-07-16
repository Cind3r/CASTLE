import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class WeightMolecule:
    """A discrete molecular unit representing a spatial region of weights"""
    atomic_weight: float  # Average of the kxk region
    atomic_symbol: str    # Symbol based on weight magnitude
    position: Tuple[int, int]  # Position in the lattice
    size: Tuple[int, int]      # Size of the molecule (kxk)
    
    def __str__(self):
        return f"{self.atomic_symbol}_{self.atomic_weight:.2f}"
    
    def __hash__(self):
        return hash((self.atomic_symbol, round(self.atomic_weight, 2)))
    
    def __eq__(self, other):
        return (self.atomic_symbol == other.atomic_symbol and 
                abs(self.atomic_weight - other.atomic_weight) < 0.01)