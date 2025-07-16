import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

# self made imports
from molecules import WeightMolecule

@dataclass
class MolecularLattice:
    """A 2D lattice structure of weight molecules"""
    molecules: List[List[WeightMolecule]]
    layer_name: str
    epoch: int
    lattice_id: str
    
    def __post_init__(self):
        if not self.lattice_id:
            self.lattice_id = self._generate_lattice_id()
    
    def _generate_lattice_id(self): # -> str:
        """Generate unique ID based on molecular composition"""
        molecule_string = ""
        for row in self.molecules:
            for mol in row:
                molecule_string += str(mol)
        return hashlib.md5(molecule_string.encode()).hexdigest()[:8]
    
    def get_molecular_formula(self): # -> str:
        """Get chemical-like formula for the lattice"""
        molecule_counts = defaultdict(int)
        for row in self.molecules:
            for mol in row:
                molecule_counts[mol.atomic_symbol] += 1
        
        formula = ""
        for symbol, count in sorted(molecule_counts.items()):
            if count > 1:
                formula += f"{symbol}{count}"
            else:
                formula += symbol
        return formula
    
    def can_be_assembled_from(self, available_lattices: List['MolecularLattice']): # -> bool:
        """Check if this lattice can be assembled from available molecular components"""
        our_molecules = set()
        for row in self.molecules:
            for mol in row:
                our_molecules.add(mol)
        
        available_molecules = set()
        for lattice in available_lattices:
            for row in lattice.molecules:
                for mol in row:
                    available_molecules.add(mol)
        
        return our_molecules.issubset(available_molecules)