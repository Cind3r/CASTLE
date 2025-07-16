import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# self made imports
from molecules import WeightMolecule
from lattice import MolecularLattice


class MolecularAssemblyTracker:
    """Assembly theory tracker using molecular lattice structures"""
    
    def __init__(self, molecule_size: int = 2, weight_precision: int = 3):
        
        self.molecule_size = molecule_size  # kxk size for each molecule
        self.weight_precision = weight_precision
        
        # Assembly structures
        self.atomic_library = set()  # All unique molecules discovered
        self.lattice_library = {}    # All lattice structures by ID
        self.assembly_pathways = {}  # How lattices are assembled
        self.assembly_indices = {}   # Assembly complexity of each lattice
        
        # Tracking
        self.layer_lattices = defaultdict(list)  # Lattices per layer over time
        self.epoch_data = []
        
        # Reuse tracking
        self.molecule_reuse = defaultdict(set)  # Which lattices use each molecule
        self.lattice_reuse = defaultdict(list)  # Temporal reuse of lattices
        
        # Define atomic symbols based on weight magnitudes
        self.atomic_symbols = {
            'Ze': (0.0, 0.01),      # Zero-like
            'Sm': (0.01, 0.1),      # Small
            'Md': (0.1, 0.5),       # Medium
            'Lg': (0.5, 1.0),       # Large
            'Xl': (1.0, 2.0),       # Extra Large
            'Xx': (2.0, float('inf'))  # Extreme
        }

    def compute_assembly_gradient_modifier(self, lattice: MolecularLattice, original_grad: torch.Tensor): # -> torch.Tensor:
        """Modify gradients based on molecular assembly complexity"""
        modifier = torch.ones_like(original_grad)
        
        # Get assembly properties
        assembly_index = self.calculate_assembly_index(lattice)
        
        # Convert lattice back to tensor positions
        for i, molecule_row in enumerate(lattice.molecules):
            for j, molecule in enumerate(molecule_row):
                # Calculate position in original tensor
                start_row = i * self.molecule_size
                end_row = start_row + self.molecule_size
                start_col = j * self.molecule_size
                end_col = start_col + self.molecule_size
                
                # Assembly-based modification
                if len(self.molecule_reuse[molecule]) > 1:
                    # Highly reused molecules - reduce gradient to preserve
                    modifier[start_row:end_row, start_col:end_col] *= 0.8
                elif molecule.atomic_symbol.startswith('Ze'):
                    # Zero-like weights - allow larger updates
                    modifier[start_row:end_row, start_col:end_col] *= 1.2
                elif assembly_index > 5:
                    # Complex assemblies - more conservative updates
                    modifier[start_row:end_row, start_col:end_col] *= 0.9
        
        return modifier

    def suggest_architecture_changes(self, model: nn.Module, complexity_threshold: int = 10): # -> Dict[str, str]:
        """Suggest architectural changes based on molecular complexity"""
        suggestions = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                # Get latest lattice for this layer
                if name in self.layer_lattices:
                    latest_lattice_id = self.layer_lattices[name][-1]
                    assembly_index = self.assembly_indices[latest_lattice_id]
                    
                    if assembly_index > complexity_threshold:
                        suggestions[name] = f"Consider splitting layer - complexity {assembly_index}"
                    elif assembly_index < 3:
                        suggestions[name] = f"Consider merging - low complexity {assembly_index}"
        
        return suggestions
    
    def weight_to_atomic_symbol(self, weight: float): # -> str:
        """Convert weight magnitude to atomic symbol"""
        abs_weight = abs(weight)
        
        for symbol, (min_val, max_val) in self.atomic_symbols.items():
            if min_val <= abs_weight < max_val:
                return symbol + ('+' if weight >= 0 else '-')
        
        return 'Xx' + ('+' if weight >= 0 else '-')
    
    def tensor_to_molecular_lattice(self, weight_tensor: torch.Tensor, layer_name: str, epoch: int): # -> MolecularLattice:
        """Convert weight tensor to molecular lattice structure"""
        if len(weight_tensor.shape) != 2:
            raise ValueError("Only 2D weight tensors supported")
        
        weights = weight_tensor.detach().cpu().numpy()
        rows, cols = weights.shape
        
        # Calculate lattice dimensions
        lattice_rows = rows // self.molecule_size
        lattice_cols = cols // self.molecule_size
        
        molecules = []
        
        for i in range(lattice_rows):
            molecule_row = []
            for j in range(lattice_cols):
                # Extract nxn region
                start_row = i * self.molecule_size
                end_row = start_row + self.molecule_size
                start_col = j * self.molecule_size
                end_col = start_col + self.molecule_size
                
                region = weights[start_row:end_row, start_col:end_col]
                
                # Calculate molecular properties
                atomic_weight = np.mean(region)
                atomic_symbol = self.weight_to_atomic_symbol(atomic_weight)
                
                molecule = WeightMolecule(
                    atomic_weight=round(atomic_weight, self.weight_precision),
                    atomic_symbol=atomic_symbol,
                    position=(i, j),
                    size=(self.molecule_size, self.molecule_size)
                )
                
                molecule_row.append(molecule)
                self.atomic_library.add(molecule)
            
            molecules.append(molecule_row)
        
        return MolecularLattice(
            molecules=molecules,
            layer_name=layer_name,
            epoch=epoch,
            lattice_id=None
        )
    
    def find_assembly_pathway(self, target_lattice: MolecularLattice): # -> Optional[List[str]]:
        """Find how target lattice can be assembled from existing components"""
        target_molecules = set()
        for row in target_lattice.molecules:
            for mol in row:
                target_molecules.add(mol)
        
        # Find existing lattices that could contribute molecules
        pathway = []
        remaining_molecules = target_molecules.copy()
        
        # Sort available lattices by how many molecules they can contribute
        available_lattices = list(self.lattice_library.values())
        lattice_contributions = []
        
        for lattice in available_lattices:
            if lattice.lattice_id == target_lattice.lattice_id:
                continue
            
            lattice_molecules = set()
            for row in lattice.molecules:
                for mol in row:
                    lattice_molecules.add(mol)
            
            contribution = len(lattice_molecules & remaining_molecules)
            if contribution > 0:
                lattice_contributions.append((lattice, contribution, lattice_molecules))
        
        # Greedily select lattices that contribute most molecules
        lattice_contributions.sort(key=lambda x: x[1], reverse=True)
        
        for lattice, contribution, lattice_molecules in lattice_contributions:
            if remaining_molecules & lattice_molecules:
                pathway.append(f"reuse_lattice({lattice.lattice_id})")
                remaining_molecules -= lattice_molecules
            
            if not remaining_molecules:
                break
        
        # Add any remaining molecules as atomic components
        for mol in remaining_molecules:
            pathway.append(f"atomic_molecule({mol.atomic_symbol})")
        
        return pathway if pathway else None

    def calculate_assembly_index(self, lattice: MolecularLattice): # -> int:
        """Calculate assembly complexity of a lattice"""
        if lattice.lattice_id in self.assembly_indices:
            return self.assembly_indices[lattice.lattice_id]
        
        # Get assembly pathway
        pathway = self.find_assembly_pathway(lattice)
        
        if not pathway:
            # New lattice with no reusable components
            unique_molecules = set()
            for row in lattice.molecules:
                for mol in row:
                    unique_molecules.add(mol)
            assembly_index = len(unique_molecules)
        else:
            # Count reused components and atomic additions
            reused_lattices = sum(1 for step in pathway if step.startswith('reuse_lattice'))
            atomic_additions = sum(1 for step in pathway if step.startswith('atomic_molecule'))
            
            # Assembly index = number of assembly steps
            assembly_index = reused_lattices + atomic_additions
        
        self.assembly_indices[lattice.lattice_id] = assembly_index
        return assembly_index
    
    def track_molecule_reuse(self, lattice: MolecularLattice):
        """Track which molecules are reused across lattices"""
        for row in lattice.molecules:
            for mol in row:
                self.molecule_reuse[mol].add(lattice.lattice_id)
    
    def track_epoch(self, model: nn.Module, epoch: int):
        """Track lattice structures for all layers in an epoch"""
        epoch_data = {
            'epoch': epoch,
            'layer_lattices': {},
            'new_lattices': [],
            'assembly_stats': {}
        }
        
        assembly_indices = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                # Convert to molecular lattice
                lattice = self.tensor_to_molecular_lattice(param, name, epoch)
                
                # Store in library
                self.lattice_library[lattice.lattice_id] = lattice
                
                # Find assembly pathway
                pathway = self.find_assembly_pathway(lattice)
                if pathway:
                    self.assembly_pathways[lattice.lattice_id] = pathway
                
                # Calculate assembly index
                assembly_index = self.calculate_assembly_index(lattice)
                assembly_indices.append(assembly_index)
                
                # Track reuse
                self.track_molecule_reuse(lattice)
                self.layer_lattices[name].append(lattice.lattice_id)
                self.lattice_reuse[lattice.lattice_id].append(epoch)
                
                # Store epoch data
                epoch_data['layer_lattices'][name] = {
                    'lattice_id': lattice.lattice_id,
                    'molecular_formula': lattice.get_molecular_formula(),
                    'assembly_index': assembly_index,
                    'pathway': pathway
                }
                
                epoch_data['new_lattices'].append({
                    'layer': name,
                    'lattice_id': lattice.lattice_id,
                    'molecular_formula': lattice.get_molecular_formula(),
                    'assembly_index': assembly_index
                })
        
        # Calculate epoch statistics
        epoch_data['assembly_stats'] = {
            'total_lattices': len(epoch_data['new_lattices']),
            'avg_assembly_index': np.mean(assembly_indices) if assembly_indices else 0,
            'max_assembly_index': max(assembly_indices) if assembly_indices else 0,
            'total_molecules': len(self.atomic_library),
            'total_lattices_library': len(self.lattice_library)
        }
        
        self.epoch_data.append(epoch_data)
        
        print(f"Epoch {epoch}: {len(epoch_data['new_lattices'])} lattices, "
              f"{len(self.atomic_library)} unique molecules discovered")
    
    # ========== ANALYSIS METHODS =============
    # MOVE BELOW METHODS TO ANALYSIS CLASS
    # ==============================================
    
    def print_molecular_analysis(self):
        """Print comprehensive molecular assembly analysis"""
        print("=== MOLECULAR LATTICE ASSEMBLY ANALYSIS ===")
        print(f"Total unique molecules discovered: {len(self.atomic_library)}")
        print(f"Total lattice structures: {len(self.lattice_library)}")
        print(f"Lattices with assembly pathways: {len(self.assembly_pathways)}")
        
        # Show molecular library
        print(f"\nMolecular library (by atomic symbol):")
        molecules_by_symbol = defaultdict(list)
        for mol in self.atomic_library:
            molecules_by_symbol[mol.atomic_symbol].append(mol)
        
        for symbol, molecules in sorted(molecules_by_symbol.items()):
            print(f"  {symbol}: {len(molecules)} variants")
        
        # Most complex lattices - FIX: Add safety check
        print(f"\n=== MOST COMPLEX LATTICES ===")
        complex_lattices = sorted(self.assembly_indices.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        
        for lattice_id, assembly_index in complex_lattices:
            # FIX: Check if lattice exists in library before accessing
            if lattice_id in self.lattice_library:
                lattice = self.lattice_library[lattice_id]
                print(f"Lattice {lattice_id}: Assembly Index {assembly_index}")
                print(f"  Layer: {lattice.layer_name}, Epoch: {lattice.epoch}")
                print(f"  Molecular Formula: {lattice.get_molecular_formula()}")
                
                if lattice_id in self.assembly_pathways:
                    pathway = self.assembly_pathways[lattice_id]
                    print(f"  Assembly pathway: {' → '.join(pathway)}")
                print()
            else:
                # FIX: Handle missing lattice gracefully
                print(f"Lattice {lattice_id}: Assembly Index {assembly_index}")
                print(f"  WARNING: Lattice details not found in library")
                print(f"  This may indicate a synchronization issue between assembly_indices and lattice_library")
                print()
        
        # Reuse analysis
        print("=== MOLECULAR REUSE ANALYSIS ===")
        highly_reused_molecules = {mol: lattices for mol, lattices in self.molecule_reuse.items()
                                if len(lattices) > 1}
        print(f"Molecules reused across lattices: {len(highly_reused_molecules)}")
        
        temporally_reused_lattices = {lid: epochs for lid, epochs in self.lattice_reuse.items()
                                    if len(epochs) > 1}
        print(f"Lattices reused across epochs: {len(temporally_reused_lattices)}")
        
        # Show most reused molecules
        if highly_reused_molecules:
            print(f"\nMost reused molecules:")
            for mol, lattices in sorted(highly_reused_molecules.items(), 
                                    key=lambda x: len(x[1]), reverse=True)[:5]:
                print(f"  {mol}: used in {len(lattices)} lattices")
        
        # Debug information
        print(f"\n=== DEBUG INFORMATION ===")
        print(f"Assembly indices count: {len(self.assembly_indices)}")
        print(f"Lattice library count: {len(self.lattice_library)}")
        print(f"Missing lattices: {set(self.assembly_indices.keys()) - set(self.lattice_library.keys())}")
    
    def visualize_lattice_evolution(self):
        """Visualize how lattice structures evolve"""
        import matplotlib.pyplot as plt
        
        epochs = [data['epoch'] for data in self.epoch_data]
        total_molecules = [data['assembly_stats']['total_molecules'] for data in self.epoch_data]
        avg_complexity = [data['assembly_stats']['avg_assembly_index'] for data in self.epoch_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot molecular discovery
        ax1.plot(epochs, total_molecules, 'b-o', label='Total Unique Molecules')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Unique Molecules Discovered')
        ax1.set_title('Molecular Discovery Over Time')
        ax1.grid(True)
        ax1.legend()
        
        # Plot assembly complexity
        ax2.plot(epochs, avg_complexity, 'r-o', label='Average Assembly Index')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Assembly Complexity')
        ax2.set_title('Lattice Assembly Complexity Evolution')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()