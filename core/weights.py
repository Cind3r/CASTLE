import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hashlib
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class WeightTracker:
    
    """
    Tracks weight patterns across epochs by binning weights into discrete codes
    and analyzing how these 'building blocks' evolve and interact.
    """
    
    def __init__(self, n_bins=10, bin_method='quantile'):
        self.n_bins = n_bins
        self.bin_method = bin_method  # 'quantile' or 'uniform'
        self.layer_histories = {}
        self.atomic_codes = {}
        self.epoch_snapshots = []
        self.building_blocks = defaultdict(list)
        
    def create_atomic_code(self, binned_weights, layer_name):
        """Create a unique atomic code for a weight configuration"""
        # Create a hash of the binned weight pattern
        weight_str = ''.join(map(str, binned_weights.flatten()))
        hash_obj = hashlib.md5(weight_str.encode())
        atomic_code = hash_obj.hexdigest()[:8]  # Short hash for readability
        
        return f"{layer_name}_{atomic_code}"
    
    def bin_weights(self, weights):
        """Bin weights into discrete categories"""
        weights_flat = weights.flatten()
        
        if self.bin_method == 'quantile':
            # Use quantile-based binning
            bins = np.quantile(weights_flat, np.linspace(0, 1, self.n_bins + 1))
            bins = np.unique(bins)  # Remove duplicates
            if len(bins) <= 1:
                return np.zeros_like(weights_flat, dtype=int)
            binned = np.digitize(weights_flat, bins) - 1
        else:
            # Use uniform binning
            bins = np.linspace(weights_flat.min(), weights_flat.max(), self.n_bins + 1)
            binned = np.digitize(weights_flat, bins) - 1
        
        # Ensure bins are within valid range
        binned = np.clip(binned, 0, self.n_bins - 1)
        return binned.reshape(weights.shape)
    
    def track_epoch(self, model, epoch):
        """Track weight patterns for all layers in current epoch"""
        epoch_data = {
            'epoch': epoch,
            'layers': {},
            'building_blocks': {},
            'interactions': {}
        }
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy()
                binned_weights = self.bin_weights(weights)
                atomic_code = self.create_atomic_code(binned_weights, name)
                
                # Store layer information
                epoch_data['layers'][name] = {
                    'weights': weights,
                    'binned_weights': binned_weights,
                    'atomic_code': atomic_code,
                    'weight_stats': {
                        'mean': weights.mean(),
                        'std': weights.std(),
                        'min': weights.min(),
                        'max': weights.max()
                    }
                }
                
                # Track building blocks
                if atomic_code not in self.building_blocks:
                    self.building_blocks[atomic_code] = []
                self.building_blocks[atomic_code].append(epoch)
                
                # Store in layer history
                if name not in self.layer_histories:
                    self.layer_histories[name] = []
                self.layer_histories[name].append({
                    'epoch': epoch,
                    'atomic_code': atomic_code,
                    'binned_weights': binned_weights.copy()
                })
        
        # Analyze layer interactions (simplified correlation between atomic codes)
        layer_names = list(epoch_data['layers'].keys())
        for i, layer1 in enumerate(layer_names):
            for layer2 in layer_names[i+1:]:
                # Calculate similarity between binned weight patterns
                bins1 = epoch_data['layers'][layer1]['binned_weights'].flatten()
                bins2 = epoch_data['layers'][layer2]['binned_weights'].flatten()
                
                # Resize to same length for comparison
                min_len = min(len(bins1), len(bins2))
                correlation = np.corrcoef(bins1[:min_len], bins2[:min_len])[0, 1]
                
                interaction_key = f"{layer1}_{layer2}"
                epoch_data['interactions'][interaction_key] = {
                    'correlation': correlation if not np.isnan(correlation) else 0,
                    'layer1_code': epoch_data['layers'][layer1]['atomic_code'],
                    'layer2_code': epoch_data['layers'][layer2]['atomic_code']
                }
        
        self.epoch_snapshots.append(epoch_data)
    
    def get_building_block_evolution(self):
        """Analyze how building blocks evolve across epochs"""
        evolution = {}
        
        for layer_name, history in self.layer_histories.items():
            evolution[layer_name] = {
                'codes_per_epoch': [entry['atomic_code'] for entry in history],
                'unique_codes': len(set(entry['atomic_code'] for entry in history)),
                'code_changes': []
            }
            
            # Track when codes change
            for i in range(1, len(history)):
                if history[i]['atomic_code'] != history[i-1]['atomic_code']:
                    evolution[layer_name]['code_changes'].append(i)
        
        return evolution
    
    def visualize_weight_evolution(self, layer_name, max_epochs=None):
        """Visualize how weights evolve in a specific layer"""
        if layer_name not in self.layer_histories:
            print(f"Layer {layer_name} not found in history")
            return
        
        history = self.layer_histories[layer_name]
        if max_epochs:
            history = history[:max_epochs]
        
        epochs = [entry['epoch'] for entry in history]
        codes = [entry['atomic_code'] for entry in history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot atomic code changes
        unique_codes = list(set(codes))
        code_to_num = {code: i for i, code in enumerate(unique_codes)}
        code_nums = [code_to_num[code] for code in codes]
        
        ax1.plot(epochs, code_nums, 'o-', linewidth=2, markersize=6)
        ax1.set_ylabel('Atomic Code ID')
        ax1.set_title(f'Weight Pattern Evolution - {layer_name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot weight statistics over time
        means = [np.mean(self.epoch_snapshots[i]['layers'][layer_name]['weights']) 
                for i in range(len(history))]
        stds = [np.std(self.epoch_snapshots[i]['layers'][layer_name]['weights']) 
               for i in range(len(history))]
        
        ax2.plot(epochs, means, 'b-', label='Mean', linewidth=2)
        ax2.plot(epochs, stds, 'r-', label='Std', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Weight Value')
        ax2.set_title(f'Weight Statistics - {layer_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_interaction_matrix(self, epoch):
        """Get layer interaction matrix for a specific epoch"""
        if epoch >= len(self.epoch_snapshots):
            print(f"Epoch {epoch} not found in snapshots")
            return None
        
        snapshot = self.epoch_snapshots[epoch]
        interactions = snapshot['interactions']
        
        # Create interaction matrix
        layer_names = list(snapshot['layers'].keys())
        n_layers = len(layer_names)
        interaction_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i != j:
                    key1 = f"{layer1}_{layer2}"
                    key2 = f"{layer2}_{layer1}"
                    
                    if key1 in interactions:
                        interaction_matrix[i, j] = interactions[key1]['correlation']
                    elif key2 in interactions:
                        interaction_matrix[i, j] = interactions[key2]['correlation']
        
        return interaction_matrix, layer_names
