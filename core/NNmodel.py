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
from typing import Dict, List, Tuple, Optional

# self made imports
from core.weights import WeightTracker
from core.assembly import MolecularAssemblyTracker

class MolecularNeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.1):
        super().__init__()
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Don't add dropout to output layer
                self.layers.append(nn.Dropout(dropout))

        # Initialize weights to create molecular patterns
        self._initialize_weights()
    

    def initialize_with_molecular_structure(layer: nn.Linear, tracker: MolecularAssemblyTracker, reference_layer_name: str):
        """Initialize layer weights based on molecular patterns from reference layer"""
        
        if reference_layer_name in tracker.layer_lattices:
            # Get the most recent lattice structure
            latest_lattice_id = tracker.layer_lattices[reference_layer_name][-1]
            reference_lattice = tracker.lattice_library[latest_lattice_id]
            
            # Initialize weights to preserve successful molecular patterns
            with torch.no_grad():
                for i, molecule_row in enumerate(reference_lattice.molecules):
                    for j, molecule in enumerate(molecule_row):
                        if len(tracker.molecule_reuse[molecule]) > 1:
                            # This molecule was successful - use its pattern
                            target_weight = molecule.atomic_weight
                            
                            # Apply to corresponding region in new layer
                            if i < layer.weight.shape[0] and j < layer.weight.shape[1]:
                                layer.weight[i, j] = target_weight
    
    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize with weights
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear) and layer != self.layers[-1]:
                x = torch.relu(x)
        return torch.sigmoid(x)


# ========================================================
# ========== Neural Network with Weight Tracking ==========
# ========================================================

class WeightNeuralNet(nn.Module):
    """Neural network with built-in weight tracking capabilities. Only meant to be used with the WeightTracker class in weights.py."""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        super(WeightNeuralNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # For binary classification
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def get_layer_weights(self):
        """Extract weights from all layers"""
        weights = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                weights[name] = param.data.cpu().numpy()
        return weights

# Training function with tracking
def train_with_tracking(model, X_train, y_train, X_test, y_test, 
                       epochs=100, lr=0.01, track_every=5):
    """Train model while tracking weight evolution"""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Initialize tracker
    tracker = WeightTracker(n_bins=8, bin_method='quantile')
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training history
    train_losses = []
    test_accuracies = []
    
    print("Starting training with weight tracking...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions == y_test_tensor).float().mean().item()
            test_accuracies.append(test_accuracy)
        
        # Track weights at specified intervals
        if epoch % track_every == 0 or epoch == epochs - 1:
            tracker.track_epoch(model, epoch)
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test Accuracy={test_accuracy:.4f}")
    
    return model, tracker, train_losses, test_accuracies

