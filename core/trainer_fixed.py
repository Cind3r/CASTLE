import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# self made imports
from core.assembly import MolecularAssemblyTracker
from core.NNmodel import MolecularNeuralNet

def train_with_molecular_parameter_updates(model, X_train, y_train, X_test, y_test, 
                                         epochs=100, lr=0.001, track_every=10,
                                         molecule_size=2, 
                                         complexity_reward_weight=0.05,  # Weight for complexity reward
                                         max_beneficial_complexity=12):  # Cap for beneficial complexity
    """Train with molecular lattice assembly tracking and complexity rewards"""
    
    # Initialize tracker
    tracker = MolecularAssemblyTracker(molecule_size=molecule_size)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    losses = []
    accuracies = []
    complexities = []
    complexity_rewards = []
    
    print("Starting molecular assembly tracking training with complexity rewards...")
    
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        base_loss = criterion(outputs, y_train_tensor)
        
        # REWARD ASSEMBLY COMPLEXITY INSTEAD OF PENALIZING IT
        total_complexity_reward = 0
        complexity_count = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                # Convert to molecular lattice
                lattice = tracker.tensor_to_molecular_lattice(param, name, epoch)
                
                # Calculate assembly complexity
                assembly_index = tracker.calculate_assembly_index(lattice)
                
                # REWARD COMPLEXITY (instead of penalty)
                if assembly_index <= max_beneficial_complexity:
                    # Linear reward for beneficial complexity
                    complexity_reward = assembly_index * complexity_reward_weight
                else:
                    # Diminishing returns for very high complexity
                    excess = assembly_index - max_beneficial_complexity
                    complexity_reward = (max_beneficial_complexity * complexity_reward_weight + 
                                       excess * complexity_reward_weight * 0.1)
                
                total_complexity_reward += complexity_reward
                complexity_count += 1
        
        # Average the complexity reward
        avg_complexity_reward = total_complexity_reward / max(complexity_count, 1)
        
        # SUBTRACT the reward from loss (lower loss = better performance)
        final_loss = base_loss - avg_complexity_reward
        
        # Ensure loss doesn't go negative (optional safety check)
        final_loss = torch.clamp(final_loss, min=0.001)
        
        final_loss.backward()
        
        # Apply assembly-guided gradient modifications BEFORE optimizer step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2 and param.grad is not None:
                    # Get current lattice
                    lattice = tracker.tensor_to_molecular_lattice(param, name, epoch)
                    
                    # Get gradient modifier based on assembly
                    grad_modifier = tracker.compute_assembly_gradient_modifier(lattice, param.grad)
                    
                    # Apply modifier to gradients
                    param.grad *= grad_modifier
        
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            accuracy = (test_predictions == y_test_tensor).float().mean()
        
        losses.append(final_loss.item())
        accuracies.append(accuracy.item())
        complexity_rewards.append(avg_complexity_reward)
        
        # Track molecular assembly for complexity calculation
        if epoch % track_every == 0:
            tracker.track_epoch(model, epoch)
            
            # Calculate current complexity for tracking
            current_complexity = 0
            count = 0
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    lattice = tracker.tensor_to_molecular_lattice(param, name, epoch)
                    idx = tracker.calculate_assembly_index(lattice)
                    current_complexity += idx
                    count += 1
            
            if count > 0:
                current_complexity /= count
                complexities.append(current_complexity)
            
            # Print complexity reward info
            print(f"Epoch {epoch}: Loss={final_loss.item():.4f}, "
                  f"Accuracy={accuracy.item():.4f}, "
                  f"Complexity Reward={avg_complexity_reward}")
    
    return model, tracker, losses, accuracies, complexities
