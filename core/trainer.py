import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# self made imports
from core.assembly import MolecularAssemblyTracker
from core.NNmodel import MolecularNeuralNet

# Basic train method
def train_basic(model, X_train, y_train, X_test, y_test, 
                                epochs=100, lr=0.001, track_every=10,
                                molecule_size=2):
    
    """Train model with molecular lattice assembly tracking"""
    
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
    
    print("Starting molecular assembly tracking training...")
    
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            accuracy = (test_predictions == y_test_tensor).float().mean()
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
        # Track molecular assembly
        if epoch % track_every == 0:
            tracker.track_epoch(model, epoch)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy.item():.4f}")
    
    return model, tracker, losses, accuracies

# Train method with loss update on assembly (rewarding complexity)
def train_reward_complexity(model, X_train, y_train, X_test, y_test, 
                                         epochs=100, lr=0.001, track_every=10,
                                         molecule_size=2, 
                                         complexity_reward_weight=0.1,  # Changed from penalty to reward
                                         max_beneficial_complexity=15):  # Cap for beneficial complexity
    """Train model with molecular lattice assembly tracking and complexity rewards"""
    
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
        optimizer.step()
        
        # Apply assembly-guided gradient modifications
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2 and param.grad is not None:
                    # Get current lattice
                    lattice = tracker.tensor_to_molecular_lattice(param, name, epoch)
                    
                    # Get gradient modifier based on assembly
                    grad_modifier = tracker.compute_assembly_gradient_modifier(lattice, param.grad)
                    
                    # Apply modifier to gradients
                    param.grad *= grad_modifier
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            accuracy = (test_predictions == y_test_tensor).float().mean()
        
        losses.append(final_loss.item())
        accuracies.append(accuracy.item())
        complexity_rewards.append(avg_complexity_reward)
        
        # Track molecular assembly
        if epoch % track_every == 0:
            tracker.track_epoch(model, epoch)
            
            # Print complexity reward info
            print(f"Epoch {epoch}: Loss={final_loss.item():.4f}, "
                  f"Accuracy={accuracy.item():.4f}, "
                  f"Complexity Reward={avg_complexity_reward}")
    
    return model, tracker, losses, accuracies, complexity_rewards

# ===================================================
# =========== EVOLUTION TRAINING ================
# ===================================================

def evolve_architecture_with_molecules(X_train: torch.Tensor, hidden_layer_sizes: list, model: nn.Module, 
                                     tracker: MolecularAssemblyTracker,
                                     complexity_threshold: int = 12): # -> nn.Module:
    """Evolve model architecture based on molecular assembly complexity"""

    # Get current architecture info
    input_size = X_train.shape[1]  # Use the actual input size
    current_hidden_sizes = hidden_layer_sizes  # From your original model

    # Analyze current complexity
    if not tracker.assembly_indices:
        print("No assembly data available, keeping current architecture")
        return model
    
    max_complexity = max(tracker.assembly_indices.values())
    suggestions = tracker.suggest_architecture_changes(model, complexity_threshold)
    
    print(f"Current max complexity: {max_complexity}")
    print(f"Architecture suggestions: {suggestions}")
    
    # Determine new architecture
    if max_complexity > 15:
        # Add more layers for high complexity
        new_hidden_sizes = current_hidden_sizes + [current_hidden_sizes[-1] // 2]
        print(f"Adding layer: {current_hidden_sizes} -> {new_hidden_sizes}")
    elif max_complexity > 12:
        # Expand existing layers
        new_hidden_sizes = [int(size * 1.2) for size in current_hidden_sizes]
        print(f"Expanding layers: {current_hidden_sizes} -> {new_hidden_sizes}")
    else:
        # Keep current architecture
        new_hidden_sizes = current_hidden_sizes
        print("No architecture changes needed")
    
    # Create evolved model with proper structure
    evolved_model = MolecularNeuralNet(
        input_size=input_size,
        hidden_sizes=new_hidden_sizes,
        output_size=1,
        dropout=0.2
    )
    
    # Transfer weights from original model where possible
    try:
        with torch.no_grad():
            # Copy compatible layer weights
            original_params = dict(model.named_parameters())
            for name, param in evolved_model.named_parameters():
                if name in original_params:
                    orig_param = original_params[name]
                    # Only copy if shapes match
                    if param.shape == orig_param.shape:
                        param.data.copy_(orig_param.data)
                        print(f"Transferred weights for {name}")
    except Exception as e:
        print(f"Weight transfer failed: {e}")
        print("Using random initialization for evolved model")
    
    return evolved_model


def evolve_architecture_with_complexity_rewards(X_train: torch.Tensor, hidden_layer_sizes: list, model: nn.Module, 
                                               tracker: MolecularAssemblyTracker,
                                               complexity_target: int = 10): # -> nn.Module:
    """Evolve model architecture to achieve target complexity levels"""
    
    # Get current architecture info
    input_size = X_train.shape[1]  # Use the actual input size
    current_hidden_sizes = hidden_layer_sizes  # From your original model
    
    # Analyze current complexity
    if not tracker.assembly_indices:
        print("No assembly data available, keeping current architecture")
        return model
    
    avg_complexity = np.mean(list(tracker.assembly_indices.values()))
    max_complexity = max(tracker.assembly_indices.values())
    
    print(f"Current avg complexity: {avg_complexity:.2f}, max: {max_complexity}")
    
    # Evolve based on complexity goals
    if avg_complexity < complexity_target:
        # Increase complexity by adding layers or expanding existing ones
        if max_complexity < complexity_target * 0.7:
            # Add more layers
            new_hidden_sizes = current_hidden_sizes + [current_hidden_sizes[-1] // 2]
            print(f"Adding layer for complexity: {current_hidden_sizes} -> {new_hidden_sizes}")
        else:
            # Expand existing layers
            new_hidden_sizes = [int(size * 1.5) for size in current_hidden_sizes]
            print(f"Expanding layers for complexity: {current_hidden_sizes} -> {new_hidden_sizes}")
    elif avg_complexity > complexity_target * 1.5:
        # Reduce complexity if it's too high
        new_hidden_sizes = [max(4, int(size * 0.8)) for size in current_hidden_sizes]
        print(f"Reducing complexity: {current_hidden_sizes} -> {new_hidden_sizes}")
    else:
        # Complexity is in good range
        new_hidden_sizes = current_hidden_sizes
        print("Complexity in target range, no changes needed")
    
    # Create evolved model
    evolved_model = MolecularNeuralNet(
        input_size=input_size,
        hidden_sizes=new_hidden_sizes,
        output_size=1,
        dropout=0.2
    )
    
    # Transfer weights where possible
    try:
        with torch.no_grad():
            original_params = dict(model.named_parameters())
            for name, param in evolved_model.named_parameters():
                if name in original_params:
                    orig_param = original_params[name]
                    if param.shape == orig_param.shape:
                        param.data.copy_(orig_param.data)
                        print(f"Transferred weights for {name}")
    except Exception as e:
        print(f"Weight transfer failed: {e}")
    
    return evolved_model

# ===================================================
# =========== MAIN TRAINING METHOD ============
# ===================================================

def train_with_molecular_parameter_updates(model, X_train, y_train, X_test, y_test, 
                                         epochs=100, lr=0.001, track_every=10,
                                         molecule_size=2, 
                                         complexity_range=(3, 8)):
    """Train with appropriate complexity management - supports both binary and multi-class"""
    
    # Initialize tracker
    tracker = MolecularAssemblyTracker(molecule_size=molecule_size)
    
    # Data conversion and setup
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Handle different target formats
    if len(y_train.shape) == 1:
        # Binary classification: convert to proper shape
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
        is_binary = True
    else:
        # Multi-class classification: use categorical targets
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)
        criterion = nn.CrossEntropyLoss()
        is_binary = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accuracies = []
    complexities = []
    
    print(f"Starting complexity-managed training ({'binary' if is_binary else 'multi-class'})...")
    
    for epoch in range(epochs):
        # Track molecular assembly
        if epoch % track_every == 0 or epoch == 0:
            tracker.track_epoch(model, epoch)
            
            # Calculate current complexity
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
            
            # Adjust learning based on complexity
            complexity_modifier = 1.0
            if current_complexity < complexity_range[0]:
                complexity_modifier = 1.2
            elif current_complexity > complexity_range[1]:
                complexity_modifier = 0.8
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * complexity_modifier
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        
        if is_binary:
            loss = criterion(outputs, y_train_tensor)
        else:
            # For multi-class, convert one-hot to class indices
            targets = torch.argmax(y_train_tensor, dim=1)
            loss = criterion(outputs, targets)
        
        loss.backward()
        
        # Apply guided updates based on molecular structure
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2 and param.grad is not None:
                    lattice = tracker.tensor_to_molecular_lattice(param, name, epoch)
                    
                    # Encourage reuse of effective patterns
                    for i, molecule_row in enumerate(lattice.molecules):
                        for j, molecule in enumerate(molecule_row):
                            if len(tracker.molecule_reuse.get(molecule, [])) > 1:
                                start_i = i * tracker.molecule_size
                                end_i = start_i + tracker.molecule_size
                                start_j = j * tracker.molecule_size
                                end_j = start_j + tracker.molecule_size
                                
                                param.grad[start_i:end_i, start_j:end_j] *= 0.9
        
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            
            if is_binary:
                test_predictions = (torch.sigmoid(test_outputs) > 0.5).float()
                accuracy = (test_predictions == y_test_tensor).float().mean().item()
            else:
                test_predictions = torch.argmax(test_outputs, dim=1)
                test_targets = torch.argmax(y_test_tensor, dim=1)
                accuracy = (test_predictions == test_targets).float().mean().item()
            
            accuracies.append(accuracy)
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            complexity_msg = f", Complexity: {current_complexity:.2f}" if epoch % track_every == 0 else ""
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}{complexity_msg}")
    
    return model, tracker, losses, accuracies, complexities


# Other train methods with molecular assembly updates
# def train_with_molecular_parameter_updates(model, X_train, y_train, X_test, y_test, 
#                                          epochs=100, lr=0.001, track_every=5,
#                                          molecule_size=2, assembly_weight=0.1):
#     """Train with molecular assembly-guided parameter updates"""
    
#     tracker = MolecularAssemblyTracker(molecule_size=molecule_size)
    
#     # Convert to tensors
#     X_train_tensor = torch.FloatTensor(X_train)
#     y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
#     X_test_tensor = torch.FloatTensor(X_test)
#     y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCELoss()
    
#     losses = []
#     accuracies = []
#     assembly_losses = []
    
#     print("Starting molecular assembly-guided training...")
    
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
        
#         # Standard forward pass
#         outputs = model(X_train_tensor)
#         base_loss = criterion(outputs, y_train_tensor)
        
#         # Track molecular assembly
#         if epoch % track_every == 0:
#             tracker.track_epoch(model, epoch)
        
#         # Compute assembly complexity penalty
#         assembly_penalty = 0.0
#         lattice_data = {}
        
#         for name, param in model.named_parameters():
#             if 'weight' in name and len(param.shape) == 2:
#                 # Convert to molecular lattice
#                 lattice = tracker.tensor_to_molecular_lattice(param, name, epoch)
#                 lattice_data[name] = lattice
                
#                 # Penalize excessive complexity
#                 assembly_index = tracker.calculate_assembly_index(lattice)
#                 if assembly_index > 8:  # Complexity threshold
#                     assembly_penalty += (assembly_index - 8) * 0.01
        
#         # Combined loss
#         total_loss = base_loss + assembly_weight * assembly_penalty
#         total_loss.backward()
        
#         # Modify gradients based on molecular structure
#         for name, param in model.named_parameters():
#             if name in lattice_data and param.grad is not None:
#                 lattice = lattice_data[name]
#                 grad_modifier = tracker.compute_assembly_gradient_modifier(lattice, param.grad)
#                 param.grad.data *= grad_modifier
        
#         optimizer.step()
        
#         # Post-optimization molecular adjustments
#         if epoch % (track_every * 2) == 0:
#             apply_molecular_constraints(model, tracker)
        
#         # Evaluation
#         model.eval()
#         with torch.no_grad():
#             test_outputs = model(X_test_tensor)
#             test_predictions = (test_outputs > 0.5).float()
#             accuracy = (test_predictions == y_test_tensor).float().mean()
        
#         losses.append(base_loss.item())
#         assembly_losses.append(assembly_penalty)
#         accuracies.append(accuracy.item())
        
#         if epoch % 20 == 0:
#             print(f"Epoch {epoch}: Loss={base_loss.item():.4f}, "
#                   f"Assembly Penalty={assembly_penalty:.4f}, "
#                   f"Accuracy={accuracy.item():.4f}")
    
#     return model, tracker, losses, accuracies, assembly_losses

def apply_molecular_constraints(model: nn.Module, tracker: MolecularAssemblyTracker):
    """Apply molecular-based constraints to model parameters"""
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            lattice = tracker.tensor_to_molecular_lattice(param, name, 0)
            
            with torch.no_grad():
                # Stabilize highly reused molecules
                for i, molecule_row in enumerate(lattice.molecules):
                    for j, molecule in enumerate(molecule_row):
                        if len(tracker.molecule_reuse[molecule]) > 2:
                            # Reduce variance in highly reused molecular regions
                            start_row = i * tracker.molecule_size
                            end_row = start_row + tracker.molecule_size
                            start_col = j * tracker.molecule_size
                            end_col = start_col + tracker.molecule_size
                            
                            region = param[start_row:end_row, start_col:end_col]
                            mean_val = region.mean()
                            param[start_row:end_row, start_col:end_col] = \
                                0.8 * region + 0.2 * mean_val