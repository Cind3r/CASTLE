import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def exponential_model(complexity, A_base, gamma, beta):
    """
    Exponential model for accuracy-complexity relationship:
    A(t) = A_base + γ * (1 - exp(-β * A_sys(t)))
    """
    return A_base + gamma * (1 - np.exp(-beta * complexity))

def linear_model(complexity, A_min, delta):
    """
    Linear model for accuracy-complexity relationship:
    A(t) ≈ A_min + δ * A_sys(t)
    """
    return A_min + delta * complexity

def validate_complexity_accuracy_relationship(tracker, accuracies, dataset_name):
    """
    Validate the relationship between complexity and accuracy
    """
    # Extract complexity values from tracker
    epochs = [data['epoch'] for data in tracker.epoch_data]
    complexities = [data['assembly_stats']['avg_assembly_index'] for data in tracker.epoch_data]
    
    # Extract corresponding accuracy values
    # Ensure we have matching data points
    acc_values = [accuracies[epoch] for epoch in epochs if epoch < len(accuracies)]
    complexities = complexities[:len(acc_values)]
    
    # Create a dataframe for analysis
    df = pd.DataFrame({
        'epoch': epochs[:len(acc_values)],
        'complexity': complexities,
        'accuracy': acc_values
    })
    
    # Fit the exponential model
    try:
        params_exp, _ = curve_fit(
            exponential_model, 
            df['complexity'], 
            df['accuracy'],
            bounds=([0, 0, 0], [1, 1, 100]),  # Reasonable bounds for parameters
            maxfev=10000
        )
        A_base, gamma, beta = params_exp
        
        # Calculate fitted values and metrics
        y_pred_exp = exponential_model(df['complexity'], A_base, gamma, beta)
        r2_exp = r2_score(df['accuracy'], y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(df['accuracy'], y_pred_exp))
    except:
        print(f"Could not fit exponential model for {dataset_name}")
        A_base, gamma, beta = None, None, None
        r2_exp, rmse_exp = None, None
        y_pred_exp = np.zeros_like(df['accuracy'])
    
    # Fit the linear model
    try:
        params_lin, _ = curve_fit(
            linear_model, 
            df['complexity'], 
            df['accuracy'],
            bounds=([0, -1], [1, 1])  # Reasonable bounds for parameters
        )
        A_min, delta = params_lin
        
        # Calculate fitted values and metrics
        y_pred_lin = linear_model(df['complexity'], A_min, delta)
        r2_lin = r2_score(df['accuracy'], y_pred_lin)
        rmse_lin = np.sqrt(mean_squared_error(df['accuracy'], y_pred_lin))
    except:
        print(f"Could not fit linear model for {dataset_name}")
        A_min, delta = None, None
        r2_lin, rmse_lin = None, None
        y_pred_lin = np.zeros_like(df['accuracy'])
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Accuracy vs Complexity Scatter
    plt.subplot(2, 2, 1)
    plt.scatter(df['complexity'], df['accuracy'], c=df['epoch'], cmap='viridis', s=50)
    plt.colorbar(label='Epoch')
    plt.xlabel('Assembly Complexity')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name}: Accuracy vs Complexity')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Fitted Models
    plt.subplot(2, 2, 2)
    plt.scatter(df['complexity'], df['accuracy'], alpha=0.6, label='Actual Data')
    
    # Sort for smooth curve plotting
    sorted_idx = np.argsort(df['complexity'])
    sorted_complexity = df['complexity'].values[sorted_idx]
    
    if A_base is not None:
        plt.plot(sorted_complexity, exponential_model(sorted_complexity, A_base, gamma, beta), 
                'r-', linewidth=2, label=f'Exponential Model (R²={r2_exp:.3f})')
    if A_min is not None:
        plt.plot(sorted_complexity, linear_model(sorted_complexity, A_min, delta), 
                'g--', linewidth=2, label=f'Linear Model (R²={r2_lin:.3f})')
    
    plt.xlabel('Assembly Complexity')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name}: Model Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy and Complexity over Epochs
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax1.plot(df['epoch'], df['accuracy'], 'b-', marker='o', linewidth=2, label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['complexity'], 'r-', marker='x', linewidth=2, label='Complexity')
    ax2.set_ylabel('Assembly Complexity', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title(f'{dataset_name}: Accuracy and Complexity Evolution')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results summary
    print(f"\n=== {dataset_name} Complexity-Accuracy Relationship ===")
    
    if A_base is not None:
        print(f"Exponential Model: A(t) = {A_base:.4f} + {gamma:.4f} * (1 - exp(-{beta:.4f} * A_sys(t)))")
        print(f"  R² = {r2_exp:.4f}, RMSE = {rmse_exp:.4f}")
    
    if A_min is not None:
        print(f"Linear Model: A(t) = {A_min:.4f} + {delta:.4f} * A_sys(t)")
        print(f"  R² = {r2_lin:.4f}, RMSE = {rmse_lin:.4f}")
    
    # Return the best model and its metrics
    if r2_exp is not None and r2_lin is not None:
        best_model = "Exponential" if r2_exp > r2_lin else "Linear"
        print(f"Best fitting model: {best_model}")
        return df, best_model, max(r2_exp, r2_lin)
    elif r2_exp is not None:
        print(f"Best fitting model: Exponential")
        return df, "Exponential", r2_exp
    elif r2_lin is not None:
        print(f"Best fitting model: Linear")
        return df, "Linear", r2_lin
    else:
        print("No valid model fit")
        return df, None, None
    
def analyze_complexity_accuracy_relationship(complexities, accuracies, dataset_name):
    """Analyze relationship between complexity and accuracy with better models"""
    
    # Create dataframe for analysis
    if len(complexities) < len(accuracies):
        track_every = len(accuracies) // len(complexities)
        sampled_epochs = list(range(0, len(accuracies), track_every))[:len(complexities)]
        sampled_accuracies = [accuracies[i] for i in sampled_epochs]
        df = pd.DataFrame({
            'complexity': complexities,
            'accuracy': sampled_accuracies,
            'epoch': sampled_epochs
        })
    else:
        # If lengths match or complexities are more frequent, just trim
        min_len = min(len(complexities), len(accuracies))
        df = pd.DataFrame({
            'complexity': complexities[:min_len],
            'accuracy': accuracies[:min_len],
            'epoch': list(range(min_len))
        })
    
    # Test multiple relationship models
    plt.figure(figsize=(12, 10))
    
    # 1. Scatter plot with epoch coloring
    plt.subplot(2, 2, 1)
    plt.scatter(df['complexity'], df['accuracy'], c=df['epoch'], cmap='viridis')
    plt.colorbar(label='Epoch')
    plt.xlabel('Complexity')
    plt.ylabel('Accuracy')
    plt.title('Complexity vs Accuracy (colored by epoch)')
    
    # 2. Linear model
    plt.subplot(2, 2, 2)
    
    # Try quadratic model to capture non-linear relationship
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    X = df['complexity'].values.reshape(-1, 1)
    y = df['accuracy'].values
    
    # Linear fit
    model_lin = LinearRegression()
    model_lin.fit(X, y)
    y_lin = model_lin.predict(X)
    r2_lin = r2_score(y, y_lin)
    
    # Quadratic fit
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_quad = LinearRegression()
    model_quad.fit(X_poly, y)
    y_quad = model_quad.predict(X_poly)
    r2_quad = r2_score(y, y_quad)
    
    # Plot results
    plt.scatter(df['complexity'], df['accuracy'], alpha=0.7)
    
    # Sort for smooth curve plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_lin_sorted = y_lin[sort_idx]
    y_quad_sorted = y_quad[sort_idx]
    
    plt.plot(X_sorted, y_lin_sorted, 'r-', label=f'Linear (R²={r2_lin:.3f})')
    plt.plot(X_sorted, y_quad_sorted, 'g-', label=f'Quadratic (R²={r2_quad:.3f})')
    plt.xlabel('Complexity')
    plt.ylabel('Accuracy')
    plt.title('Model Fitting')
    plt.legend()
    
    # 3. Inverted U-shape test (theory predicts optimal complexity)
    plt.subplot(2, 2, 3)
    
    # Fit an inverted U-shape model: y = a + b*x - c*x²
    def inverted_u(x, a, b, c):
        return a + b*x - c*x**2
    
    try:
        params, _ = curve_fit(inverted_u, df['complexity'], df['accuracy'])
        a, b, c = params
        
        x_range = np.linspace(df['complexity'].min(), df['complexity'].max(), 100)
        y_pred = inverted_u(x_range, a, b, c)
        
        # Calculate optimal complexity
        optimal_x = b / (2*c) if c > 0 else None
        
        plt.scatter(df['complexity'], df['accuracy'], alpha=0.7)
        plt.plot(x_range, y_pred, 'b-', label='Inverted U-model')
        
        if optimal_x is not None and optimal_x > df['complexity'].min() and optimal_x < df['complexity'].max():
            plt.axvline(x=optimal_x, color='k', linestyle='--', alpha=0.5)
            plt.text(optimal_x, df['accuracy'].min(), f'Optimal: {optimal_x:.2f}', 
                     ha='center', va='bottom')
        
        plt.xlabel('Complexity')
        plt.ylabel('Accuracy')
        plt.title('Inverted U-shape Test')
        plt.legend()
    except:
        plt.text(0.5, 0.5, 'Could not fit inverted U-model', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. Time evolution
    plt.subplot(2, 2, 4)
    plt.plot(df['epoch'], df['complexity'], 'r-', label='Complexity')
    plt.plot(df['epoch'], df['accuracy'], 'b-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.title('Evolution over Training')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_complexity_analysis.png')
    plt.show()
    
    # Print analysis
    print(f"\n=== {dataset_name} Complexity-Accuracy Analysis ===")
    print(f"Linear model: accuracy = {model_lin.intercept_:.4f} + {model_lin.coef_[0]:.4f} * complexity")
    print(f"Linear R²: {r2_lin:.4f}")
    
    if r2_quad > r2_lin:
        print(f"Quadratic model better explains the relationship (R²={r2_quad:.4f})")
        b1, b2 = model_quad.coef_[1], model_quad.coef_[2]
        optimal = -b1 / (2*b2) if b2 != 0 else None
        if optimal is not None and optimal > 0:
            print(f"Estimated optimal complexity: {optimal:.2f}")
    
    return df