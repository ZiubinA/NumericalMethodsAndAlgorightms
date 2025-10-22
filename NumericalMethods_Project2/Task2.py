import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = 'data_for_task4.tsv'
try:
    df = pd.read_csv(file_name, sep='\t')
except FileNotFoundError:
    print(f"Error: Could not find the file '{file_name}'.")
    exit()

input_cols = ['LotArea', 'OverallQual', 'YearBuilt']
output_col = ['SalePrice']

df_norm = df.copy()
scalers = {}

for col in input_cols + output_col:
    min_val = df[col].min()
    max_val = df[col].max()
    scalers[col] = {'min': min_val, 'max': max_val}
    df_norm[col] = (df[col] - min_val) / (max_val - min_val)

X_norm = df_norm[input_cols].values
y_norm = df_norm[output_col].values

def predict(W, X):
    W_matrix = W.reshape((3, 3))
    
    # Calculate hidden layer outputs (N4, N5, N6)
    N_hidden = X.dot(W_matrix)
    
    # final prediction (N8)
    y_pred = np.sum(N_hidden, axis=1)
    
    return y_pred.reshape(-1, 1)

def funct_mse(W, X, y_true):
    y_pred = predict(W, X)
    error = y_true - y_pred
    mse = np.mean(error**2)
    return mse

def funct_mae(W, X, y_true):
    y_pred = predict(W, X)
    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    return mae

def grad(W, X, y_true):
    funct_val = funct_mse(W, X, y_true)
    h = 1e-8 
    
    gradient = np.zeros_like(W)
    
    # Calculate partial derivative
    for i in range(len(W)):
        W_h = W.copy()
        W_h[i] += h
        
        funct_val_h = funct_mse(W_h, X, y_true)
        
        gradient[i] = (funct_val_h - funct_val) / h
        
    return gradient

print("Starting optimization")

W_initial = np.array([1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
W = W_initial.copy()

max_iter = 500  
step = 1.0      
tol = 1e-7     

mse_history = []
mae_history = []

for i in range(max_iter):
    # Calculate current errors
    mse_old = funct_mse(W, X_norm, y_norm)
    mae_old = funct_mae(W, X_norm, y_norm)
    mse_history.append(mse_old)
    mae_history.append(mae_old)
    
    if (i % 50 == 0) or (i == max_iter - 1):
        print(f"Iteration {i:4d}: MSE = {mse_old:.6f}, MAE = {mae_old:.6f}")
    
    gradient = grad(W, X_norm, y_norm)
    
    if np.linalg.norm(gradient) < tol:
        print(f"\nConvergence reached at iteration {i}.")
        break
        
    # Find the "fastest" step (Backtracking Line Search)
    temp_step = step
    while True:
        W_new = W - temp_step * gradient
        mse_new = funct_mse(W_new, X_norm, y_norm)
        
        if mse_new < mse_old:
            W = W_new
            step = temp_step * 1.2  
            break
        
        temp_step = temp_step / 2.0
        
        if temp_step < 1e-12:
            W = W_new 
            break
    
    if temp_step < 1e-12:
         print(f"\nStep size too small. Stopping at iteration {i}.")
         break

W_final = W
print("Optimization finished.\n")

print("Report 1: Error vs. Iteration Graphs")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mse_history)
plt.title('Mean Square Error (MSE) vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('MSE (on normalized data)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(mae_history)
plt.title('Mean Absolute Error (MAE) vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('MAE (on normalized data)')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nReport 2: Error Values (Before vs. After)")
error_data = {
    'Metric': ['MSE', 'MAE'],
    'Before Optimization': [mse_history[0], mae_history[0]],
    'After Optimization': [mse_history[-1], mae_history[-1]]
}
print(pd.DataFrame(error_data).to_string(index=False))

print("\nReport 3: Weight Coefficients (Before vs. After)")
weight_data = {
    'Weight': [f'w{i+1}' for i in range(9)],
    'Initial Weights': W_initial,
    'Optimized Weights': W_final
}
pd.set_option('display.float_format', '{:.6f}'.format)
print(pd.DataFrame(weight_data).to_string(index=False))
pd.reset_option('display.float_format')

print("\nReport 4: Price Predictions (First 10 Objects)")

y_pred_norm_initial = predict(W_initial, X_norm)
y_pred_norm_final = predict(W_final, X_norm)

price_scaler = scalers['SalePrice']
price_min = price_scaler['min']
price_max = price_scaler['max']

y_pred_initial = y_pred_norm_initial * (price_max - price_min) + price_min
y_pred_final = y_pred_norm_final * (price_max - price_min) + price_min

df_report = df.head(10)[input_cols + output_col].copy()
df_report['Predicted (Initial W)'] = y_pred_initial[:10]
df_report['Predicted (Optimized W)'] = y_pred_final[:10]

pd.set_option('display.float_format', '{:,.2f}'.format)
print(df_report.to_string(index=True))
pd.reset_option('display.float_format')