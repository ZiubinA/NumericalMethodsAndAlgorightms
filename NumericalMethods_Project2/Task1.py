import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import time

n = 5  
m = 3  

S_existing = np.random.uniform(-10, 10, (n, 2))

print(f"Optimizing locations for {m} new stores")
print(f"There are {n} existing stores at:\n{S_existing}\n")

# Helper Cost Functions
def cost_store_to_store(p1, p2):
    sq_dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return np.exp(-0.2 * sq_dist)

def cost_boundary(p):
    x, y = p
    
    #squared distance to the *nearest* boundary
    dx = max(0, abs(x) - 10)
    dy = max(0, abs(y) - 10)
    sq_dist_to_boundary = dx**2 + dy**2
    
    if sq_dist_to_boundary == 0:
        return 0
    else:
        return np.exp(0.25 * sq_dist_to_boundary) - 1

def funct(X):
    S_new = X.reshape((m, 2))
    
    total_cost = 0
    
    for j in range(m):
        p_j = S_new[j]  
        cost_j = 0      

        for i in range(n):
            p_i_existing = S_existing[i]
            cost_j += cost_store_to_store(p_j, p_i_existing)

        for k in range(m):
            if j == k: 
                continue
            p_k_new = S_new[k]
            cost_j += cost_store_to_store(p_j, p_k_new)
        
        # Calculate C^B 
        cost_j += cost_boundary(p_j)

        total_cost += cost_j
        
    return total_cost

def grad(X):

    funct_val = funct(X)
    h = 1e-9
    
    gradient = np.zeros_like(X)
    
    for i in range(len(X)):
        X_h = X.copy()
        X_h[i] += h 
        
        gradient[i] = (funct(X_h) - funct_val) / h
        
    return gradient

x = np.random.uniform(-10, 10, (1, 2 * m))
step = 0.1  

funct_old = funct(x[0])
gradient = grad(x[0])
direction = gradient / np.linalg.norm(gradient)
iterationsInSameDirection = 0
functionValues = [funct_old]

print("Starting optimization...")
start_time = time.time()

for i in range(1000):
    x[0] = x[0] - step * direction
    
    funct_new = funct(x[0])
    iterationsInSameDirection += 1
    
    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1:5d}, Total Cost: {funct_new:7.6e}")
    
    if funct_new > funct_old:
        x[0] = x[0] + step * direction
        step = step / 1.05
        
        gradient = grad(x[0])
        if np.linalg.norm(gradient) < 1e-9:
             print("Gradient norm is near zero. Stopping.")
             break
        direction = gradient / np.linalg.norm(gradient)
        
        iterationsInSameDirection = 0
    else:
        funct_old = funct_new
    
    functionValues.append(funct_new)  
    
    if abs(funct_new) < 1e-9 or step < 1e-12:
        break

end_time = time.time()
print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

print("\n--- Final Results ---")
final_locations = x[0].reshape((m, 2))
print(f"Optimal locations for {m} new stores:")
print(final_locations)

final_cost = funct(x[0])
print(f"\nFinal minimal total cost: {final_cost:7.6e}")

plt.figure()
plt.plot(functionValues)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.title('Cost vs. Iteration')
plt.show()

plt.figure(figsize=(8, 8))
plt.title('City Map with Store Locations')
plt.scatter(S_existing[:, 0], S_existing[:, 1], c='blue', label=f'Existing Stores (n={n})', s=100)
plt.scatter(final_locations[:, 0], final_locations[:, 1], c='red', marker='*', label=f'New Stores (m={m})', s=200)

plt.plot([-10, 10], [-10, -10], 'k-')
plt.plot([-10, 10], [10, 10], 'k-')
plt.plot([-10, -10], [-10, 10], 'k-')
plt.plot([10, 10], [-10, 10], 'k-')

plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()