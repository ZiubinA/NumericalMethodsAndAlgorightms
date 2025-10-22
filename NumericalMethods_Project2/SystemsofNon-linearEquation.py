import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def funct(x1, x2):
    Z1 = x1**2 + 2 * (x2 - np.cos(x1))**2 - 20
    Z2 = x1**2 * x2 - 2
    return np.array([[Z1], [Z2]])

def dfunct(x1, x2):
    dZ1_dx1 = 2*x1 + 4*(x2 - np.cos(x1)) * np.sin(x1)
    dZ1_dx2 = 4*(x2 - np.cos(x1))
    dZ2_dx1 = 2*x1*x2
    dZ2_dx2 = x1**2
    return np.array([
        [dZ1_dx1, dZ1_dx2],
        [dZ2_dx1, dZ2_dx2]
    ])

def Z1(x1, x2):
    return x1**2 + 2 * (x2 - np.cos(x1))**2 - 20

def Z2(x1, x2):
    return x1**2 * x2 - 2

def newton_method(x1_0, x2_0, max_iter=100, tolerance=1e-6):
    x = np.array([[x1_0], [x2_0]], dtype=float)
    
    for _ in range(max_iter):
        f_x = funct(x[0,0], x[1,0])
        norm_f = np.linalg.norm(f_x)
        
        if norm_f < tolerance:
            return x.flatten() # Success
        
        J_x = dfunct(x[0,0], x[1,0])
        
        if abs(np.linalg.det(J_x)) < 1e-9:
            return None # Singular matrix
            
        x = x - np.linalg.inv(J_x) @ f_x

    return None # Failed to converge

print("System Functions Defined")

print("Generating Plot 1: Graphical Solution")
plt.figure(figsize=(9, 7))

x1_plot = np.linspace(-6, 6, 400)
x2_plot = np.linspace(0, 4, 400) 
X1_p, X2_p = np.meshgrid(x1_plot, x2_plot)

Z_1_plot = Z1(X1_p, X2_p)
Z_2_plot = Z2(X1_p, X2_p)

# Plot Z1 = 0 contour
plt.contour(X1_p, X2_p, Z_1_plot, levels=[0], colors='red', linestyles='--')
# Plot Z2 = 0 contour
plt.contour(X1_p, X2_p, Z_2_plot, levels=[0], colors='blue')

# Create legend handles
line_Z1 = plt.Line2D([], [], color='red', linestyle='--', label='$Z_1(x_1, x_2) = 0$')
line_Z2 = plt.Line2D([], [], color='blue', label='$Z_2(x_1, x_2) = 0$')
plt.legend(handles=[line_Z1, line_Z2])

plt.grid(True, linestyle=':', alpha=0.7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Graphical Solution: Intersections of Contours')
plt.ylim(0, 4)
plt.show() 

print("\nPreparing Plot 2: Mesh of Initial Guesses")
step = 1.0
x1_vals = np.arange(-10, 10 + step, step)
x2_vals = np.arange(-10, 10 + step, step)

solutions = []
colors = ['r', 'g', 'b', 'm', 'c', 'y']
convergence_tolerance = 1e-2 

print("Pass 1: Finding unique solutions")
for x1 in x1_vals:
    for x2 in x2_vals:
        sol = newton_method(x1, x2)
        
        if sol is not None:
            if not any(np.linalg.norm(sol - s) < convergence_tolerance for s in solutions):
                solutions.append(sol)
                print(f"Found new solution: {sol}")

print(f"Found {len(solutions)} unique solutions")

plt.figure(figsize=(10, 10))
ax = plt.gca() # Get current axes
ax.set_aspect('equal', adjustable='box')

x1_contour = np.linspace(-10, 10, 400)
x2_contour = np.linspace(-10, 10, 400)
X1_c, X2_c = np.meshgrid(x1_contour, x2_contour)
Z_1_contour = Z1(X1_c, X2_c)
Z_2_contour = Z2(X1_c, X2_c)
plt.contour(X1_c, X2_c, Z_1_contour, levels=[0], colors='black')
plt.contour(X1_c, X2_c, Z_2_contour, levels=[0], colors='black')
print("Contours plotted on mesh graph")

print("Pass 2: Plotting basins and non-converged points")
legend_handles = []

for i, sol in enumerate(solutions):
    color = colors[i % len(colors)]
    basin_x = []
    basin_y = []
    
    for x1 in x1_vals:
        for x2 in x2_vals:
            sol_i = newton_method(x1, x2)
            if sol_i is not None and np.linalg.norm(sol_i - sol) < convergence_tolerance:
                basin_x.append(x1)
                basin_y.append(x2)
                
    plt.plot(basin_x, basin_y, color + 'o', markersize=8)
    label_text = f"Solution {i+1}: [{sol[0]:.4f} {sol[1]:.4f}]"
    plt.plot(sol[0], sol[1], color + 's', markersize=15, markeredgecolor='black')
    legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, label=label_text, markersize=8, linestyle=''))

non_conv_x = []
non_conv_y = []
for x1 in x1_vals:
    for x2 in x2_vals:
        if newton_method(x1, x2) is None:
            non_conv_x.append(x1)
            non_conv_y.append(x2)

plt.plot(non_conv_x, non_conv_y, 'ko', markersize=8) 
legend_handles.append(plt.Line2D([0], [0], marker='o', color='k', label='No Convergence', markersize=8, linestyle=''))

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.grid(True)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Mesh of Initial Guesses')
plt.legend(handles=legend_handles)
plt.show() 

print("\nCalculated Solutions (Newton's Method)")
print("="*80)
print(f"{'Solution':<13} | {'Calculated Solution (x1, x2)':<30}")

for idx, sol in enumerate(solutions):
    sol_str = f"({sol[0]:.6f}, {sol[1]:.6f})"
    print(f"{idx+1:<13} | {sol_str:<30}")

# Verification using scipy.optimize.fsolve
def fsolve_func(x):
    x1 = x[0]
    x2 = x[1]
    Z1 = x1**2 + 2 * (x2 - np.cos(x1))**2 - 20
    Z2 = x1**2 * x2 - 2
    return [Z1, Z2]

print("\nVerifying Solutions with scipy.optimize.fsolve")
for idx, sol in enumerate(solutions):
    fsolve_sol = fsolve(fsolve_func, sol)
    print(f"Solution {idx+1}:")
    print(f"  Our Newton Result:  ({sol[0]:.7f}, {sol[1]:.7f})")
    print(f"  scipy.fsolve Result: ({fsolve_sol[0]:.7f}, {fsolve_sol[1]:.7f})")
    print(f"  Match: {np.allclose(sol, fsolve_sol)}")
    print("-" * 30)