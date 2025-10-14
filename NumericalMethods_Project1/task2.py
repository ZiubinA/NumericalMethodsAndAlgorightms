import numpy as np
import math
import matplotlib.pyplot as plt

m = 60
t1 = 4
v1 = 30
g = 9.8

def equation_for_c(c):
    return (m * g / c) * (1 - np.exp(-(c / m) * t1)) - v1

def dfc(c):
    return (-m * g / np.power(c, 2)) * (1 - np.exp(-(c / m) * t1)) + (g * t1 / c) * np.exp(-(c / m) * t1)

dx = 1
x = np.arange(1, 10+dx, dx)
y = equation_for_c(x)
plt.title("Graph of f(c)")
plt.ylabel("f(c)")
plt.plot(x, y, 'b', label='f(c)')
plt.xlim([1, 10])
plt.ylim([-10, 10])
plt.grid()
plt.axhline(0, color='red', linestyle='--') 

def newtons_method(initial_guess, max_iterations=100, eps=1e-8):
    c = initial_guess
    for i in range(max_iterations):
        f_c = equation_for_c(c)
        df_c = dfc(c)
        if np.abs(f_c) < eps:
            return c
        if df_c == 0:
            return None
        c = c - f_c / df_c
    return c

initial_guess = 10.0
solution = newtons_method(initial_guess)

if solution is not None:
    print(f"\n'c' is approximately {solution:.4f} Ns/m.")
    plt.plot(solution, equation_for_c(solution), 'go', label=f'Root at c ≈ {solution:.4f}')

plt.legend()
plt.show()