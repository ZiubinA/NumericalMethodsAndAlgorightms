import numpy as np
import matplotlib.pyplot as plt
import math

def fx(x):
    return np.exp(-x)*np.cos(x)*np.sin(np.power(x, 2) - 1)

def f_prime(x):
   return np.exp(-x)*(2 * x * np.cos(x) * np.cos(np.power(x, 2) - 1)-(np.cos(x) + np.sin(x))*np.sin(np.power(x, 2) - 1))

xmin, xmax = 7, 8
h = 0.05

dx = 0.0001;  x=np.arange(xmin,xmax, dx); y=fx(x)
plt.xlabel("x"); plt.ylabel("y"); plt.plot(x, y, 'b'); plt.grid()

list = []

while xmin<xmax:
  if np.sign(fx(xmin)) !=np.sign(fx(xmin+h)):
    list.append(xmin)
    list.append(xmin+h)
    print("Root : [%.3f ;  %.3f]" % (xmin,xmin+h))
    plt.plot([xmin], [0], 'or')
    plt.plot([xmin+h], [0 ], 'og')
  xmin += h

dx= 0.0001 
x=np.arange(7, 8+dx, dx)
y = fx(x)

plt.title('Transcendental function')
plt.xlabel("x");plt.ylabel("y")
plt.plot(x, y, 'b')
plt.xlim([7, 8])
plt.ylim([-0.001, 0.001])
plt.grid()
plt.savefig('graphT2.png')
plt.show()
for i in range(0, len(list), 2):
  iterations = 0
  xs = list[i] 
  xe = list[i+1]
  k = np.abs(fx(xs)/ fx(xe))
  xmid = (xs + k*xe)/(1+k)
  eps = 1e-8
  while np.abs(fx(xmid)) > eps:
    k = np.abs(fx(xs)/ fx(xe))
    xmid = (xs + k*xe)/(1+k)
    if np.sign(fx(xmid)) == np.sign(fx(xs)):
        xs = xmid
    else:
        xe = xmid
    iterations += 1
  print(f"Root Fx = {fx(xmid):.2e}, x = {xmid:.5f}", 'Iterations:', iterations)

dx= 0.0001 
x=np.arange(7, 8+dx, dx)
y = fx(x)
plt.title('Roots Found by Newton\'s Method')
plt.xlabel("x"); plt.ylabel("y")
plt.plot(x, y, 'b', label='f(x)')
plt.xlim([7, 8])
plt.ylim([-0.001, 0.001])
plt.grid()
print('start newton')
for i in range(0, len(list), 2):
    xi = list[i] 
    eps = 1e-8 
    iterations = 0
    while np.abs(fx(xi)) > eps:
        xi_bef = xi
        
        if f_prime(xi) == 0:
            print("Warning: Derivative is zero. Skipping this bracket.")
            break
            
        xi = xi - fx(xi) / f_prime(xi)
        iterations += 1
        print(f"Fx = {fx(xi):.3f} / x = {xi:.3f}",'Iterations:', iterations)

        plt.plot([xi_bef], [fx(xi_bef)], 'ob', markersize=4) #previouse point 
        plt.plot([xi_bef, xi], [fx(xi_bef), 0], 'r-') # Tangent line
        plt.plot([xi], [0], 'or') # Current guess on x-axis
        
plt.legend()
plt.show()

dx= 0.0001 
x=np.arange(7, 8+dx, dx)
y = fx(x)
plt.title('Roots Found by Quasi-Newton\'s Method')
plt.xlabel("x"); plt.ylabel("y")
plt.plot(x, y, 'b', label='f(x)')
plt.xlim([7, 8])
plt.ylim([-0.001, 0.001])
plt.grid()
print('start Quasi-Newton')
def dfx(x):
  h = 1e-6
  return (fx(x) - fx(x-h)) / h

eps = 1e-8
for i in range(0, len(list), 2):
   xi = list[i]
   iterations = 0
   while np.abs(fx(xi)) > eps:
      xi_bef = xi
  
      xi = xi - (1 / dfx(xi)) * fx(xi)
      iterations += 1
      print("Fx = " + str(fx(xi)) + "  /  x = " + str(xi),'Iterations:', iterations)
    
      plt.xlabel("x"); plt.ylabel("y"); plt.plot(x, y, 'b'); plt.grid()
      plt.plot([xi], [0], 'or')
      plt.plot([xi_bef, xi], [fx(xi_bef), 0], 'r-')
      plt.plot([xi, xi], [0, fx(xi)], 'g--')
   print(xi)
plt.show()

dx= 0.0001 
x=np.arange(7, 8+dx, dx)
y = fx(x)
plt.title('Roots Found by Secant\'s Method')
plt.xlabel("x"); plt.ylabel("y")
plt.plot(x, y, 'b', label='f(x)')
plt.xlim([7, 8])
plt.ylim([-0.001, 0.001])
plt.grid()
print('start secant')
for i in range(0, len(list), 2):
    xi = list[i] 
    xd = xi - (1 / dfx(xi)) * fx(xi)
    eps = 1e-8 
    iterations = 0
    while np.abs(fx(xi)) > eps:
       x_new = xi -(fx(xi)*(xi-xd)/(fx(xi)-fx(xd)))
       xi = xd
       xd = x_new
       iterations += 1
       print("Fx = " + str(fx(xi)) + "  /  x = " + str(xi),'Iterations:', iterations)
    
       plt.xlabel("x"); plt.ylabel("y"); plt.plot(x, y, 'b'); plt.grid()
       plt.plot([xi], [0], 'or')
       plt.plot([xi_bef, xi], [fx(xi_bef), 0], 'r-')
       plt.plot([xi, xi], [0, fx(xi)], 'g--')
    print(x_new)

plt.show()