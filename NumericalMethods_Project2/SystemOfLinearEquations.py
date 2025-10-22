import numpy as np

def solve_gausian(A_in, b_in, system_name):
    print(f"\nSolving {system_name}")
    A = A_in.copy()
    b = b_in.copy()
    
    n = (np.shape(A))[0]   # number of equations
    nb = (np.shape(b))[1]  # number of solutions
    A1 = np.hstack((A, b))  # expanding matrix
    print("Initial Matrix")
    print(A1)

    try:
        for i in range (0,n-1):
            
            if np.abs(A1[i, i]) < 1e-12:
                if np.abs(A1[i, n]) > 1e-12:
                    print(f"\nERROR")
                    print(f"Row {i} shows a contradiction (0 = {A1[i,n]}).")
                    print("RESULT: NO SOLUTION")
                else:
                    print(f"\nERROR")
                    print(f"Row {i} is all zeros (0 = 0).")
                    print("RESULT: INFINITELY MANY SOLUTIONS")
                return
                
            for j in range (i+1,n):
                A1[j,i:n+nb]=A1[j,i:n+nb]-A1[i,i:n+nb]*A1[j,i]/A1[i,i];
                A1[j,i]=0;
        
        if np.abs(A1[n-1, n-1]) < 1e-12:
            if np.abs(A1[n-1, n]) > 1e-12:
                print(f"\nERROR")
                print(f"Last row shows a contradiction (0 = {A1[n-1,n]}).")
                print("RESULT: NO SOLUTION")
            else:
                print(f"\nERROR")
                print(f"Last row is all zeros (0 = 0).")
                print("ESULT: INFINITELY MANY SOLUTIONS")
            return

        # backward step
        x=np.zeros(shape=(n,nb))
        for i in range (n-1,-1,-1):
            sum_ax = np.dot(A1[i, i+1:n], x[i+1:n, :])
            x[i,:] = (A1[i, n:n+nb] - sum_ax) / A1[i,i]

        print("\nSolution x")
        print(x)
        print("\nVerification (A*x)")
        verification = np.dot(np.matrix(A_in), np.matrix(x))
        print(verification)
        print("Matches original b:", np.allclose(verification, b_in))

    except ZeroDivisionError:
        print(f"The pivot A[{i},{i}] was zero.")
        print("RESULT: NO SOLUTION or INFINITE SOLUTIONS")
        print(A1)

def solve_qr(A_in, b_in, system_name):
    print(f"\nSolving {system_name} using QR")
    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)   
    n = A.shape[0]
    nb = b.shape[1]

    Q = np.identity(n)
    
    for i in range (0,n-1):
        z = A[i:n, i:i+1] 
        
        zp = np.zeros_like(z)
        zp[0, 0] = np.linalg.norm(z)
        
        omega = z - zp
        
        norm_omega = np.linalg.norm(omega)
        if norm_omega < 1e-12:
            continue
            
        omega = omega / norm_omega 
        Qi = np.identity(n-i) - 2 * np.dot(omega, omega.T)
        A[i:n, :] = Qi.dot(A[i:n, :])
        Q[:, i:n] = Q[:, i:n].dot(Qi)
    R = A 
    
    print("\nResulting R matrix:")
    print(R)
    print("\nResulting Q matrix:")
    print(Q)
    
    # Check for singularity
    if np.abs(R[n-1, n-1]) < 1e-12:
        print("\nERROR: Matrix is singular (last pivot of R is zero).")
        b1_check = Q.transpose().dot(b)
        if np.abs(b1_check[n-1, 0]) > 1e-12:
            print(f"Contradiction found: 0 * x_n = {b1_check[n-1, 0]}")
            print("RESULT: NO SOLUTION")
        else:
            print("Row of zeros found: 0 * x_n = 0")
            print("RESULT: INFINITELY MANY SOLUTIONS")
        return

    # --- Backward Step ---
    b1 = Q.transpose().dot(b)
    x = np.zeros(shape=(n,nb))
    
    for i in range (n-1,-1,-1):
        sum_ax = np.dot(R[i, i+1:n], x[i+1:n, :])
        x[i,:] = (b1[i,:] - sum_ax) / R[i,i]

    print("\nSolution x:")
    print(x)
    print("\nVerification (A*x):")
    verification = np.dot(np.array(A_in), x)
    print(verification)
    print("Matches original b:", np.allclose(verification, b_in))


def solve_simple_iteration(A_in, b_in, system_name, nitmax=1000, eps=1e-12):
    print(f"\nolving {system_name} using Simple Iteration")
    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float) 
    n = A.shape[0]

    diag_A = np.diag(A)
    if np.any(np.abs(diag_A) < 1e-12):
        print("\nERROR: Zero element found on the diagonal.")
        print("The method cannot proceed due to division by zero.")
        print("RESULT: METHOD FAILED")
        return     

    alpha=np.array([1, 1, 1, 1])  
    
    D_inv_matrix = np.diag(1.0 / diag_A)
    
    Atld = D_inv_matrix.dot(A) - np.diag(alpha)
    btld = D_inv_matrix.dot(b)
    
    x = np.zeros(shape=(n, 1))
    x1 = np.zeros(shape=(n, 1))

    for it in range(0, nitmax):
        x1 = ((btld-Atld.dot(x)).transpose()/alpha).transpose()
        denominator = np.linalg.norm(x) + np.linalg.norm(x1)
        
        if denominator < eps:
            prec = 0.0
        else:
            prec = np.linalg.norm(x1 - x) / denominator
        if prec < eps:
            print(f"\nConverged in {it + 1} iterations.")
            x[:] = x1[:] 
            break
        
        x[:] = x1[:]    

    if np.isnan(x).any():
        print("Last computed solution x:")
        print(x)
        print("RESULT: FAILED TO CONVERGE (Diverged)")
        return
    elif it == nitmax - 1:
        print(f"\nERROR: Did not converge after {nitmax} iterations.")
        print("Last computed solution x:")
        print(x)
        print("RESULT: FAILED TO CONVERGE")
        return

    print("\nSolution x:")
    print(x)
    
    print("\nVerification (A*x):")
    verification = np.dot(np.array(A_in), x)
    print(verification)
    print("Matches original b:", np.allclose(verification, b_in, atol=eps*100))


A8 = np.matrix([[4.0, 3.0, -1.0, 1.0],
                [3.0, 9.0, -2.0, -2.0],
                [-1.0, -2.0, 11.0, -1.0],
                [1.0, -2.0, -1.0, 5.0]])
b8 = (np.matrix([12.0, 10.0, -28.0, 16.0])).transpose()

A13 = np.matrix([[1.0, -2.0, 3.0, 4.0],
                 [1.0, 0.0, -1.0, 1.0],
                 [2.0, -2.0, 2.0, 5.0],
                 [0.0, -7.0, 3.0, 1.0]])
b13 = (np.matrix([11.0, -4.0, 7.0, 2.0])).transpose()

A20 = np.matrix([[2.0, 4.0, 6.0, -2.0],
                 [1.0, 3.0, 1.0, -3.0],
                 [1.0, 1.0, 5.0, 1.0],
                 [2.0, 3.0, -3.0, -2.0]])
b20 = (np.matrix([2.0, 1.0, 7.0, 2.0])).transpose()

solve_gausian(A8, b8, "System 8")
solve_gausian(A13, b13, "System 13")
solve_gausian(A20, b20, "System 20")
solve_qr(A8, b8, "System 8")
solve_qr(A13, b13, "System 13")
solve_qr(A20, b20, "System 20")
solve_simple_iteration(A8, b8, "System 8")
solve_simple_iteration(A13, b13, "System 13")
solve_simple_iteration(A20, b20, "System 20")