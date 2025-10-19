import numpy as np

def solve_with_your_logic(A_in, b_in, system_name):
    print(f"\n========== Solving {system_name} ==========")
    A = A_in.copy()
    b = b_in.copy()
    
    n = (np.shape(A))[0]   # number of equations
    nb = (np.shape(b))[1]  # number of solutions
    A1 = np.hstack((A, b))  # expanding matrix
    print("--- Initial Matrix ---")
    print(A1)

    try:
        for i in range (0,n-1):
            
            if np.abs(A1[i, i]) < 1e-12:
                if np.abs(A1[i, n]) > 1e-12:
                    print(f"\n--- ERROR ---")
                    print(f"Row {i} shows a contradiction (0 = {A1[i,n]}).")
                    print("--- RESULT: NO SOLUTION ---")
                else:
                    print(f"\n--- ERROR ---")
                    print(f"Row {i} is all zeros (0 = 0).")
                    print("--- RESULT: INFINITELY MANY SOLUTIONS ---")
                return
                
            for j in range (i+1,n):
                A1[j,i:n+nb]=A1[j,i:n+nb]-A1[i,i:n+nb]*A1[j,i]/A1[i,i];
                A1[j,i]=0;
        
        print("\n--- After Forward Step ---")
        print(A1)
        
        if np.abs(A1[n-1, n-1]) < 1e-12:
            if np.abs(A1[n-1, n]) > 1e-12:
                print(f"\n--- ERROR ---")
                print(f"Last row shows a contradiction (0 = {A1[n-1,n]}).")
                print("--- RESULT: NO SOLUTION ---")
            else:
                print(f"\n--- ERROR ---")
                print(f"Last row is all zeros (0 = 0).")
                print("--- RESULT: INFINITELY MANY SOLUTIONS ---")
            return

        # backward step
        x=np.zeros(shape=(n,nb))
        for i in range (n-1,-1,-1):
            sum_ax = np.dot(A1[i, i+1:n], x[i+1:n, :])
            x[i,:] = (A1[i, n:n+nb] - sum_ax) / A1[i,i]

        print("\n--- Solution x ---")
        print(x)
        print("\n--- Verification (A*x) ---")
        verification = np.dot(np.matrix(A_in), np.matrix(x))
        print(verification)
        print("Matches original b:", np.allclose(verification, b_in))


    except ZeroDivisionError:
        print("\n--- CRASH: ZeroDivisionError ---")
        print(f"The pivot A[{i},{i}] was zero.")
        print("--- RESULT: NO SOLUTION or INFINITE SOLUTIONS ---")
        print(A1)

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

solve_with_your_logic(A8, b8, "System 8")
solve_with_your_logic(A13, b13, "System 13")
solve_with_your_logic(A20, b20, "System 20")