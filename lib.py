import numpy as np
from scipy.linalg import eigh

def create_F(file_name, C):
    """
    Creates Fock matrix F with coefficients C

    Parameters
	----------
    file_name: str
        Filename where the h_pq and <pr|g|qs> integrals are stored.
    C: np.array()
        New coefficients

    Returns
    ----------
    F: np.array()
        Fock matrix
    """

    return 

def solve_Roothan_eqs(file_name, C_0, S, eps, i_max = 100):
    """
    Solves the iterative generalized eigenvalue problem F(C)C = E*SC

    Parameters
    ----------
    file_name: str
        Filename where the h_pq and <pr|g|qs> integrals are stored.
    C_0: np.array(N)
        Initial coefficients
    S: np.array(N,N)
        Overlap matrix
    eps: float
        Precision with which to find the iterative problem
    i_max: int
        Maximum number of iterations

    Returns
    ----------
    E: np.array(N)
        Array with all the eigenvalues ordered from lowest to largest
    C: np.array(N,N)
        Array with the coefficients of each eigenvector ordered as E
    
    """

    counter = 1

    C_old = C_0
    E_old = 0

    while counter < i_max:
        F = create_F(file_name, C_old)
        E, C = eigh(F, S)

        if np.max(np.abs((E-E_old)/E)) < eps:
            break
        else:
            E_old = E
            C_old = C
            counter += 1

    return E, C