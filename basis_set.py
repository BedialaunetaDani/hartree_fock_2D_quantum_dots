import numpy as np
from scipy import special

#######################################################################################
# Hermite polynomials

def Hermite_pol(x, n):
	"""
	Evaluates the Hermite polynomial of order n at x.

	Parameters
	==========
	x : float or np.ndarray
		Values in which to evaluate the Hermite polynomial
	n : int
		Oder of the Hermite polynomial

	Returns
	=======
	float or np.ndarray
	"""
	return special.eval_hermite(n, x)


#######################################################################################
# Harmonic oscillator wavefunctions

def HO_wf(x, n):
	"""
	Evaluates the n^th eigenfunction of the Harmonic Oscillator at x.
	The units of x are sqrt(hbar/m*omega). 

	Parameters
	==========
	x : float or np.ndarray
		Values in which to evaluate eigenfunction
	n : int
		Number of the eigenfunction

	Returns
	=======
	float or np.ndarray
	"""
	return Hermite_pol(x, n) * np.exp(-x**2 / 2) / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))