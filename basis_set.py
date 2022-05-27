import numpy as np
from scipy import special

#######################################################################################
# Hermite and associated Laguerre polynomials

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


def ass_Laguerre_pol(x, n, m):
	"""
	Evaluates the asssociated Laguerre polynomial of order n and alpha = m at x, 
	i.e. L_n^(alpha) (x)

	Parameters
	==========
	x : float or np.ndarray
		Values in which to evaluate the associated Laguerre polynomial
	n : int
		Oder of the associated Laguerre polynomial
	m : float
		Alpha from the formula of associated Laguerre polynomial

	Returns
	=======
	float or np.ndarray
	"""
	return special.eval_genlaguerre(n, m, x)


#######################################################################################
# Single Basis functions from doi.org/10.1103/PhysRevB.53.9952
# Units of length is the effective Bohr radius
# Units of energy is twice the effective Rydberg

EFFECTIVE_RYDBERG = 11.61E-3 # eV
EFFECTIVE_BOHR_RADIUS = 99E-10 # m
L_X = 4.95E-9 # m
L_Z = 7.425E-9 # m
OMEGA_X = 1/(L_X/EFFECTIVE_BOHR_RADIUS)**2 # adimensional
OMEGA_Z = 1/(L_Z/EFFECTIVE_BOHR_RADIUS)**2 # adimensional


def X_nml(x, y, z, n, m, l, omega_x=OMEGA_X, omega_z=OMEGA_Z):
	"""
	Single Basis wave functions from doi.org/10.1103/PhysRevB.53.9952 defined in Eq. (3),
	evaluated at r = (x,y,z). 

	Parameters
	==========
	x, y, z : np.ndarray
		Values in which to evaluate eigenfunction
	n : int
		Quantum number for the motion in the radial direction
	m : int
		magnetic quantum number for the angular momentum L_z
	l : int
		Quantum number for the motion in the z direction
	omega_x : float
		Angular frequency of the Harmonic Oscillators in the x,y directions
	omega_z : float
		Angular frequency of the Harmonic Oscillators in the z direction

	Returns
	=======
	X : np.ndarray
		Single Basis wave function n,m,l evaluated at x,y,z
	"""

	rho = np.sqrt(x**2 + y**2)
	e_itheta = (x + 1j*y)/rho

	X = np.sqrt((omega_x*np.math.factorial(n))/(np.pi*np.math.factorial(n + np.abs(m)))) * \
		(omega_x*rho**2)**(np.abs(m)/2)*ass_Laguerre_pol(omega_x*rho**2, n, np.abs(m)) * \
		e_itheta * np.exp(-omega_x*rho**2 / 2) * \
		np.sqrt(np.sqrt(omega_z/np.pi)/(2**l*np.math.factorial(l))) * \
		Hermite_pol(np.sqrt(omega_z)*z, l) * np.exp(-omega_z*z**2 / 2)

	return X


def E_nml(n, m, l, omega_x=OMEGA_X, omega_z=OMEGA_Z):
	"""
	Single Basis energies from doi.org/10.1103/PhysRevB.53.9952 defined in Eq. (4).

	Parameters
	==========
	n : int
		Quantum number for the motion in the radial direction
	m : int
		magnetic quantum number for the angular momentum L_z
	l : int
		Quantum number for the motion in the z direction
	omega_x : float
		Angular frequency of the Harmonic Oscillators in the x,y directions
	omega_z : float
		Angular frequency of the Harmonic Oscillators in the z direction

	Returns
	=======
	float
	"""
	return omega_x*(2*n + np.abs(m) + 1) + omega_z*(l + 0.5)


#######################################################################################
# Harmonic oscillator wavefunctions

def HO_wf(x, n, omega=OMEGA_X):
	"""
	Evaluates the n^th eigenfunction of the Harmonic Oscillator at x.
	The units of x are sqrt(hbar/m*omega). 

	Parameters
	==========
	x : float or np.ndarray
		Values in which to evaluate eigenfunction
	n : int
		Number of the eigenfunction
	omega : float
		Angular frequency of the Harmonic Oscillator

	Returns
	=======
	float or np.ndarray
	"""
	return omega**0.25 * Hermite_pol(np.sqrt(omega)*x, n) * np.exp(-omega*x**2 / 2) / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))


def HO_wf_3D(x, y, z, nx, ny, nz, omega_x=OMEGA_X, omega_y=OMEGA_X, omega_z=OMEGA_Z):
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
	return HO_wf(x, nx, omega=omega_x)*HO_wf(y, ny, omega=omega_y)*HO_wf(z, nz, omega=omega_z)