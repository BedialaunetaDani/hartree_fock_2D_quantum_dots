import numpy as np
from scipy import special
from scipy import integrate


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
	x : float or np.ndarray(N)
		Values in which to evaluate eigenfunction
	n : int
		Number of the eigenfunction
	omega : float
		Angular frequency of the Harmonic Oscillator

	Returns
	=======
	float or np.ndarray(N)
	"""
	return omega**0.25 * Hermite_pol(np.sqrt(omega)*x, n) * np.exp(-omega*x**2 / 2) / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))

def HO_wf_abs(x,n,omega = OMEGA_X):
	"""
	Evaluates the absolute value of the n^th eigenfunction of the Harmonic Oscillator at x.
	The units of x are sqrt(hbar/m*omega). 

	Parameters
	==========
	x : float or np.ndarray(N)
		Values in which to evaluate eigenfunction
	n : int
		Number of the eigenfunction
	omega : float
		Angular frequency of the Harmonic Oscillator

	Returns
	=======
	float or np.ndarray(N)
	"""
	return np.absolute(HO_wf(x,n,omega))

def HO_wf_xabs(x,n,omega =OMEGA_X):
	"""
	Evaluates the x times the absolute value of the n^th eigenfunction of the Harmonic Oscillator at x.
	The units of x are sqrt(hbar/m*omega). 

	Parameters
	==========
	x : float or np.ndarray(N)
		Values in which to evaluate eigenfunction
	n : int
		Number of the eigenfunction
	omega : float
		Angular frequency of the Harmonic Oscillator

	Returns
	=======
	float or np.ndarray(N)
	"""
	return x*np.absolute(HO_wf(x,n,omega))

def HO_wf_3D(x, y, z, nx, ny, nz, omega_x=OMEGA_X, omega_y=OMEGA_X, omega_z=OMEGA_Z):
	"""
	Evaluates the n^th eigenfunction of the (anisotropic) Harmonic Oscillator at x,y,z.
	The units of x,y,z are sqrt(hbar/m*omega). 

	Parameters
	==========
	x, y, z : float or np.ndarray(N)
		Position in which to evaluate the wave function
	nx, ny, nz : int
		Number of the eigenfunction for each cartesian coordinate
	omega_x, omega_y, omega_z : float
		Harmonic osciallation constant for each cartesian coordinate

	Returns
	=======
	float or np.ndarray(N)
	"""
	return HO_wf(x, nx, omega=omega_x)*HO_wf(y, ny, omega=omega_y)*HO_wf(z, nz, omega=omega_z)


def index_to_q_numbers(k):
	"""
	Returns the quantum numbers nx, ny, nz associated with the basis index k

	Parameters
	----------
	k: int
		Index of the basis from 1 to 14

	Returns
	----------
	nz ,ny, nz : int 
		Quantum numbers
	"""

	q_numbers = np.array([(0,0,0),(0,0,1),(0,1,0),(1,0,0),(0,1,1),(1,0,1),(1,1,0),(1,1,1),(1,1,2),(1,2,1),(2,1,1),(1,2,2),(2,1,2),(2,2,1)])

	return q_numbers[k]


def HO_wf_3D_basis(R, k, omega_x=OMEGA_X, omega_y=OMEGA_X, omega_z=OMEGA_Z):
	"""
	Evaluates the k^th basis function of the 3D HO at position R 

	Parameters
	----------
	k: int
		Index of the basis from 1 to 14
	R: np.ndarray(3,N)
		Position of the electron x, y, z
		at N diferent coordinates (to evaluate random walkers simultaneously)

	Returns
	----------
	psi : np.ndarray(N)
		Value of the wf at N points
	"""
	
	nx, ny, nz = index_to_q_numbers(k)
	
	psi = HO_wf_3D(R[0], R[1], R[2], nx, ny, nz, omega_x, omega_y, omega_z)

	return psi

def basis_radius(k):
		"""
		Calculates the inverse of the mean radius of an element of the basis. The biggest 
		radius in the three directions is taken.

		Parameters
		----------
		k: int
			Indx that specifies the base element

		Returns
		----------
		alpha : float
			Inverse of the mean radius
		"""
		nx, ny, nz = index_to_q_numbers(k)
		
		alphax = integrate.quad(HO_wf_abs, 0, np.inf, args=nx)[0]/integrate.quad(HO_wf_xabs, 0, np.inf, args=nx)[0]

		if (nx == ny):
			alphay = alphax
		else:
			alphay = integrate.quad(HO_wf_abs, 0, np.inf, args=ny)[0]/integrate.quad(HO_wf_xabs, 0, np.inf, args=ny)[0]

		if ((nx == ny)or(ny == nz)):
			alphaz = alphax
		else:
			alphaz = integrate.quad(HO_wf_abs, 0, np.inf, args=nz)[0]/integrate.quad(HO_wf_xabs, 0, np.inf, args=nz)[0]
		
		return min(alphax,alphay,alphaz)

def two_body_integrand(R, indices):
	"""
	Returns the value of the <pr|g|qs> integrand evaluated at R

	Parameters
	----------

	R: np.ndarray(6,N)
		Positions of the two electrons x1,y1,z1,x2,y2,z2 
		at N diferent coordinates (to evaluate random walkers simultaneously)
	indices: np.ndarray(4)
		Indices of the basis functions 

	Returns
	----------
	I : np.ndarray(N)
		
	"""
	p,q,r,s = indices[0],indices[1],indices[2],indices[3]

	alpha, beta, gamma, delta = basis_radius(p), basis_radius(q), basis_radius(r), basis_radius(s)

	a = alpha + beta
	b = gamma + delta

	R[0:3] = a**(-0.5)*R[0:3]
	R[3:6] = b**(-0.5)*R[3:6]

	r1 = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
	r2 = np.sqrt(R[3]**2 + R[4]**2 + R[5]**2)
	r12 = abs(r1-r2)

	I = (a*b)**(-1.5)*HO_wf_3D_basis(R[0:3],p)*HO_wf_3D_basis(R[3:6],r)*(1/r12)*HO_wf_3D_basis(R[0:3],q)*HO_wf_3D_basis(R[3:6],s)

	return I

def sampling_function(R):
		"""
		Function used to generate the sampling for the Monte Carlo Method.

		Parameters
		----------
		R: np.ndarray(6, N)
			Coordinates of the position of 2 electrons 

		Returns
		----------
		value : np.ndarray(N)
			Value of the sampling function at points R
		"""

		if R.ndim>1:
			value = 1/(2*np.pi)**3*np.exp(-0.5*np.sum(R**2,axis=1))
			
		else:
			value = 1/(2*np.pi)**3*np.exp(-0.5*np.sum(R**2,axis=0))

		return value


#######################################################################################
# Basis functions for He from Jos book

ALPHA_1 =  0.298073
ALPHA_2 =  1.242567
ALPHA_3 =  5.782948
ALPHA_4 = 38.474970

def He_wf(x, y, z, n):
	"""
	Evaluates the n^th eigenfunction of the Hydrogen atom with exponential coefficient alpha_i. 
	The units of x are atomic units. 

	Parameters
	==========
	x, y, z : float or np.ndarray
		Position in which to evaluate the wave function
	n : int
		Number of the wave function

	Returns
	=======
	float or np.ndarray
	"""

	if n == 1:
		alpha = ALPHA_1
	elif n == 2:
		alpha = ALPHA_2
	elif n == 3:
		alpha = ALPHA_3
	elif n == 4:
		alpha = ALPHA_4
	else:
		alpha = None

	r2 = x**2 + y**2 + z**2

	return np.exp(-alpha*r2)


def He_wf_basis(R,k):
	"""
	Evaluates the k^th eigenfunction of the Hydrogen atom with exponential coefficient alpha_i. 
	The units of x are atomic units. 

	Parameters
	==========
	R : np.ndarray(3,N)
		Position in which to evaluate the wave function
	k : int
		Number of the wave function

	Returns
	=======
	float or np.ndarray(N)
	"""
	return He_wf(R[0],R[1],R[2],k)


def He_two_body_integrand(x1,y1,z1,x2,y2,z2,p,q,r,s):
	"""
	Integrad that goes into the monte carlo method in order to compute the 
	two electron integrals.

	Parameters
	==========
	x1,y1,z1,x2,y2,z2 : float or np.ndarray(N)
		Position in which to evaluate the wave function
	p,q,r,s : int
		Indices of the wave function

	Returns
	=======
	float or np.ndarray(N)
	"""
	R=np.array([x1,y1,z1,x2,y2,z2])
	r1 = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
	r2 = np.sqrt(R[3]**2 + R[4]**2 + R[5]**2)
	r12 = abs(r1-r2)

	I = He_wf_basis(R[0:3],p)*He_wf_basis(R[3:6],r)*(1/r12)*He_wf_basis(R[0:3],q)*He_wf_basis(R[3:6],s)

	return I
	######

