import numpy as np
import os


##########################################################
#                  SELF-CONSISTENT FIELD
##########################################################

def SCF(N_electrons, integrals, S, max_iter_SCF=200, eps_SCF=1E-5, max_delta_rho=0, C=None):
	"""
	Self-Consistent Field for Hartree Fock

	Parameters
	----------
	N_electrons : int
		Number of electrons in the system
	integrals : class
		Stores all the information about the one-electron and two-electron integrals
	S : np.ndarray(N_basis, N_basis)
		Overlap matrix for the basis functions
	max_iter_SCF : int
		Maximum number of iterations in SCF until it gets restarted
	eps_SCF : float
		If the change of the density matrix is less than eps_SCF, it has converged
	max_delta_rho : float
		If the change of the density matrix is more than max_delta_rho, it uses partially uses previous density matrix
	C : np.ndarray(N_basis, N_basis)
		Initial value of the coefficients to be used in SCF
		If None, it is set to 0
	"""

	N_basis = S.shape[0]
	converged = False

	while not converged: # restart SCF if we reached maximum number of iterations
		n_iterations = 0

		if C is None: C = np.zeros((N_basis, N_basis))
		rho = density_matrix(C, N_electrons)
		rho_old = np.zeros((N_basis, N_basis))

		while n_iterations < max_iter_SCF:
			n_iterations += 1

			F = create_F_matrix(rho, integrals)

			if (S == np.eye(N_basis)).all(): # orthonormal single basis
				E, C = np.linalg.eigh(F)
			else:
				E, C = scp.eigh(F, S)

			rho = density_matrix(C, N_electrons)

			if delta_rho(rho, rho_old) > max_delta_rho: # checks if change of rho is too large
				alfa = np.random.rand()
				rho = alfa*rho + (1 - alfa)*rho_old

			total_E = total_energy(rho, F, integrals)
			print("E = {:0.7f} | N(SCF) = {}".format(total_E, n_iterations))

			if delta_rho(rho, rho_old) < eps_SCF: # checks convergence of SCF
				converged = True
				print("SCF CONVERGED! E = {:0.10f}".format(total_E))
				break

			total_E_old = total_E
			rho_old = rho

		print("SCF not converged!\nRestarting again...")

	return


def create_F_matrix(rho, integrals):
	"""
	Creates the Fock matrix with coefficients C, given by
	F[p,q] = h[p,q] + 2J[p,q] - K[p,q]

	Parameters
	----------
	rho: np.ndarray(N_basis, N_basis)
		Density matrix of the system
	integrals : two_body_integrals() class
		Class with all the information regarding the <pr|g|qs> integrals

	Returns
	-------
	F: np.ndarray(N_basis, N_basis)
		Fock matrix
	"""
	
	N_basis = rho.shape[0]
	F = np.zeros((N_basis, N_basis))

	for p in range(N_basis):
		for q in range(N_basis):
			F[p, q] += integrals.get_1(p+1, q+1) # add h matrix
			for r in range(N_basis):
				for s in range(N_basis):
					F[p, q] += rho[r,s]*(integrals.get_2(p+1, q+1, r+1, s+1) - 0.5*integrals.get_2(p+1, r+1, q+1, s+1))

	return F


def density_matrix(C, N_electrons):
	"""
	Returns the density matrix of the system given its coefficients

	Parameters
	----------
	C : np.ndarray(N_basis, N_basis)
		Coefficients of the system
	N_electrons : int
		Number of electrons in the system

	Returns
	-------
	rho : np.ndarray(N_basis, N_basis)
		Density matrix of the system
	"""

	N_basis = C.shape[0]
	rho = np.zeros((N_basis, N_basis))

	for p in range(N_basis):
		for q in range(N_basis):
			for k in range(int(N_electrons/2)):
				rho[p,q] = 2*C[p,k]*np.conjugate(C[q,k])

	return rho


def total_energy(rho, F, integrals):
	"""
	Returns the density matrix of the system given its coefficients

	Parameters
	----------
	rho : np.ndarray(N_basis, N_basis)
		Density matrix of the system
	F : np.ndarray(N_basis, N_basis)
		Fock matrix
	integrals : class
		Class with all the information regarding the <pr|g|qs> integrals

	Returns
	-------
	E : float
		Total energy of the system
	"""

	N_basis = rho.shape[0]
	E = 0

	for p in range(N_basis):
		for q in range(N_basis):
			E += 0.5*rho[p,q]*(integrals.get_1(p+1,q+1) + F[p,q])

	return E


def delta_rho(rho, rho_old): 
	"""
	Calculate change in density matrix using Root Mean Square Deviation (RMSD)

	Parameters
	----------
	rho : np.ndarray(N_basis, N_basis)
		Density matrix of the system
	rho_old : np.ndarray(N_basis, N_basis)
		Density matrix of the system in the previous SCF iteration

	Returns
	-------
	delta : float
		Root Mean Square Deviation (RMSD) of rho and rho_old
	"""

	N_basis = rho.shape[0]
	delta = 0

	for p in range(N_basis):
		for q in range(N_basis):
			delta = delta + (rho[p,q] - rho_old[p,q])**2

	return np.sqrt(delta)


##########################################################
#               SINGLE-BASIS INTEGRATION
##########################################################

def MC_integration(integrand, cov, N_points=1000000):
	"""
	Computes integral using Monte Carlo integration. 
	Sampling function is a multivariate normal with mean = 0 and covariance matrix = cov. 
	Function to integrate is given by integrand. 

	Parameters
	----------
	integrand : function
		Function to integrate using Monte Carlo
	cov : np.ndarray(dim, dim)
		Covariance matrix for the multivariate normal distribution
	N_points : int
		Number of points to sample from distribution for Monte Carlo integral

	Returns
	-------
	I : float
		Value of the integral
	I_err : float 
		Error of the integral
	"""

	dim = cov.shape[0]
	steps = np.random.multivariate_normal(np.zeros(dim), cov, N_points) 
	I = np.average(steps)
	I_err = np.std(steps)/np.sqrt(N_points)

	return I, I_err


class integral_master():
	"""
	Calculates, stores and retrieves the values of the one-electron and two-electron integrals
	"""
	def __init__(self, N_basis):
		"""
		Initialization of the object

		Parameters
		----------
		N_basis : int
			Number of single basis functions

		Returns
		-------
		None
		"""

		self.integral_dict_1 = None
		self.integral_dict_2 = None
		self.N_basis = N_basis

		return

	def calculate(self, file_name, analyical_1, analyical_2=None, MC_args={"f_cov":None, "f_integrand":None, "N_points":1}):
		"""
		Calculates the one-electron and two-electron integrals and stores them in file. 
		If no function for the analytical value is given, it uses Monte Carlo integration

		Parameters
		----------
		file_name : str
			File name that stores the values of the integrals
		analyical_1 : function
			Analytical function for the one-electron integrals
		analytical_2 : function
			Analytical function for the two-electron integrals
		MC_args : dict
			Arguments of the Monte Carlo integration
			f_cov returns the covariance matrix given p,r,q,s
			f_integrand returns the integrand function given p,r,q,s

		Returns
		-------
		None
		"""

		if file_name in os.listdir():
			print("Integral file already exists. Not computing the integrals. ")
			self.load_integrals(file_name)
			return

		integral_dict_1 = {}
		integral_dict_2 = {}

		# 1-body integrals
		for p in range(1, self.N_basis + 1):
			for q in range(1, self.N_basis + 1):
				integral_dict_1[(p, q)] = analyical_1(p, q)

		# 2-body integrals
		for p in range(1, self.N_basis + 1):
			for q in range(1, p + 1):
				for r in range(1, p):
					for s in range(1, r + 1):
						if analyical_2 is None:
							I = self.calculate_2(p, r, q, s, MC_args)
						else:
							I = analyical_2(p, r, q, s)
					
						integral_dict_2[(p, r, q, s)] = I
						integral_dict_2[(q, r, p, s)] = I
						integral_dict_2[(p, s, q, r)] = I
						integral_dict_2[(r, p, s, q)] = I
						integral_dict_2[(q, s, p, r)] = I
						integral_dict_2[(r, q, s, p)] = I
						integral_dict_2[(s, p, r, q)] = I
						integral_dict_2[(s, q, r, p)] = I
				r = p
				for s in range(1, q + 1):
					if analyical_2 is None:
						I = self.calculate_2(p, r, q, s, MC_args)
					else:
						I = analyical_2(p, r, q, s)
				
					integral_dict_2[(p, r, q, s)] = I
					integral_dict_2[(q, r, p, s)] = I
					integral_dict_2[(p, s, q, r)] = I
					integral_dict_2[(r, p, s, q)] = I
					integral_dict_2[(q, s, p, r)] = I
					integral_dict_2[(r, q, s, p)] = I
					integral_dict_2[(s, p, r, q)] = I
					integral_dict_2[(s, q, r, p)] = I

		np.save(file_name, np.array([integral_dict_1, integral_dict_2]))

		self.load_integrals(file_name)

		return

	def calculate_2(self, p, r, q, s, MC_args):
		"""
		Calculates the value of the <pr|g|qs> integrals by monte carlo integration methods

		Parameters
		----------
		p, q, r, s: int
			Indices that specify the <pr|g|qs> integral

		Returns
		----------
		I : float
			Value of the <pr|g|qs> integral
		"""

		cov = MC_args["f_cov"](p, r, q, s)
		integrand = MC_args["f_integrand"](p, r, q, s)
		N_points = MC_args["N_points"]
		
		I, I_err = MC_integration(integrand, cov, N_points=N_points)

		return I

	def load_integrals(self, file_name):
		"""
		Loads the values of the integrals in this object

		Parameters
		----------
		file_name : str
			File name that stores the values of the integrals

		Returns
		-------
		None
		"""

		self.integral_dict_1, self.integral_dict_2 = np.load(file_name, allow_pickle=True)

		return

	def get_1(self, p, q):
		"""
		Returns the value of the h_pq integrals from the class dictionaries

		Parameters
		----------
		p, q: int
			Indices that specify the h_pq integral

		Returns
		-------
		I : float
			Value of the h_pq integral
		"""

		I = self.integral_dict_1[(p, q)]

		return I

	def get_2(self, p, r, q, s):
		"""
		Returns the value of the <pr|g|qs> integrals from the class dictionaries

		Parameters
		----------
		p, r, q, s: int
			Indeces that specify the <pr|g|qs> integral

		Returns
		-------
		I : float
			Value of the <pr|g|qs> integral
		"""

		I = self.integral_dict_2[(p, r, q, s)]

		return I