import numpy as np
from scipy.linalg import eigh
from scipy import integrate
import os

import mc_integration as mc
import basis_set as bs


class integral_master():
	"""
	Calculates, stores and retrieves the values of the <pr|g|qs> integrals
	"""
	def __init__(self, dimension):
		"""
		Initialization of the object

		Parameters
		----------
		dimension : int
			Number of single basis functions

		Returns
		-------
		None
		"""

		self.integral_dict_1 = None
		self.integral_dict_2 = None
		self.dimension = dimension

		return

	def calculate(self, file_name):
		"""
		Calculates the <pr|g|qs> integrals and stores them in file

		Parameters
		----------
		file_name : str
			File name that stores the values of the integrals

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
		for p in range(1, self.dimension + 1):
			for q in range(1, self.dimension + 1):
				if p == q:
					I = self.calculate_1(p, q)
				else:
					I = 0

				integral_dict_1[(p, q)] = I

		# 2-body integrals
		for p in range(1, self.dimension + 1):
			for q in range(1, p + 1):
				for r in range(1, p):
					for s in range(1, r + 1):
						I = self.calculate_2(p, r, q, s)
					
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
					I = self.calculate_2(p, r, q, s)
				
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

		
	def calculate_1(self, p, q):
		"""
		Calculates the value of the h_pq integrals by direct integration

		Parameters
		----------
		p, q: int
			Indices that specify the h_pq integral

		Returns
		----------
		I : float
			Value of the h_pq integral
		"""
		nx,ny,nz = bs.index_to_q_numbers(p)

		I = (p==q)*(bs.OMEGA_X*(nx + ny + 1) + bs.OMEGA_Z*(nz + 0.5)) # nx,ny,nz should come from the basis, so from p and q but how?

		return I


	def calculate_2(self, p, r, q, s):
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

		system_size = 5
		N_walkers = 400
		N_steps = 10000
		N_skip = 1000
		trial_move = 0.6

		integrand = bs.two_body_integrand
		indices = np.array([p,r,q,s])
		dimension = 6

		sampling = bs.sampling_function
		
		I, deltaI, acceptance_ratio, trial_move = mc.MC_integration(sampling, integrand, indices, dimension, N_steps, N_walkers, N_skip, system_size, N_cores=1, trial_move = trial_move)

		return I
	


def create_F_matrix(rho, integrals):
    """
    Creates the Fock matrix with coefficients C, given by
    F[p,q] = h[p,q] + 2J[p,q] - K[p,q]

    Parameters
    ----------
    rho: np.ndarray(N, N)
        Density matrix of the system
    integrals : two_body_integrals() class
        Class with all the information regarding the <pr|g|qs> integrals

    Returns
    -------
    F: np.ndarray(N, N)
        Fock matrix
    """
	
    Nbasis = rho.shape[0]
    F = np.zeros((Nbasis, Nbasis))

    for p in range( Nbasis ):
        for q in range(Nbasis):
            F[p, q] += integrals.get_1(p+1, q+1) # add h matrix
            for r in range(Nbasis):
                for s in range(Nbasis):
                    F[p, q] += rho[r,s]*(integrals.get_2(p+1, q+1, r+1, s+1) - 0.5*integrals.get_2(p+1, r+1, q+1, s+1))
    return F

def density_matrix(C, N_electrons):
	"""
	Returns the density matrix of the system given its coefficients

	Parameters
	----------
	C : np.ndarray(N, N)
		Coefficients of the system
	N_electrons : int
		Number of electrons in the system

	Returns
	-------
	rho : np.ndarray(N, N)
		Density matrix of the system
	"""

	Nbasis = C.shape[0]
	rho = np.zeros((Nbasis, Nbasis))

	for p in range(Nbasis):
		for q in range(Nbasis):
			for k in range(int(N_electrons/2)):
				rho[p,q] = 2*C[p,k]*np.conjugate(C[q,k])

	return rho


def total_energy(rho, F, integrals):
	"""
	Returns the density matrix of the system given its coefficients

	Parameters
	----------
	rho : np.ndarray(N, N)
		Density matrix of the system
	F : np.ndarray(N, N)
		Fock matrix
	integrals : wo_body_integrals() class
		Class with all the information regarding the <pr|g|qs> integrals

	Returns
	-------
	E : float
		Total energy of the system
	"""

	Nbasis = rho.shape[0]
	E = 0

	for p in range(Nbasis):
		for q in range(Nbasis):
			E += 0.5*rho[p,q]*(integrals.get_1(p+1,q+1) + F[p,q])

	return E


def delta_rho(rho, rho_old): 
	"""
	Calculate change in density matrix using Root Mean Square Deviation (RMSD)

	Parameters
	----------
	rho : np.ndarray(N, N)
		Density matrix of the system
	rho_old : np.ndarray(N, N)
		Density matrix of the system in the previous SCF iteration

	Returns
	-------
	delta : float
		Root Mean Square Deviation (RMSD) of rho and rho_old
	"""

	Nbasis = rho.shape[0]
	delta = 0

	for p in range(Nbasis):
		for q in range(Nbasis):
			delta = delta + (rho[p,q] - rho_old[p,q])**2

	return np.sqrt(delta)