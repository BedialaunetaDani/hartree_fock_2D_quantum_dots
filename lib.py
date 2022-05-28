import numpy as np
from scipy.linalg import eigh
import os


class integral_master():
	"""
	Calculates, stores and retrieves the values of the <pr|g|qs> integrals
	"""
	def __init__(self, dimension):
		self.integral_dict_1 = None
		self.integral_dict_2 = None
		self.dimension = dimension
		return

	def calculate(self, file_name):
		"""
		Calculates the <pr|g|qs> integrals and stores them in file
		"""

		if file_name in os.listdir():
			print("Integral file already exsists. Not computing the integrals. ")
			return

		integral_dict_1 = {}
		integral_dict_2 = {}

		# 1-body integrals
		for p in range(1, self.dimension+1):
			for q in range(1, self.dimension+1):
				if p == q:
					I = self.calculate_1(p, q)
				else:
					I = 0

				integral_dict_1[(p, q)] = I

		# 2-body integrals
		for p in range(1, self.dimension+1):
			for q in range(1, p+1):
				for r in range(1, p):
					for s in range(1, r+1):
						I = self.calculate_2(p, r, q, s)

						integral_dict_2[(p, r, q, s)] = I
						integral_dict_2[(q, r, p, s)] = I
						integral_dict_2[(p, s, q, r)] = I
						integral_dict_2[(r, p, s, q)] = I
				r = p
				for s in range(1, q+1):
					I = self.calculate_2(p, r, q, s)
					
					integral_dict_2[(p, r, q, s)] = I
					integral_dict_2[(q, r, p, s)] = I
					integral_dict_2[(p, s, q, r)] = I
					integral_dict_2[(r, p, s, q)] = I

		np.save(file_name, np.array([integral_dict_1, integral_dict_2]))

		return

	def load_integrals(self, file_name):
		"""
		Loads the values of the integrals in this object
		"""

		self.integral_dict_1 = np.load(file_name)[0]
		self.integral_dict_2 = np.load(file_name)[1]

		return

	def get_1(self, p, q):
		"""
		Returns the value of the h_pq integrals

		Parameters
		----------
		p, q: int
			Indeces that specify the h_pq integral

		Returns
		-------
		I : float
			Value of the h_pq integral
		"""

		I = self.integral_dict_1[(p, q)]

		return I

	def get_2(self, p, r, q, s):
		"""
		Returns the value of the <pr|g|qs> integrals

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


def create_F_matrix(rho, integrals):
	"""
	Creates the Fock matrix with coefficients C, given by
	F[p,q] = h[p,q] + 2*J[p,q] - K[p,q]

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

	for p in range(Nbasis):
		for q in range(Nbasis):
			F[p, q] += integrals.get_1(p, q) # add h matrix
			for r in range(Nbasis):
				for s in range(Nbasis):
					F[p, q] += rho[r,s]*(integrals.get_2(p, q, r, s) - 0.5*integrals.get_2(p, r, q, s))

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
			E += 0.5*rho[p,q]*(integrals.get_1(p,q) + F[p,q])

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


def solve_Roothan_eqs(file_name, C_0, S, eps, i_max = 100):
	"""
	Solves the iterative generalized eigenvalue problem F(C)C = E*SC

	Parameters
	----------
	file_name: str
		Filename where the h_pq and <pr|g|qs> integrals are stored.
	C_0: np.ndarray(N, N)
		Initial coefficients
	S: np.ndarray(N, N)
		Overlap matrix
	eps: float
		Precision with which to find the iterative problem
	i_max: int
		Maximum number of iterations

	Returns
	-------
	E: np.ndarray(N)
		Vector with all the eigenvalues ordered from lowest to largest
	C: np.ndarray(N, N)
		Matrix with the coefficients of each eigenvector ordered as E
	"""

	counter = 1

	C_old = C_0
	E_old = 0

	while counter < i_max:
		F = create_F_matrix(file_name, C_old)
		E, C = eigh(F, S)

		if np.max(np.abs((E-E_old)/E)) < eps:
			break
		else:
			E_old = E
			C_old = C
			counter += 1

	return E, C