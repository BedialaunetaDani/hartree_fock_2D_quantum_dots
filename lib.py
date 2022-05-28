import numpy as np
from scipy.linalg import eigh
import os


class integral_master():
	"""
	Calculates, stores and retrieves the values of the <pr|g|qs> integrals
	"""
	def __init__(self):
		self.integral_list = None
		return

	def calculate(self, file_name):
		"""
		Calculates the <pr|g|qs> integrals and stores them in file
		"""

		if file_name in os.listdir():
			print("Integral file already exsists. Not computing the integrals. ")
			return

		return

	def load_integrals(self, file_name):
		"""
		Loads the values of the integrals in this object
		"""
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

		I = 0

		return I

	def get_2(self, p, q, r, s):
		"""
		Returns the value of the <pr|g|qs> integrals

		Parameters
		----------
		p, q, r, s: int
			Indeces that specify the <pr|g|qs> integral

		Returns
		-------
		I : float
			Value of the <pr|g|qs> integral
		"""

		I = 0

		return I


def create_F_matrix(C, integrals):
	"""
	Creates the Fock matrix with coefficients C, given by
	F[p,q] = h[p,q] + 2*J[p,q] - K[p,q]

	Parameters
	----------
	C: np.ndarray(N, N)
		Coefficient matrix
	integrals : two_body_integrals() class
		Class with all the information regarding the <pr|g|qs> integrals

	Returns
	-------
	F: np.ndarray(N, N)
		Fock matrix
	"""

	Nbasis = C.shape[0]
	F = np.zeros((Nbasis, Nbasis))

	for p in range(Nbasis):
		for q in range(Nbasis):
			F[p, q] += integrals.get_1(p, q) # add h matrix
			for k in range(Nbasis):
				for r in range(Nbasis):
					for s in range(Nbasis):
						F[p, q] += 2*np.conjugate(C[r, k])*C[s, k]*integrals.get_2(p, r, q, s) # add 2*J matrix
						F[p, q] += -np.conjugate(C[r, k])*C[s, k]*integrals.get_2(p, r, s, q) # add -K matrix

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
			E += rho[p,q]*(integrals.get_1(p,q) + 0.5*F[p,q])

	return E


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