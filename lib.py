import numpy as np
from scipy.linalg import eigh

import mc_integration as mc
import basis_set as bs


class two_body_integrals():
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
			Indices that specify the h_pq integral

		Returns
		----------
		I : float
			Value of the h_pq integral
		"""
		nx,ny,nz = 1,1,1

		I = (p==q)*(bs.OMEGA_X*(nx + ny + 1) + bs.OMEGA_Z*(nz + 0.5)) # nx,ny,nz should come from the basis, so from p and q but how?

		return I
	
	###poner los indices com prqs

	def get_2(self, p, q, r, s):
		"""
		Returns the value of the <pr|g|qs> integrals

		Parameters
		----------
		p, q, r, s: int
			Indices that specify the <pr|g|qs> integral

		Returns
		----------
		I : float
			Value of the <pr|g|qs> integral
		"""

		
		I=0

		return I

		
	def calculate_1(self,p,q):
		"""
		Calculates the value of the h_pq integrals 

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
		Calculates the value of the <pr|g|qs> integrals

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
		integrand = bs.two_body_integrand
		indices = np.array([p,r,q,s])
		dimension = 6
		
		I = mc.MC_integration(integrand, indices, dimension, N_steps, N_walkers, N_skip, system_size)

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
	----------
	F: np.ndarray(N, N)
		Fock matrix
	"""

	Nbasis = C.shape[0]
	F = np.zeros((Nbasis, Nbasis))

	for p in range(Nbasis):
		for q in range(Nbasis):
			for k in range(Nbasis):
				for r in range(Nbasis):
					for s in range(Nbasis):
						F[p, q] += 2*np.conjugate(C[r, k])*C[s, k]*integrals.get_2(p, r, q, s) # add 2*J matrix
						F[p, q] += -np.conjugate(C[r, k])*C[s, k]*integrals.get_2(p, r, s, q) # add -K matrix
						F[p, q] += integrals.get_1(p, q) # add h matrix

	return F


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
	----------
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