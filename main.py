import numpy as np
import lib
from scipy.linalg import eigh

############################################

N_electrons = 2
N_basis = 2
integrals_file = "integrals.csv"

normalized_wf = True

max_iter_SCF = 100
eps_SCF = 1E-4

############################################

# Previous calculations to HF
integrals = lib.integral_master()
C = np.random.rand(N_basis, N_basis)

	# One- and Two-body integrals
integrals.calculate(integrals_file)

	# Normalization of wave functions
if not normalized_wf:
	# calculate S matrix
	# (...)
	SVAL, SVEC = np.linalg.eigh(S) 
	SVAL_minhalf = (np.diag(SVAL**(-0.5))) 
	X = np.dot(SVEC, np.dot(SVAL_minhalf, np.transpose(SVEC)))
else:
	S = np.eye(N_basis)

# Self Consistent Field
n_iterations = 0
total_E_old = np.inf

while n_iterations < max_iter_SCF:
	n_iterations += 1

	F = lib.create_F_matrix(C, integrals)
	
	if normalized_wf:
		E, C = eigh(F, S)
	else:
		F_prime = np.conjugate(X.transpose()) @ F @ X
		E, C_prime = eigh(F_prime, S)
		C = X @ C_prime
	
	rho = lib.density_matrix(C, N_electrons)
	total_E = lib.total_energy(rho, F, integrals)

	if np.max(np.abs((total_E - total_E_old) / total_E)) < eps_SCF:
		break
	
	total_E_old = total_E

	print("E = {:0.7f} | N(SCF) = {}".format(total_E, n_iterations))