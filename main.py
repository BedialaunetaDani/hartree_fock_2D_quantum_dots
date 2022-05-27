import numpy as np
import lib
from scipy.linalg import eigh

############################################

N_electrons = 2
integrals_file = "integrals.csv"

normalized_wf = True

max_iter_SCF = 100
eps_SCF = 1E-4

############################################

# Previous calculations to HF
N_basis = N_electrons
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
E_old = np.inf
while n_iterations < max_iter_SCF:
	n_iterations += 1

	F = lib.create_F_matrix(C, integrals)
	if not normalized_wf:
		F = np.conjugate(X.transpose()) @ F @ X

	E, C = eigh(F, S)

	if np.max(np.abs((E - E_old)/E)) < eps_SCF:
		break
	
	E_old = E
	if not normalized_wf:
		C = X @ C

	print("N(SCF) = {}".format(n_iterations))