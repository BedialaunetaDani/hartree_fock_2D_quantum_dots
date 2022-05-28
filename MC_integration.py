import multiprocessing
import numpy as np
from scipy.optimize import brentq 
import os


#######################################
#	    MONTE CARLO INTEGRATION       #
#######################################

# ---WALKER GENERATION---

def random_walker(function, indices, N_steps, init_point, trial_move):
	"""
	Returns steps of a random walker that follows a Markov chain given a probability
	density function using the Metropolis algorithm. 

	Parameters
	----------
	function : function(r)
		Function integrated over r
	N_steps : int
		Number of steps that the random walker takes
	init_point : np.ndarray(dim)
		Starting point of the random walker
	trial_move : float
		Standard deviation that defines the trial move according to a normal distribution

	Returns
	-------
	steps : np.ndarray(N_steps, dim)
		Steps of the random walker
	acceptance_probability: np.ndarray(N_steps)
		Acceptance probability of a walker at each point
	acceptance_ratio : float
		Acceptance ratio of the random walker
	"""

	dim = init_point.shape[0]
	steps = np.zeros((N_steps,dim))
	steps[0] = init_point
	acceptance_ratio = 0
	acceptance_probability = np.zeros(N_steps)

	for i in np.arange(1, N_steps):

		next_point = steps[i-1] + np.random.normal(0, trial_move, size=dim)
		step_acceptance = function(next_point, indices)/function(steps[i-1], indices)
		acceptance_probability[i] = min(step_acceptance,1)

		if (np.random.rand(1) <= step_acceptance).all():
			steps[i] = next_point
			acceptance_ratio += 1/N_steps
		else:
			steps[i] = steps[i-1]
	
	return steps, acceptance_probability, acceptance_ratio


def random_walkers(function, indices, N_steps, init_points, trial_move):
	"""
	Returns steps of N_walkers random walkers that follows a Markov chain given a probability
	density function using the Metropolis algorithm. 

	Parameters
	----------
	function : function(r)
		Function integrated over r
	N_steps : int
		Number of steps that each random walker takes
	init_point : np.ndarray(N_walkers, dim)
		Starting points of all random walkers
	trial_move : float
		Standard deviation that defines the trial move according to a normal distribution

	Returns
	-------
	steps : np.ndarray(N_steps, N_walkers, dim)
		Steps of the random walker
	acceptance_probability: np.ndarray(N_steps)
		Acceptance probability of a walker at each point
	acceptance_ratio : np.ndarray(N_walkers)
		Acceptance ratio of the random walkers
	"""

	N_walkers, dim = init_points.shape
	steps = np.zeros((N_steps, N_walkers, dim))
	steps[0] = init_points
	acceptance_ratio = np.zeros(N_walkers)
	acceptance_probability = np.zeros(N_steps)

	for i in np.arange(1, N_steps):

		next_point = steps[i-1] + np.random.normal(0, trial_move, size=(N_walkers,dim))
		step_acceptance = np.minimum(function(next_point, indices)/function(steps[i-1], indices),1)
		acceptance_probability[i] = step_acceptance[0]
		to_change = np.where(np.random.rand(N_walkers) <= step_acceptance)

		steps[i] = steps[i-1]
		steps[i, to_change] = next_point[to_change]

		acceptance_ratio[to_change] += 1/N_steps

	return steps, acceptance_probability, acceptance_ratio


#---WALKER INTIALIZATION---

def rand_init_point(system_size, dim, N_points):
	"""
	Returns a random initial point for the random walkers given a typical size of the system according to a normal distribution. 

	Parameters
	----------
	system_size : float
		Typical size of the system, e.g. for the Hydrogen atom it could be 2*a_0
	dim : int
		Dimension of the configuration space, i.e. number of degrees of freedom in the system
	N_points : int
		Number of initial points with dimension dim

	Returns
	-------
	init_point : np.ndarray(N_points, dim)
		Random initial point for the random walkers
	"""

	init_point = np.random.normal(scale=system_size, size=(N_points, dim))

	return init_point


def dev_acceptance_ratio(trial_move, function, indices, dim, N_av=100):
	"""
	Returns the deviation of the acceptance ratio from 0.5 for a random walker with given N_av steps 

	Parameters
	----------
	trial_move : float
		Current initial trial move variance
	function : function(r)
		Function integrated over r
	dim : int
		Dimension of the configuration space, i.e. number of degrees of freedom in the system
	N_av : int
		Number of steps the walker takes to compute the acceptance ratio average

	Returns
	-------
	dev_acceptance_ratio : float
		Average acceptance ratio for a random walker given N_av steps
	"""

	acceptance_ratio = 0
	steps = np.zeros((N_av, dim))
	steps[0] = np.zeros(dim)

	for i in np.arange(1, N_av):
		next_point = steps[i-1] + np.random.normal(scale=trial_move, size=(dim))
		ratio = min(function(next_point, indices)/function(steps[i-1], indices), 1)
		if np.random.rand(1) <= ratio:
			steps[i] = next_point
		else:
			steps[i] = steps[i-1]
		acceptance_ratio += ratio

	dev_ratio = acceptance_ratio/N_av - 0.5

	return dev_ratio


def find_optimal_trial_move(function, indices, dim, trial_move_init, maxiter=5000, N_av=2000, tol=0.05):
	"""
	Returns trial_move such that the corresponding acceptance ratio is 0.5+-tol. 

	Parameters
	----------
	function : function(r)
		Function integrated over r
	dim : int
		Dimension of the configuration space, i.e. number of degrees of freedom in the system
	trial_move_init : float
		Initial guess of the initial trial move variance
	maxiter : int
		Maximum number of iterations 
	N_av : int
		Number of steps the walker takes to compute the acceptance ratio average
	tol : float
		Tolerance for the deviation of the acceptance ratio from 0.5

	Returns
	-------
	opt_trial_move: float
		trial_move such that the corresponding acceptance ratio is 0.5 +- tol
	"""

	arguments = (function, indices, dim, N_av)

	# Find a zero in dev_av_rate between 10*trial_move_init and trial_move_init/100,
	# the function must be of oposite signs at the two points
	opt_trial_move = brentq(dev_acceptance_ratio, trial_move_init/1000, 10*trial_move_init, args=arguments, maxiter=maxiter, xtol=tol) 
												
	return opt_trial_move


#---MONTE CARLO INTEGRATION---

def MC_integration_core(function, indices, dim, trial_move, file_name, N_steps=5000, N_walkers=250, N_skip=0, L_start=1):
	"""
	Returns expectation value of the energy E(alpha) averaged over N_walkers random walkers using Monte Carlo integration. 
	This function is used for parallelization of the MC integration. 

	Parameters
	----------
	function : function(r)
		Function integrated over r
	dim : int
		Dimension of the configuration space, i.e. number of degrees of freedom in the system
	trial_move : float
		Trial move for the random walkers
	file_name : str
		Name of the file to store the E_alpha values for each walker
	N_steps : int
		Number of steps that the random walker takes
	N_walkers : int
		Number of random walkers
	N_skip : int
		Number of initial steps to skip for the integration
	L_start : float
		Length of the box in which the random walkers are initialized randomly

	Returns
	-------
	None
	"""

	init_points = rand_init_point(L_start, dim, N_walkers)
	steps, _, acceptance_ratio = random_walkers(function, indices,N_steps, init_points, trial_move)
	steps = steps[N_skip:, :, :]

	E_alpha_walkers = MC_average_walkers(function, steps, indices)

	f = open(file_name, "w")
	f.write("{:0.35f}\n".format(np.mean(acceptance_ratio)))
	for E_alpha in E_alpha_walkers:
		f.write("{:0.35f}\n".format(E_alpha))
	f.close()

	return 


def MC_integration(function, indices, dim, N_steps=10000, N_walkers=400, N_skip=1000, L_start=2, N_cores=-1, trial_move=None):
	"""
	Returns expectation value of the energy E(alpha) averaged over N_walkers random walkers using Monte Carlo integration. 

	Parameters
	----------
	function : function(r)
		Function integrated over r
	dim : int
		Dimension of the configuration space, i.e. number of degrees of freedom in the system
	N_steps : int
		Number of steps that the random walker takes
	N_walkers : int
		Number of random walkers
	N_skip : int
		Number of initial steps to skip for the integration
	L_start : float
		Length of the box in which the random walkers are initialized randomly
	N_cores : int
		Number of cores to use for parallisation
		If -1, it sets to the maximum number of cores available
	trial_move : float or NoneType
		Trial move for the random walkers
		If None, finds the optimal trial move. 

	Returns
	-------
	E_alpha : float
		Expectation value of the energy for given parameters of the trial wave function
	E_alpha_std : float
		Standard deviation of E_alpha computed from E_alpha_walkers (E_alpha for each walker)
	acceptance_ratio : float
		Acceptance ratio of the random walkers
	"""

	if trial_move is None:
		trial_move = find_optimal_trial_move(function, dim, 0.5*L_start) 
		print("Optimal trial_move is", trial_move, end="\r")

	# separate number of walkers for each core (multiprocessing)
	if N_cores == -1: N_cores = multiprocessing.cpu_count()
	N_walkers_per_core = int(N_walkers/N_cores)
	N_walkers_last_core = N_walkers - (N_cores-1)*N_walkers_per_core
	list_N_walkers = np.array([N_walkers_per_core]*(N_cores - 1) + [N_walkers_last_core])

	inputs = [(function, indices, dim, trial_move, "output_core{}.csv".format(i), N_steps, N, N_skip, L_start) for i, N in enumerate(list_N_walkers)]

	# multiprocessing
	with multiprocessing.Pool(processes=N_cores) as pool: 
		data_outputs = pool.starmap(MC_integration_core, inputs)

	# load data
	E_alpha_walkers = []
	acceptance_ratio = []
	for i in range(N_cores):
		f = open("output_core{}.csv".format(i), "r")
		data = f.read()
		f.close()
		acceptance_ratio += [float(data[:data.index("\n")])]
		E_alpha_walkers += [float(E) for E in data.split("\n")[1:-1]]
		os.remove("output_core{}.csv".format(i))

	E_alpha_walkers = np.array(E_alpha_walkers)
	acceptance_ratio = np.array(acceptance_ratio)

	# average and std
	E_alpha = np.average(E_alpha_walkers)
	E_alpha_std = np.std(E_alpha_walkers) / np.sqrt(N_walkers)
	acceptance_ratio = np.average(acceptance_ratio)

	return E_alpha, E_alpha_std, acceptance_ratio, trial_move


def MC_average_walkers(function, steps, indices):
	"""
	Computes expectation value of energy given a distribution of steps

	Parameters
	----------
	function : function(r)
		Function integrated over r
	steps : np.ndarray(N_steps, N_walkers, dim)
		Points to be used in the computation of the integral
		N_steps is the number of steps each walker takes
		N_walkers is the number of walkers
		dim is the dimension of the integral space
	
	Returns
	-------
	E_alpha : np.ndarray(N_walkers)
		Expectation value of the energy for given parameters of the trial wave function
	"""

	E_local = function(steps, indices) # E_local.shape = (N_steps, N_walkers)
	E_alpha_walkers = np.average(E_local, axis=0) # E_alpha_walkers.shape = (N_walkers)

	return E_alpha_walkers


#######################################
#			  SAVE RESULTS
#######################################

def save(file_name, alpha_list, data_list, alpha_labels=None, data_labels=["E", "var(E)", "acceptance_ratio", "trial_move"]):
	"""
	Saves alpha and data results to file_name.

	Parameters
	----------
	file_name : str
		Name of the CSV file to store the results
	alpha_list : list of np.ndarray
		List of alpha values
	data_list : list of np.ndarray
		List of data values
	alpha_labels : list of str
		Labels for the elements of alpha_list[i]
		If None, it assumes labels: alpha1, alpha2, alpha3, ...
	data_labels : list of str
		Labels for the elements of data_list[i]
		If None, it assumes labels: data1, data2, data3, ...

	Returns
	-------
	None
	"""

	f = open(file_name, "w")
	header = ""
	if alpha_labels is None:
		header += ",".join(["alpha{}".format(i+1) for i in range(len(alpha_list[0]))]) + ","
	else: 
		header += ",".join(alpha_labels) + ","
	if data_labels is None:
		header += ",".join(["data{}".format(i+1) for i in range(len(data_list[0]))]) + ","
	else: 
		header += ",".join(data_labels)
	header += "\n"
	f.write(header)

	for k in range(len(alpha_list)):
		s = ("{},"*len(alpha_list[k])).format(*alpha_list[k])
		s += ("{},"*len(data_list[k])).format(*data_list[k])
		s = s[:-1] # deletes last comma
		s += "\n"
		f.write(s)

	f.close()

	return 