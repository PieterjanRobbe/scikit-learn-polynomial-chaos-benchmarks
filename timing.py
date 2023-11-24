# import statements
import argparse
import chaospy as cp
import numpy as np
import os
import pickle as pk
import pygpc

# qualified import statements
from collections import OrderedDict
from math import factorial, prod
from time import time
from scipy.stats import uniform

# =============================================================================
# define a function to compute the PC expansion and the main sensitivity
# indices using sklearn
from sklearn.polynomial_chaos import PolynomialChaosRegressor

def main_sens_sklearn(X, y, degree):

    # fit pce
    pce = PolynomialChaosRegressor(uniform(), degree=degree)
    pce.fit(X, y)

    # return main sensitivity indices
    return pce.main_sens()

# =============================================================================
# define a function to compute the PC expansion and the main sensitivity
# indices using chaospy
def main_sens_chaospy(X, y, degree):

    # fit pce
    dist = cp.Iid(cp.Uniform(), X.shape[1])
    expansion = cp.generate_expansion(degree, dist)
    pce = cp.fit_regression(expansion, X.T, y)

    # return main sensitivity indices
    return cp.Sens_m(pce, dist)

# =============================================================================
# define a function to compute the PC expansion and the main sensitivity
# indices using pygpc
def main_sens_pygpc(X, y, degree):

    # extract dimension
    dimension = X.shape[1]

    # define parameters
    parameters = OrderedDict()
    for j in range(dimension):
        parameters[f"x{j + 1}"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])

    # define grid
    grid = pygpc.RandomGrid(parameters_random=parameters, coords=X)

    # define results
    results = y.reshape(-1, 1)

    # set options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [degree]*dimension
    options["order_max"] = degree
    options["interaction_order"] = degree
    options["error_type"] = "loocv"
    options["n_samples_validation"] = None
    options["fn_results"] = f"/tmp/pygpc_dim_{dimension}_deg_{degree}"
    options["save_session_format"] = ".pkl"
    options["verbose"] = False

    # define algorithm
    algorithm = pygpc.Static_IO(parameters=parameters, 
                                options=options,
                                grid=grid,
                                results=results)
    
    # initialize gpc session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()

    # read session
    session = pygpc.read_session(fname=session.fn_session,
                                 folder=session.fn_session_folder)
    
    # compute Sobol indices
    pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             algorithm="standard")
    sobol, gsens = pygpc.get_sens_summary(options["fn_results"], parameters)

    # return main sensitivity indices
    return sobol.values[:dimension].ravel()

# =============================================================================
# compute main sensitivity indices
def compute_main_sens(dimension, degree, method):

    # Let's fix the random seed for reproducibility.
    np.random.seed(2023)

    # First, let's define the parameters in the model.
    a = np.array([1, 2, 5, 10, 20, 50, 100, 500])
    a = a[:dimension]

    # compute number of regression points
    n = (dimension - 1) * factorial(degree + dimension) // factorial(degree) // factorial(dimension)

    # Next, let's generate some input/output data.
    distribution = uniform()
    X = distribution.rvs((n, dimension))
    y = prod((abs(4*X_j - 2) + a_j) / (1 + a_j) for a_j, X_j in zip(a, X.T))

    # compute main sensitivity indices
    if method == "sklearn":
        return main_sens_sklearn(X, y, degree)
    elif method == "chaospy":
        return main_sens_chaospy(X, y, degree)
    elif method == "pygpc":
        return main_sens_pygpc(X, y, degree)

# =============================================================================
# main function
def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=8, help="dimension")
    parser.add_argument("-k", "--degree", type=int, default=7, help="degree")
    parser.add_argument("-m", "--method", type=str, default="sklearn", help="{sklearn, chaospy, pygpc}")
    args = parser.parse_args()

    # compute main sensitivity indices
    start_time = time()
    main_sens = compute_main_sens(args.dimension, args.degree, args.method)
    duration = time() - start_time

    # save results
    os.makedirs("timings", exist_ok=True)
    file = f"main_sens_d{args.dimension}_k{args.degree}_{args.method}"
    with open(os.path.join("timings", file), "wb") as f:
        pk.dump(main_sens, f)
    file = f"duration_d{args.dimension}_k{args.degree}_{args.method}"
    with open(os.path.join("timings", file), "wb") as f:
        pk.dump(duration, f)

# =============================================================================
# use script as standalone
if __name__ == "__main__":
    main()