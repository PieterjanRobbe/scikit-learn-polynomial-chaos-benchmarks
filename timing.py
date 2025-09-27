# import statements
import argparse
import numpy as np
import os
import pickle as pk

# qualified import statements
from collections import OrderedDict
from math import factorial, prod
from time import time
from scipy.stats import uniform

# =============================================================================
# define a function to compute the PC expansion and the main sensitivity
# indices using sklearn
from sklearn.polynomial_chaos import PolynomialChaosExpansion

def main_sens_sklearn(X, y, degree):

    # fit pce
    pce = PolynomialChaosExpansion(uniform(), degree=degree)
    pce.fit(X, y)

    # return main sensitivity indices
    return pce.main_sens().flatten()

# =============================================================================
# define a function to compute the PC expansion and the main sensitivity
# indices using chaospy
import chaospy as cp

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
import pygpc

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
# define a function to compute the PC expansion and the main sensitivity
# indices using openturns
import openturns as ot

def main_sens_openturns(X, y, degree):
    # Ensure shapes and convert to OT Samples
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    in_s = ot.Sample(X.tolist())
    out_s = ot.Sample(y.tolist())

    dimension = X.shape[1]

    # Joint distribution: independent Uniform(0,1) per input
    marginals = [ot.Uniform(0.0, 1.0) for _ in range(dimension)]
    distribution = ot.ComposedDistribution(marginals)  # (Independent copula by default)

    # Polynomial basis consistent with each marginal
    poly_factories = [
        ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i))
        for i in range(dimension)
    ]
    enumerate_fn = ot.LinearEnumerateFunction(dimension)
    product_basis = ot.OrthogonalProductPolynomialFactory(poly_factories, enumerate_fn)

    # Total-degree ≤ degree
    index_max = enumerate_fn.getStrataCumulatedCardinal(degree)

    # Fixed basis size strategy
    adaptive_strategy = ot.FixedStrategy(product_basis, index_max)

    # Least-squares PCE on provided design (X, y)
    algo = ot.FunctionalChaosAlgorithm(in_s, out_s, distribution, adaptive_strategy)
    algo.run()
    result = algo.getResult()

    # Sobol' main indices
    sobol = ot.FunctionalChaosSobolIndices(result)
    return np.array([sobol.getSobolIndex(i) for i in range(dimension)])

# =============================================================================
# compute main sensitivity indices
def compute_main_sens(dimension, degree, method):

    # First, let's define the parameters in the model.
    a = np.array([1, 2, 5, 10, 20, 50, 100, 500])
    a = a[:dimension]

    # compute number of regression points
    n = (dimension - 1) * factorial(degree + dimension) // factorial(degree) // factorial(dimension)

    # Next, let's generate some input/output data.
    distribution = uniform()
    X = distribution.rvs((n, dimension), random_state=2025)
    y = prod((abs(4*X_j - 2) + a_j) / (1 + a_j) for a_j, X_j in zip(a, X.T))

    # compute main sensitivity indices
    if method == "sklearn":
        return main_sens_sklearn(X, y, degree)
    elif method == "chaospy":
        return main_sens_chaospy(X, y, degree)
    elif method == "pygpc":
        return main_sens_pygpc(X, y, degree)
    elif method == "openturns":
        return main_sens_openturns(X, y, degree)

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
    print(main_sens)
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