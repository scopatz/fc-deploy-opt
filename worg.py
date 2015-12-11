# coding: utf-8
import os
import sys
import uuid
import json
import time
import subprocess
from math import ceil
from copy import deepcopy

import george
import numpy as np
import pandas as pd
import cymetric as cym

import dtw

MONTH_SHUFFLE = (1, 7, 10, 4, 8, 6, 12, 2, 5, 9, 11, 3)
NULL_SCHEDULE = {'build_times': [{'val': 1}], 
                 'n_build': [{'val': 0}], 
                 'prototypes': [{'val': 'LWR'}]}
LWR_PROTOTYPE = {'val': 'LWR'}
OPT_H5 = 'opt.h5'
SIM_JSON = 'sim.json'
OT_JSON = 'once-through.json'

BASE_SIMS = {}

def base_sim(basename):
    """A cached base simulator."""
    if basename not in BASE_SIMS:
        with open(basename) as f:
            BASE_SIMS[basename] = json.load(f)
    sim = deepcopy(BASE_SIMS[basename])
    return sim

def deploy_inst_schedule(Θ):
    if np.sum(Θ) == 0:
        return NULL_SCHEDULE
    sched = {'build_times': {'val': []},
             'n_build': {'val': []},
             'prototypes': {'val': []}}
    build_times = sched['build_times']['val']
    n_build = sched['n_build']['val']
    prototypes = sched['prototypes']['val']
    m = 0
    for i, θ in enumerate(Θ):
        if θ <= 0:
            continue
        build_times.append(i*12 + MONTH_SHUFFLE[m])
        n_build.append(int(θ))
        prototypes.append('LWR')
        m = (m + 1) % 12
    return sched

def make_sim(Θ, basename=OT_JSON, inpname=SIM_JSON, **state):
    sim = base_sim(basename)
    inst = sim['simulation']['region']['institution']
    inst['config']['DeployInst'] = deploy_inst_schedule(Θ)
    with open(inpname, 'w') as f:
        json.dump(sim, f)
    return sim


# Simulate
# =========
# Now let's build some tools to run simulations and extract a GWe time series.

def run(inpname=SIM_JSON, dbname=OPT_H5, **state):
    """Runs a simulation and returns the sim id."""
    cmd = ['cyclus', '--warn-limit', '0', '-o', dbname, inpname]
    proc = subprocess.run(cmd, check=True, universal_newlines=True, 
                          stdout=subprocess.PIPE)
    simid = proc.stdout.rsplit(None, 1)[-1]
    return simid

month_to_year = lambda x: x//12
mwe_month_to_gwe_year = lambda x: 1e-3*x/12

def extract_gwe(simid, T, dbname=OPT_H5, **state):
    """Computes the annual GWe for a simulation."""
    zero_gwe = pd.DataFrame({'GWe': np.zeros(T)}, index=np.arange(T))
    zero_gwe.index.name = 'Time'
    with cym.dbopen(dbname) as db:
        evaler = cym.Evaluator(db)
        raw = evaler.eval('TimeSeriesPower', 
                          conds=[('SimId', '==', uuid.UUID(simid))])
    ano = pd.DataFrame({'Time': raw.Time.apply(month_to_year), 
                        'GWe': raw.Value.apply(mwe_month_to_gwe_year)})
    gwe = ano.groupby('Time').sum()
    gwe = (gwe + zero_gwe).fillna(0.0)
    return np.array(gwe.GWe)


# Distancing
# ========
# Now let's build some tools to distance between a GWe time series and 
# a demand curve.

#DEFAULT_DEMAND = 90 * (1.01**np.arange(YEARS))  # 1% growth

def d(f, g):
    """The dynamic time warping distance between a GWe time series and a 
    demand curve.
    """
    rtn = dtw.distance(f[:, np.newaxis], g[:, np.newaxis])
    return rtn

def gwed(Θ, f, simid_s, **state):
    """For a given deployment schedule Θ, return the GWe time series and 
    the distance to the demand function f.
    """
    make_sim(Θ, **state)
    simid = run(**state)
    simid_s.append(simid)
    gwe = extract_gwe(simid, **state)
    d_s = d(f, gwe)
    return gwe, d_s


def add_sim(Θ, f, Θs, G, D, Θ_s, G_s, D_s, sim_time_s, **state):
    """Add a simulation to the known simulations by performing the simulation.
    """
    t0 = time.time()
    g_s, d_s = gwed(Θ, f=f, **state)
    Θs.append(Θ)
    G.append(g_s)
    D.append(d_s)
    Θ_s.append(Θ)
    G_s.append(g_s)
    D_s.append(d_s)
    t1 = time.time()
    sim_time_s.append(t1 - t0)


# Optimizer
# =======
# Now let's add some tools to do the estimation phase of the optimization.

def gp_gwe(Θs, G, T, tol, N, **state):
    """Create a Gaussian process regression model for GWe."""
    S = len(G)
    t = np.arange(T)
    P = len(N)
    ndim = P + 1
    x = np.empty((S*T, ndim), dtype=int)
    y = np.concatenate(G)
    y_mean = np.mean(y)
    for i in range(S):
        x[i*T:(i+1)*T, 0] = t
        x[i*T:(i+1)*T, 1:] = Θs[i][np.newaxis,:]
    yerr = tol * y_mean
    kernel = float(y_mean) * george.kernels.Matern32Kernel(1.0, ndim=ndim)
    gp = george.GP(kernel, mean=y_mean)
    gp.compute(x, yerr=yerr, sort=False)
    gp.optimize(x, y, yerr=yerr, sort=False, verbose=False)
    return gp, x, y

def predict_gwe(Θ, gp, y, T):
    """Predict GWe for a deployment schedule Θ and a GP."""
    t = np.arange(T)
    P = len(Θ)
    ndim = P + 1
    x = np.empty((T, ndim), dtype=int)
    x[:,0] = t
    x[:,1:] = Θ[np.newaxis,:]
    mu = gp.predict(y, x, mean_only=True)
    return mu

def gp_d_inv(θ_p, D_inv, tol=1e-6):
    """Computes a Gaussian process model for a deployment parameter."""
    S = len(D_inv)
    ndim = 1
    x = θ_p
    y = D_inv
    y_mean = np.mean(y)
    yerr = tol * y_mean
    kernel = float(y_mean) * george.kernels.ExpSquaredKernel(1.0, ndim=ndim)
    gp = george.GP(kernel, mean=y_mean, solver=george.HODLRSolver)
    gp.compute(x, yerr=yerr, sort=False)
    gp.optimize(x, y, yerr=yerr, sort=False, verbose=False)
    return gp, x, y

def weights_p_poisson(D, θ_p, range_p):
    M_p = range_p[0]
    N_p = range_p[-1]
    p_min = np.argmin(D)
    lam = θ_p[p_min]
    fact = np.cumprod([1.0] + list(range(1, N_p + 1)))[M_p:N_p + 1]
    weights_p = np.exp(-lam) * (lam**range_p) / fact
    return weights_p

def weights(Θs, D, M, N, tol, **state):
    P = len(N)
    θ_ps = np.array(Θs)
    D = np.asarray(D)
    D_inv = D**-1 
    W = []
    for p in range(P):
        θ_p = θ_ps[:,p]
        range_p = np.arange(M[p], N[p] + 1)
        # try gaussian process of weights
        gp, _, _ = gp_d_inv(θ_p, D_inv, tol=tol)
        d_inv_np = gp.predict(D_inv, range_p, mean_only=True)
        if np.all(np.isnan(d_inv_np)) or np.all(d_inv_np <= 0.0):
            # try poisson, in event of failure
            d_inv_np = weights_p_poisson(D, θ_p, range_p)
        elif np.any(d_inv_np < 0.0):
            d_inv_np[d_inv_np < 0.0] = np.min(d_inv_np[d_inv_np > 0.0])
        # ensure they are normalized
        d_inv_np_tot = d_inv_np.sum()
        w_p = d_inv_np / d_inv_np_tot
        W.append(w_p)
    return W

def guess_scheds_stoch(gp, y, Γ, f, T, **state):
    """Guess a new deployment schedule, given a number of samples Γ, 
    weights W, and Guassian process for the GWe.
    """
    W = weights(**state)
    P = len(W)
    Θ_γs = np.empty((Γ, P), dtype=int)
    for p in range(P):
        w_p = W[p]
        Θ_γs[:, p] = np.random.choice(len(w_p), size=Γ, p=w_p)
    Δ = []
    for γ in range(Γ):
        Θ_γ = Θ_γs[γ]
        g_star = predict_gwe(Θ_γ, gp, y, T)
        d_star = d(f, g_star)
        Δ.append(d_star)
    γ = np.argmin(Δ)
    Θ_γ = Θ_γs[γ]
    return Θ_γ, Δ[γ]

def guess_scheds_inner(gp, y, Θs, f, M, N, **state):
    """Guess a new deployment schedule using an inner product sweep.
    """
    P = len(N)
    Θ = np.array(Θs[0], dtype=int)
    for p in range(P):
        d_p = []
        range_p = np.arange(M[p], N[p] + 1, dtype=int)
        for n_p in range_p:
            Θ[p] = n_p
            g_star = predict_gwe(Θ, gp, y, T=p+1)[:p+1]
            d_star = d(f[:p+1], g_star)
            d_p.append(d_star)
        Θ[p] = range_p[np.argmin(d_p)]
    return Θ, np.min(d_p)

def estimate(est_time_s, method_s, winner_s, hyperparameters_s, **state):
    """Runs an estimation step, returning a new deployment schedule."""
    t0 = time.time()
    gp, x, y = gp_gwe(**state)
    method = method_s[-1]
    if method == 'stochastic':
        Θ, dmin = guess_scheds_stoch(gp, y, **state)
        winner = 'stochastic'
    elif method == 'inner-prod':
        Θ, dmin = guess_scheds_inner(gp, y, **state)
        winner = 'inner-prod'
    elif method == 'all':
        Θ_stoch, dmin_stoch = guess_scheds_stoch(gp, y, **state)
        Θ_inner, dmin_inner = guess_scheds_inner(gp, y, **state)
        Θ, winner = (Θ_stoch, 'stochastic') if dmin_stoch < dmin_inner else \
                    (Θ_inner, 'inner-prod')
    else:
        raise ValueError('method {} not known'.format(method))
    winner_s.append(winner)
    hyperparameters_s.append(gp.kernel[:])
    t1 = time.time()
    est_time_s.append(t1 - t0)
    return Θ

def str_current(state):
    """Prints the most recent iteration."""
    s = state['s']
    i = s - 1
    x = 'Simulation {0}\n'.format(s)
    x += '-'*(len(x) - 1) + '\n'
    x += 'SimId {0}\n'.format(state['simid_s'][i])
    x += 'hyperparameters: {0}\n'.format(state['hyperparameters_s'][i])
    x += 'Estimate method is {0!r}\n'.format(state['method_s'][i])
    x += 'Estimate winner is {0!r}\n'.format(state['winner_s'][i])
    estt = state['est_time_s'][i]
    x += 'Estimate time:   {0} min {1} sec\n'.format(estt//60, estt%60)
    simt = state['sim_time_s'][i]
    x += 'Simulation time: {0} min {1} sec\n'.format(simt//60, simt%60)
    x += 'D: {0}\n'.format(state['D'])
    return x

def print_current(state):
    """Prints the most recent iteration."""
    print(str_current(state))
    sys.stdout.flush()

def optimize(f, N, M=None, z=2, MAX_D=0.1, MAX_S=12, T=None, Γ=None, tol=1e-6, 
             method_0='all', basename=OT_JSON, inpname=SIM_JSON, 
             dbname=OPT_H5, verbose=False, seed=None):
    # state initialization
    state = {'f': f, 'N': N, 'verbose': verbose, 'z': z, 's': 0, 'seed': seed,
             'basename': basename, 'inpname': inpname, 'dbname': dbname, 
             'tol': tol}
    T = state['T'] = len(N) if T is None else T
    M = state['M'] = np.zeros(T, dtype=int)
    Γ = state['Γ'] = int(np.sum(N - M)) if Γ is None else Γ
    if os.path.isfile(dbname):
        os.remove(dbname)
    if seed is not None:
        np.random.seed(seed)
    # horizon init
    Θs = state['Θs'] = []
    G = state['G'] = []
    D = state['D'] = []
    # per simulation init
    Θ_s = state['Θ_s'] = []
    G_s = state['G_s'] = []
    D_s = state['D_s'] = []
    est_time_s = state['est_time_s'] = [0.0, 0.0]
    sim_time_s = state['sim_time_s'] = []
    simid_s = state['simid_s'] = []
    method_s = state['method_s'] = [method_0]*2
    winner_s = state['winner_s'] = [method_0]*2
    hyperparameters_s = state['hyperparameters_s'] = [None]*2
    # run initial conditions
    add_sim(M, **state)  # lower bound
    add_sim(N, **state)  # upper bound
    s = state['s'] = 2
    while MAX_D < D[-1] and s < MAX_S:
        # set estimation method
        method_s.append(method_0)
        if method_0 == 'all' and (s%4 < 2):
            method_s[-1] = 'stochastic'
        # estimate and run sim
        Θ = estimate(**state)
        add_sim(Θ, **state)
        # figure out if this was worth doing
        idx = [int(i) for i in np.argsort(D)[:z]]
        if D[-1] == max(D):
            idx.append(-1)
        Θs = state['Θs'] = [Θs[i] for i in idx]
        G = state['G'] = [G[i] for i in idx]
        D = state['D'] = [D[i] for i in idx]
        s = state['s'] = (s + 1)
        if verbose:
            print_current(state)

