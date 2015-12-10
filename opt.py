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

DURATION = BASE_SIM['simulation']['control']['duration']
YEARS = ceil(DURATION / 12)
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

def make_sim(Θ, basename=OT_JSON, inpname=SIM_JSON):
    sim = deepcopy(BASE_SIM)
    inst = sim['simulation']['region']['institution']
    inst['config']['DeployInst'] = deploy_inst_schedule(Θ)
    with open(inpname, 'w') as f:
        json.dump(sim, f)
    return sim



# Simulate
# =========
# Now let's build some tools to run simulations and extract a GWe time series.

def run(fname='sim.json', out=OPT_H5):
    """Runs a simulation and returns the sim id."""
    cmd = ['cyclus', '--warn-limit', '0', '-o', out, fname]
    proc = subprocess.run(cmd, check=True, universal_newlines=True, stdout=subprocess.PIPE)
    simid = proc.stdout.rsplit(None, 1)[-1]
    return simid

ZERO_GWE = pd.DataFrame({'GWe': np.zeros(YEARS)}, index=np.arange(YEARS))
ZERO_GWE.index.name = 'Time'

def extract_gwe(simid, out=OPT_H5):
    """Computes the annual GWe for a simulation."""
    with cym.dbopen(out) as db:
        evaler = cym.Evaluator(db)
        raw = evaler.eval('TimeSeriesPower', conds=[('SimId', '==', uuid.UUID(simid))])
    ano = pd.DataFrame({'Time': raw.Time.apply(lambda x: x//12), 
                        'GWe': raw.Value.apply(lambda x: 1e-3*x/12)})
    gwe = ano.groupby('Time').sum()
    gwe = (gwe + ZERO_GWE).fillna(0.0)
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


#N = np.asarray(np.ceil(4*(1.01)**np.arange(YEARS)), dtype=int)  # max annual deployments

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

def gp_gwe(Θs, G, α, T=None, tol=1e-6, verbose=False):
    """Create a Gaussian process regression model for GWe."""
    S = len(G)
    T = YEARS if T is None else T
    t = np.arange(T)
    P = len(Θs[0])
    ndim = P + 1 - α
    y_mean = np.mean(G)
    y = np.concatenate(G)
    x = np.empty((S*T, ndim), dtype=int)
    for i in range(S):
        x[i*T:(i+1)*T, 0] = t
        x[i*T:(i+1)*T, 1:] = Θs[i][np.newaxis, α:]
    yerr = tol * y_mean
    #kernel = float(y_mean) * george.kernels.ExpSquaredKernel(1.0, ndim=ndim)
    #for p in range(P):
    #    kernel *= george.kernels.ExpSquaredKernel(1.0, ndim=ndim)
    #kernel = float(y_mean) * george.kernels.Matern52Kernel(1.0, ndim=ndim)
    kernel = float(y_mean) * george.kernels.Matern32Kernel(1.0, ndim=ndim)
    gp = george.GP(kernel, mean=y_mean)
    gp.compute(x, yerr=yerr, sort=False)
    gp.optimize(x, y, yerr=yerr, sort=False, verbose=verbose)
    return gp, x, y

def predict_gwe(Θ, gp, y, α, T=None):
    """Predict GWe for a deployment schedule Θ and a GP."""
    T = YEARS if T is None else T
    t = np.arange(T)
    P = len(Θ)
    ndim = P + 1 - α
    x = np.empty((T, ndim), dtype=int)
    x[:,0] = t
    x[:,1:] = Θ[np.newaxis,α:]
    mu = gp.predict(y, x, mean_only=True)
    return mu

def gp_d_inv(θ_p, D_inv, tol=1e-6, verbose=False):
    """Computes a Gaussian process model for a deployment parameter."""
    S = len(D)
    ndim = 1
    x = θ_p
    y = D_inv
    y_mean = np.mean(y)
    yerr = tol * y_mean
    kernel = float(y_mean) * george.kernels.ExpSquaredKernel(1.0, ndim=ndim)
    gp = george.GP(kernel, mean=y_mean, solver=george.HODLRSolver)
    gp.compute(x, yerr=yerr, sort=False)
    gp.optimize(x, y, yerr=yerr, sort=False, verbose=verbose)
    return gp, x, y

def weights(Θs, D, N, Nlower, α, tol=1e-6, verbose=False):
    P = len(N)
    θ_ps = np.array(Θs)
    D = np.asarray(D)
    D_inv = D**-1 
    W = [None] * α
    for p in range(α, P):
        θ_p = θ_ps[:,p]
        range_p = np.arange(Nlower[p], N[p] + 1)
        gp, _, _ = gp_d_inv(θ_p, D_inv, tol=tol, verbose=verbose)
        d_inv_np = gp.predict(D_inv, range_p, mean_only=True)
        #p_min = np.argmin(D)
        #lam = θ_p[p_min]
        #fact = np.cumprod([1.0] + list(range(1, N[p] + 1)))[Nlower[p]:N[p] + 1]
        #d_inv_np = np.exp(-lam) * (lam**range_p) / fact
        if np.all(np.isnan(d_inv_np)) or np.all(d_inv_np <= 0.0):
            # try D, instead of D^-1
            #gp, _, _ = gp_d_inv(θ_p, D, tol=tol, verbose=verbose)
            #d_np = gp.predict(D, np.arange(0, N[p] + 1), mean_only=True)
            # try setting the shortest d to 1, all others 0.
            #d_inp_np = np.zeros(N[p] + 1, dtype='f8')
            #p_min = np.argmin(D)
            #d_inv_np[np.argwhere(θ_p[p_min] == range_p)] = 1.0
            # try Poisson dist centered at min.
            p_min = np.argmin(D)
            lam = θ_p[p_min]
            fact = np.cumprod([1.0] + list(range(1, N[p] + 1)))[Nlower[p]:N[p] + 1]
            d_inv_np = np.exp(-lam) * (lam**range_p) / fact
        if np.any(d_inv_np < 0.0):
            d_inv_np[d_inv_np < 0.0] = np.min(d_inv_np[d_inv_np > 0.0])
        d_inv_np_tot = d_inv_np.sum()
        w_p = d_inv_np / d_inv_np_tot
        W.append(w_p)
    return W

def guess_scheds(Θs, W, Γ, gp, y, α, T=None):
    """Guess a new deployment schedule, given a number of samples Γ, weights W, and 
    Guassian process for the GWe.
    """
    P = len(W)
    Θ_γs = np.empty((Γ, P), dtype=int)
    Θ_γs[:, :α] = Θs[0][:α]
    for p in range(α, P):
        w_p = W[p]
        Θ_γs[:, p] = np.random.choice(len(w_p), size=Γ, p=w_p)
    Δ = []
    for γ in range(Γ):
        Θ_γ = Θ_γs[γ]
        g_star = predict_gwe(Θ_γ, gp, y, α, T=T)
        d_star = d(g_star)
        Δ.append(d_star)
    γ = np.argmin(Δ)
    Θ_γ = Θ_γs[γ]
    print('hyperparameters', gp.kernel[:])
    #print('Θ_γs', Θ_γs)
    #print('Θ_γs[γ]', Θ_γs[γ])
    #print('Predition', Δ[γ], Δ)
    return Θ_γ, Δ[γ]

def guess_scheds_loop(Θs, gp, y, N, Nlower):
    """Guess a new deployment schedule, given a number of samples Γ, weights W, and 
    Guassian process for the GWe.
    """
    P = len(N)
    Θ = np.array(Θs[0], dtype=int)
    for p in range(P):
        d_p = []
        range_p = np.arange(Nlower[p], N[p] + 1, dtype=int)
        for n_p in range_p:
            Θ[p] = n_p
            g_star = predict_gwe(Θ, gp, y, α=0, T=p+1)[:p+1]
            d_star = d(g_star, f=DEFAULT_DEMAND[:p+1])
            d_p.append(d_star)
        Θ[p] = range_p[np.argmin(d_p)]
    print('hyperparameters', gp.kernel[:])
    return Θ, np.min(d_p)

def estimate(Θs, G, D, N, Nlower, Γ, α, T=None, tol=1e-6, verbose=False, method='stochastic'):
    """Runs an estimation step, returning a new deployment schedule."""
    gp, x, y = gp_gwe(Θs, G, α, T=T, tol=tol, verbose=verbose)
    if method == 'stochastic':
        # orig
        W = weights(Θs, D, N, Nlower, α, tol=tol, verbose=verbose)
        Θ, dmin = guess_scheds(Θs, W, Γ, gp, y, α, T=T)
    elif method == 'inner-prod':
        # inner prod
        Θ, dmin = guess_scheds_loop(Θs, gp, y, N, Nlower)
    elif method == 'all':
        W = weights(Θs, D, N, Nlower, α, tol=tol, verbose=verbose)
        Θ_stoch, dmin_stoch = guess_scheds(Θs, W, Γ, gp, y, α, T=T)
        Θ_inner, dmin_inner = guess_scheds_loop(Θs, gp, y, N, Nlower)
        if dmin_stoch < dmin_inner:
            winner = 'stochastic'
            Θ = Θ_stoch
        else:
            winner = 'inner'
            Θ = Θ_inner
        print('Estimate winner is {}'.format(winner))
    else:
        raise ValueError('method {} not known'.format(method))
    return Θ

def optimize(f, N, M=None, z=2, MAX_D=0.1, MAX_S=12, T=None, Γ=None, tol=1e-6, 
             method_0='all', inpname=SIM_JSON, dbname=OPT_H5, verbose=False,
             seed=None):
    # state initialization
    state = {'f': f, 'N': N, 'verbose': verbose, 'z': z, 's': 0, 'seed': seed,
             'inpname': inpname, 'dbname': dbname}
    M = state['M'] = np.zeros(len(N), dtype=int)
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
    est_time_s = state['est_time_s'] = []
    sim_time_s = state['sim_time_s'] = []
    simid_s = state['simid_s'] = []
    method_s = state['method_s'] = [method_0]*2
    # run initial conditions
    add_sim(M, **start)  # lower bound
    add_sim(N, **start)  # upper bound
    while MAX_D < D[-1] and s < MAX_S:
        print(s)
        print('-'*18)
        Gprev = np.array(G[:z])
        t0 = time.time()
        method = 'stochastic' if s%4 < 2 else 'all'
        Θ = estimate(Θs, G, D, N, M, Γ, T=T, tol=tol, verbose=verbose, method=method)
        t1 = time.time()
        α_s = add_sim(Θ, dtol=dtol)
        t2 = time.time()
        print('Estimate time:   {0} min {1} sec'.format((t1-t0)//60, (t1-t0)%60))
        print('Simulation time: {0} min {1} sec'.format((t2-t1)//60, (t2-t1)%60))
        print(D)
        sys.stdout.flush()
        idx = [int(i) for i in np.argsort(D)[:z]]
        if D[-1] == max(D):
            idx.append(-1)
        if (len(D) == idx[0] + 1):
            print('Update α: {0} -> {1}'.format(α, α_s))
            α = α_s
        Θs = [Θs[i] forz i in idx]
        G = [G[i] for i in idx]
        D = [D[i] for i in idx]
        s += 1
        print()

