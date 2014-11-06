#!/usr/bin/env

import dadi
import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pandas as pd
import sys

## parse the snps file
dd = dadi.Misc.make_data_dict('oaks.dadi.snps')

## extract the fs from this dict with data not projected
## down to fewer than the total samples
proj = [10,6,6]
print >>sys.stderr, "PROJ:",proj
fs = dadi.Spectrum.from_data_dict(dd,
                                  pop_ids=["fl","ca","cu"],
                                  projections=proj,
                                  polarized=True)
print >>sys.stderr, "Model 1X"
print >>sys.stderr, "S:",fs.S()
###############################################################################
def IM_split1(params, ns, pts):
    ## parse params
    N1,N2,N3,T2,T1,m13,m31,m23,m32 = params
    
    ## create a search grid
    xx = dadi.Numerics.default_grid(pts)

    ## create ancestral pop
    phi = dadi.PhiManip.phi_1D(xx)

    ## split ancestral pop into two species
    phi = dadi.PhiManip.phi_1D_to_2D(xx,phi)

    ## allow drift to occur along each of these branches
    phi = dadi.Integration.two_pops(phi, xx, T2, nu1=N1, nu2=N2, m12=0., m21=0.)

    ## split pop1 into pops 1 and 3
    phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)

    ## allow drift and migration to occur along these branches
    phi = dadi.Integration.three_pops(phi, xx, T1,
                                      nu1=N1, nu2=N2, nu3=N3,
                                      m12=0.0, m13=m13,
                                      m21=0.0, m23=m23,
                                      m31=m31, m32=m32)
    ## simulate the fs
    fs = dadi.Spectrum.from_phi(phi,ns,(xx,xx,xx),
                                pop_ids=['fl', 'ca', 'cu'])
    return fs
###############################################################################

## sample sizes
ns = fs.sample_sizes

## points used for exrapolation
pts_l = [12,20,32]

## starting values for params
N1 = N2 = N3 = 1.0
T2 = 2.0
T1 = 2.0
m13 = m31 = m23 = m32 = 0.1
f  = 0.4999

## create starting parameter sets
params_IM    = np.array([N1,N2,N3,T2,T1,m13,m31,m23,m32])

## search limits
upper_IM    = [10.0, 10.0, 10.0, 5.0,  5.0,  25.0, 25.0, 25.0, 25.0]
lower_IM    = [1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-5]

###############################################################################

model = IM_split1
maxiters = 10

Func = dadi.Numerics.make_extrap_log_func(model)

p0 = dadi.Misc.perturb_params(params_IM, fold=2.,
                              upper_bound=upper_IM,
                              lower_bound=lower_IM),

print >>sys.stderr, maxiters, 'iters'
print >>sys.stderr, pts_l, 'extrapolation'
print >>sys.stderr, upper_IM, 'upper'
print >>sys.stderr, lower_IM, 'lower'
print >>sys.stderr, p0, 'start values'


#popt = dadi.Inference.optimize_log(p0, fs, Func, pts_l, 
#                                   lower_IM, upper_IM,
#                                   10, maxiters)

popt = dadi.Inference.optimize_log_fmin(p0, fs, Func, pts_l, 
                                          lower_IM, upper_IM,
                                          10, maxiters)
# popt = dadi.Inference.optimize_log_lbfgsb(p0, fs, Func, pts_l, 
#                                           lower_IM, upper_IM,
#                                           10, maxiters)

mod = Func(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(mod, fs)

print >>sys.stdout, list(popt)+[ll_opt]

