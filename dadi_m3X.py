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
print >>sys.stderr, "Model 3X"
print >>sys.stderr, "S:",fs.S()

###############################################################################
## scenario hybrid speciation forms cuba.

def admix(params, ns, pts):
    ## parse params
    N1,N2,N3,T2,T1,f = params

    ## create a search grid
    xx = dadi.Numerics.default_grid(pts)

    ## make ancestral pop that splits into two
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx,phi)

    ## allow drift to occur along each of these branches
    phi = dadi.Integration.two_pops(phi, xx, T2, nu1=N1, nu2=N2, m12=0., m21=0.)

    ## create pop 3 from a mixture of 1 and 2
    phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)

    ## allow drift and migration to occur along these branches
    ## cuba population shrinks in size after divergence
    phi = dadi.Integration.three_pops(phi, xx, T1,
                                      nu1=N1, nu2=N2, nu3=N3,
                                      m12=0.0, m13=0.0,
                                      m21=0.0, m23=0.0,
                                      m31=0.0, m32=0.0)
    ## simulate the fs
    fs = dadi.Spectrum.from_phi(phi,ns,(xx,xx,xx),
                                pop_ids=['fl', 'ca', 'cu'])
    return fs

###############################################################################

## sample sizes
ns = fs.sample_sizes

## points used for exrapolation
#pts_l = [12,20,32]
pts_l = [20,30,40]

## starting values for params
N1 = N2 = N3 = 1.0
T2 = 2.0
T1 = 2.0
m13 = m31 = m23 = m32 = 0.1
f  = 0.4999

## create starting parameter sets
##params_IM    = np.array([N1,N2,N3,T2,T1,m13,m31,m23,m32])
params_admix   = np.array([N1,N2,N3,T2,T1,f])

## search limits
upper_admix   = [10.0, 10.0, 10.0,  5.0,  5.0, 0.99999]
lower_admix   = [1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 0.00001]

###############################################################################

model = admix
maxiters = 20

Func = dadi.Numerics.make_extrap_log_func(model)

p0 = dadi.Misc.perturb_params(params_admix, fold=2.,
                              upper_bound=upper_admix,
                              lower_bound=lower_admix),

p0 = [2.2652041255656679, 0.71930406362120092, 0.14027309330498738, 0.58176402235517111, 0.025277451652610413, 0.38415615000133363]
## , -555.34275419817561]

print >>sys.stderr, maxiters, 'iters'
print >>sys.stderr, pts_l, 'extrapolation'
print >>sys.stderr, upper_admix, 'upper'
print >>sys.stderr, lower_admix, 'lower'
print >>sys.stderr, p0, 'start values'

popt = dadi.Inference.optimize_log(p0, fs, Func, pts_l, 
                                   lower_admix, upper_admix,
                                   10, maxiters)

#popt = dadi.Inference.optimize_log_lbfgsb(p0, fs, Func, pts_l, 
#                                          lower_admix, upper_admix,
#                                          10, maxiters)

mod = Func(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(mod, fs)

print >>sys.stdout, list(popt)+[ll_opt]

