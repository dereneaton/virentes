#!/usr/bin/env

import dadi
import numpy as np
import os

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
def IM_split2(params, ns, pts):
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
    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)

    ## allow drift and migration to occur along these branches
    phi = dadi.Integration.three_pops(phi, xx, T1,
                                      nu1=N1, nu2=N2, nu3=N3,
                                      m12=0.0, m13=m13,
                                      m21=0.0, m23=m23,
                                      m31=m31, m32=m32)
    ## simulate the fs
    fs = dadi.Spectrum.from_phi(phi,ns,(xx,xx,xx),
                                pop_ids=['fl', 'ca', 'cu'])
    #print fs.S(), 's in spec'
    return fs
###############################################################################
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
def simSFS_model3(theta, N1,N2,N3,T1,T12,f):
    R1,R2,R3 = np.random.random_integers(0,9999999,3)
    command = """ ms 22 7794 \
                     -t %(theta)f \
                     -I 3 10 6 6 \
                     -n 1 %(N1)f \
                     -n 2 %(N2)f \
                     -n 3 %(N3)f \
                     -en %(T1)f 2 %(N2)f \
                     -en %(T1)f 1 %(N1)f \
                     -es %(T1)f 3 %(f)f \
                     -ej %(T1)f 4 2 \
                     -ej %(T1)f 3 1 \
                     -ej %(T12)f 2 1 \
                     -en %(T12)f 1 1 \
                     -seeds %(R1)f %(R2)f %(R3)f """
    sub_dict = {'theta':theta, 'N1':N1, 'N2':N2, 'N3':N3,
                'T1':T1, 'T12':T12, 'f':f,
                'R1':R1, 'R2':R2, 'R3':R3}
    mscommand = command % sub_dict
    fs = dadi.Spectrum.from_ms_file(os.popen(mscommand), average=False)
    return fs
###############################################################################

Nloci = 7794

#############  ML estimates for model 3 ######################################

theta = 282.456200226/Nloci
N1 =  2.2562645699939852
N2 =  0.71705301735090954
N3 =  0.14307134998023263

T2 =  0.57976328038133673
T1 =  0.025790751880146609

f =   0.38345451331222341

p0_3 = [N1, N2, N3, T2, T1, f]

## Values to input to ms for simulating under model 2
msT1  = T1/2
msT12 = (T1+T2)/2

##############################################################################

## simulate data
msfs = simSFS_model3(theta, N1, N2, N3, msT1, msT12, f)

## save sfs to file
i = 1
bootname = "pboots/m3."+str(i)+".boot"
while os.path.exists(bootname):
    i += 1
    bootname = "pboots/m3."+str(i)+".boot"
msfs.to_file(bootname)
print 'sim data set '+str(i)

############  ML estimates model 2  ###########################################
theta = 249.587274958/Nloci    ## 0.0233772573097   ## per locus
N1 = 2.412336866389539
N2 = 0.66060145092919675
N3 = 0.23061219403870617

T2 = 0.71073109877391338
T1 = 0.084218967415615298

m13 = 0.138041546493 
m31 = 6.03645313087
m23 = 4.09467065467
m32 = 0.0031179372964

## starting values
p0_2 = [N1, N2, N3, T2, T1, m13, m31, m23, m32]

##############  ML estimates for model 1 #####################################
theta = 249.587274958/Nloci
N1 =  2.5013156716397473
N2 =  0.68871161421841698
N3 =  0.076807020757753017

T2 =  0.70851628972828207
T1 =  0.14901468477012897

m13 = 0.14349173980106683
m31 = 12.58678666702065
m23 = 1.2625236526206685
m32 = 21.036230092460865

## starting values
p0_1 = [N1, N2, N3, T2, T1, m13, m31, m23, m32]

#############################################################################

## sample sizes
ns = msfs.sample_sizes

## points used for exrapolation
pts_l = [12,20,32]

## search limits models 1 & 2
upper_IM    = [10.0, 10.0, 10.0, 5.0,  5.0,  20.0, 20.0, 20.0, 40.0]
lower_IM    = [1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-5]
## search limits for model 3
upper_admix = [10.0, 10.0, 10.0, 5.0,  5.0,  0.9999]
lower_admix = [1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 0.0001]

########## optimize under model 2  ##########################################
Func = dadi.Numerics.make_extrap_log_func(IM_split2)
popt = dadi.Inference.optimize_log(p0_2, msfs, Func, pts_l, 
                                   lower_IM, upper_IM,
                                   len(p0_2), 5)
mod = Func(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(mod, msfs)

outfile = open("dadi_m3_f2_boots.txt","a")
print >>outfile, "\t".join(["boot"+str(i)]+map(str,list(popt)+[ll_opt]))
outfile.close()

########## optimize under model 1  ##########################################
Func = dadi.Numerics.make_extrap_log_func(IM_split1)
popt = dadi.Inference.optimize_log(p0_1, msfs, Func, pts_l, 
                                   lower_IM, upper_IM,
                                   len(p0_1), 5)
mod = Func(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(mod, msfs)

outfile = open("dadi_m3_f1_boots.txt","a")
print >>outfile, "\t".join(["boot"+str(i)]+map(str,list(popt)+[ll_opt]))
outfile.close()

########## optimize under model 3  ##########################################
Func = dadi.Numerics.make_extrap_log_func(admix)
popt = dadi.Inference.optimize_log(p0_3, msfs, Func, pts_l, 
                                   lower_admix, upper_admix,
                                   len(p0_3), 5)
mod = Func(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(mod, msfs)
outfile = open("dadi_m3_f3_boots.txt","a")
print >>outfile, "\t".join(["boot"+str(i)]+map(str,list(popt)+[ll_opt]))
outfile.close()
############################################################################

## empirical delta values for model comparison
# deltaK_21 = -2*(ll_opt_2 - ll_opt_1)
# deltaK_23 = -2*(ll_opt_2 - ll_opt_3)

