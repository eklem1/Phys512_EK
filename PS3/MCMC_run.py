import numpy as np
import matplotlib.pyplot as plt
import wmap_camb_example

import corner

#given code
plt.ion()

#[H_0, w_bh2, w_ch2, tau, A_s, slope]
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
pars_old=pars

#[multipole index, measured power spectrum, error, instrument noise part, “cosmic variance” part]
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

plt.clf();
# plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*', alpha=0.1)
plt.plot(wmap[:,0],wmap[:,1],'.') 

cmb=wmap_camb_example.get_spectrum(pars)

plt.plot(cmb) #what really is the x values here then?
plt.xlabel("multipole index")
plt.ylabel("power spectrum")

plt.show()

# curvature matrix from only 5 params, need to add in tau
covA = np.array([[ 2.44781662e+02,  5.63475243e-02, -2.87916355e-01,
         2.61416820e+00,  1.78154791e-08,  1.67922071e+00],
       [ 5.63475243e-02,  1.32998969e-05, -6.54143274e-05,
         6.10101376e-04,  4.18922768e-12,  3.93431234e-04],
       [-2.87916355e-01, -6.54143274e-05,  3.43057631e-04,
        -3.06548839e-03, -2.07813959e-11, -1.96005473e-03],
       [ 2.61416820e+00,  6.10101376e-04, -3.06548839e-03,
         2.88468680e-02,  1.97811728e-10,  1.82789826e-02],
       [ 1.78154791e-08,  4.18922768e-12, -2.07813959e-11,
         1.97811728e-10,  1.36063573e-18,  1.25409751e-10],
       [ 1.67922071e+00,  3.93431234e-04, -1.96005473e-03,
         1.82789826e-02,  1.25409751e-10,  1.17452872e-02]])
# covA = np.eye(6)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def cmbfun(x,pars):
    #call the cmb function with the current parameters
    y = wmap_camb_example.get_spectrum(pars, len(x))
    return y

def our_chisq(data,pars):
    #we need a function that calculates chi^2 for us for the MCMC
    #routine to call
    x=data[0]
    y=data[1]
    noise=data[2]
    
    #set chi to inf if tau is negative to not get these terms
    if pars[3] < 0:
        chisq = np.inf
    else:
        model=cmbfun(x,pars)[2:]

        chisq=np.sum( (y-model)**2/noise**2)
    return chisq


def run_mcmc(pars, data, par_step, chifun, nstep=5000):
    
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    chi_cur=chifun(data,pars)
    L = np.linalg.cholesky(covA);   
    d = np.dot(L, np.random.randn(L.shape[0])) 
    
    for i in range(nstep):
        
#        pars_trial=pars+np.random.randn(npar)*par_step 
        pars_trial=pars+d*0.5 #for using corvar for steps

        chi_trial=chifun(data,pars_trial)
        
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            pars=pars_trial
            chi_cur=chi_trial
        chain[i,:]=pars
        chivec[i]=chi_cur
        
        if i%100==0:
            print("{:.2f} ".format(i/nstep), end = '')
    print("done")
    return chain,chivec


x=np.linspace(1,10,1801)

print(len(x), x[-1])

#get starting points with newton's method?

# par_sigs=np.asarray([0.01,0.01,0.01,0.01])
par_sigs=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])/100

#[6.05254431e-01 2.53491162e-04 3.22311588e-04 5.02164207e-12
# 3.39397857e-03]

#[H_0, w_bh2, w_ch2, tau, A_s, slope]
#could start using results from NM
pars_guess=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])

data=[wmap[:,0], wmap[:,1], wmap[:,2]]

chain,chivec=run_mcmc(pars_guess, data, par_sigs, our_chisq, nstep=4000)

#cut off burn in
pars_sigs_new=np.std(chain,axis=0)
pars_new=np.mean(chain,axis=0)

print(pars_sigs_new, pars_new)

np.savetxt("MCMCrun_chains1.txt", chain)
np.savetxt("MCMCrun_chi1.txt", chivec)