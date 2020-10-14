'''
Emma Klemets - 260775167

Code to run MCMC fits
'''

import numpy as np
import matplotlib.pyplot as plt
import wmap_camb_example


#[H_0, w_bh2, w_ch2, tau, A_s, slope]
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])

#[multipole index, measured power spectrum, error, instrument noise part, “cosmic variance” part]
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')


#results from NM
covA = np.array([[ 8.23871531e+00,  1.37020326e-03, -1.50471821e-02,
         2.31846631e-01,  8.19425682e-10,  4.24816638e-02],
       [ 1.37020326e-03,  4.46312117e-07, -1.62360713e-06,
         5.75638828e-05,  2.19723116e-13,  1.10183871e-05],
       [-1.50471821e-02, -1.62360713e-06,  3.43375583e-05,
        -3.78047786e-04, -1.25344410e-12, -6.34653884e-05],
       [ 2.31846631e-01,  5.75638828e-05, -3.78047786e-04,
         2.07900746e-02,  7.86708515e-11,  1.80924500e-03],
       [ 8.19425682e-10,  2.19723116e-13, -1.25344410e-12,
         7.86708515e-11,  2.99046542e-19,  6.80197028e-12],
       [ 4.24816638e-02,  1.10183871e-05, -6.34653884e-05,
         1.80924500e-03,  6.80197028e-12,  3.40806635e-04]])

#scaling step size based off of sigma to increase acceptance rate
startSig = np.array([0.1,1e-3,1e-2,1e-3, 1e-11, 1e-2])


def cmbfun(x,pars):
    #call the cmb function with the current parameters
    y = wmap_camb_example.get_spectrum(pars)[2:len(x)+2]

    return y

def our_chisq(data,pars, tau=False):
    #we need a function that calculates chi^2 for us for the MCMC
    #routine to call
    x=data[0]
    y=data[1]
    noise=data[2]
    
    
    #set chi to inf if tau is negative to not get these terms
    if pars[3] < 0:
        chisq = np.inf
    else:
        model=cmbfun(x,pars)
        
        chisq=np.sum( (y-model)**2/noise**2)

        if tau:#assuming Gaussian likelihood for tau, tau bound - 0.0544 ± 0.0073
            chisq = chisq + ((pars[3]-0.0544)/0.0073)**2

    return chisq

'''
Main function for running MCMC
Input
    pars: initial guess of the 6 parameters
    data: [x, y, err] of data
    par_step: not actually used here
    chifun: chi squared function
    nstep: number of steps
    tau: if true, applies smaller gaussian likelihood to tau

Output
    chain: 2D array of path for each parameter
    chivec: chi squared for each step
    accept_rate: acceptance rate for each step
'''
def run_mcmc(pars, data, par_step, chifun, nstep=5000, tau=False):
    
    #initalize arrays
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    #inital chi^2 value
    chi_cur=chifun(data, pars, tau)
    
    L = np.linalg.cholesky(covA)  
    
    accept_rate = np.zeros(nstep)
    naccept = 0
    i = 0
    
    while i < nstep:
        #so scaling needed as get AR~45% for this covarience matrix
        pars_trial=pars + np.dot(L, np.random.randn(L.shape[0])) #for using corvar for steps
        
        if 0 > pars[3]:
            print("skip")
            #don't count this as a steps
        else:    
#            print(pars_trial-chain[i-1,:])
            chi_trial=chifun(data,pars_trial, tau)
            
            #we now have chi^2 at our current location
            #and chi^2 in our trial location. decide if we take the step
            accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
            
            if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
                pars=pars_trial
                chi_cur=chi_trial
                naccept += 1
                
            #save step
            chain[i,:]=pars
            chivec[i]=chi_cur
            accept_rate[i] = naccept / (i+1)
#            print(accept_rate[i])
            
            if i%100==0:
                print("{:.2f}, AR: {}; ".format(i/nstep, accept_rate[i]), end = '')
            #got to next step
            i = i+1
            
    print("done")
    return chain, chivec, accept_rate


#[H_0, w_bh2, w_ch2, tau, A_s, slope]
pars_guess = [7.94057e+01, 2.4145e-02, 9.9560e-02, 2.7444e-01, 3.0890e-09, 1.0304e+00]

data=[wmap[:,0], wmap[:,1], wmap[:,2]]

chain,chivec, accept_rate=run_mcmc(pars_guess, data, startSig, our_chisq, nstep=10000, tau=False)

#quick glance at results
pars_sigs_new=np.std(chain,axis=0)
pars_new=np.mean(chain,axis=0)
pars_ave=np.mean(accept_rate)

print(pars_ave, pars_new, pars_sigs_new)

#save results
np.savetxt("MCMCrun_chains5.txt", chain)
np.savetxt("MCMCrun_chi5.txt", chivec)
np.savetxt("MCMCrun_AR5.txt", accept_rate)

######fitting for smaller tau prior#######
#mostly resutls from NM, but the tau value is far off, so I increased the guess a bit
pars_guess = [7.94057e+01, 2.4145e-02, 9.9560e-02, 4.7444e-01, 3.0890e-09, 1.0304e+00]

chain, chivec, accept_rate = run_mcmc(pars_guess, data, startSig, our_chisq, nstep=10000, tau=False)

#quick glance at results
pars_sigs_new=np.std(chain,axis=0)
pars_new=np.mean(chain,axis=0)
pars_ave=np.mean(accept_rate)

print(pars_ave, pars_new, pars_sigs_new)

#save results
np.savetxt("MCMCrun_chains_tau2.txt", chain)
np.savetxt("MCMCrun_chi_tau2.txt", chivec)
np.savetxt("MCMCrun_AR_tau2.txt", accept_rate)
