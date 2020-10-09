import numpy as np
import matplotlib.pyplot as plt
import wmap_camb_example


#given code

#[H_0, w_bh2, w_ch2, tau, A_s, slope]
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])


#[multipole index, measured power spectrum, error, instrument noise part, “cosmic variance” part]
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')


#results from NM
covA = np.array([[ 2.30233318e+02,  5.30609392e-02, -2.68331862e-01,
         2.43283786e+00,  1.67722186e-08,  1.57350347e+00],
       [ 5.30609392e-02,  1.25600880e-05, -6.09787460e-05,
         5.68963928e-04,  3.95424446e-12,  3.69553416e-04],
       [-2.68331862e-01, -6.09787460e-05,  3.17098440e-04,
        -2.82614322e-03, -1.93734416e-11, -1.81879931e-03],
       [ 2.43283786e+00,  5.68963928e-04, -2.82614322e-03,
         2.66172931e-02,  1.84716026e-10,  1.69685849e-02],
       [ 1.67722186e-08,  3.95424446e-12, -1.93734416e-11,
         1.84716026e-10,  1.28611142e-18,  1.17828370e-10],
       [ 1.57350347e+00,  3.69553416e-04, -1.81879931e-03,
         1.69685849e-02,  1.17828370e-10,  1.09796327e-02]])

startVal = np.array([8.91853726e+01, 2.66745567e-02, 8.80833909e-02, 3.85185078e-01, 3.80581496e-09 ,1.10901041e+00])

#scaling step size based off of sigma to increase acceptance rate
startSig = 0.5*np.array([2.67988840e+00, 6.36602154e-04, 3.77729627e-03, 1.49724000e-02, 1.10882123e-10, 1.41553442e-02])


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
    
#    tau bound - 0.0544 ± 0.0073
    if (0.0544 - 0.0073) < pars[3] < (0.0544 + 0.0073):
        model=cmbfun(x,pars)[2:]
        chisq=np.sum( (y-model)**2/noise**2)
#        print("do it")
        
#    if 0 <= pars[3]:
#        model=cmbfun(x,pars)[2:]
#        chisq=np.sum( (y-model)**2/noise**2)
    else:
        chisq = np.inf

    return chisq


def run_mcmc(pars, data, par_step, chifun, nstep=5000):
    
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    chi_cur=chifun(data,pars)
    L = np.linalg.cholesky(covA);   
#    d = np.dot(L, np.random.randn(L.shape[0])) 
#    print(d)
    accept_rate = np.zeros(nstep)
    naccept = 0
    
    for i in range(nstep):
        
#        pars_trial=pars+np.random.randn(npar)*par_step 
#        pars_trial=pars + par_step*np.dot(L, np.random.randn(L.shape[0])) #makes h too small?
        pars_trial=pars + 0.5*np.dot(L, np.random.randn(L.shape[0])) #for using corvar for steps
#        print(i, pars_trial)
        
        chi_trial=chifun(data,pars_trial)
        
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            pars=pars_trial
            chi_cur=chi_trial
            naccept += 1
            
        chain[i,:]=pars
        chivec[i]=chi_cur
        accept_rate[i] = naccept / (i+1)
#        print(accept_rate[i])
        
        if i%100==0:
            print("{:.2f} ".format(i/nstep), end = '')
    print("done")
    return chain, chivec, accept_rate


x=np.linspace(1,10,1801)

print(len(x), x[-1])

#get starting points with newton's method?

# par_sigs=np.asarray([0.01,0.01,0.01,0.01])
par_sigs=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])/80

#[6.05254431e-01 2.53491162e-04 3.22311588e-04 5.02164207e-12
# 3.39397857e-03]

#[H_0, w_bh2, w_ch2, tau, A_s, slope]
#could start using results from NM
pars_guess=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])

data=[wmap[:,0], wmap[:,1], wmap[:,2]]

#chain,chivec=run_mcmc(pars_guess, data, par_sigs, our_chisq, nstep=4000)
chain,chivec, accept_rate=run_mcmc(pars_guess, data, startSig, our_chisq, nstep=2000)

#cut off burn in
pars_sigs_new=np.std(chain,axis=0)
pars_new=np.mean(chain,axis=0)
pars_ave=np.mean(accept_rate)

print(pars_ave, pars_new, pars_sigs_new)

#np.savetxt("MCMCrun_chains.txt", chain)
#np.savetxt("MCMCrun_chi.txt", chivec)
#np.savetxt("MCMCrun_AR.txt", accept_rate)

#problem if my starting tau is quite far out of tau range)
#runs much faster? but weird chains, seems to plateau lots
np.savetxt("MCMCrun_chains_tau.txt", chain)
np.savetxt("MCMCrun_chi_tau.txt", chivec)
np.savetxt("MCMCrun_AR_tau.txt", accept_rate)