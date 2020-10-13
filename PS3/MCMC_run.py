import numpy as np
import matplotlib.pyplot as plt
import wmap_camb_example


#given code

#[H_0, w_bh2, w_ch2, tau, A_s, slope]
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])


#[multipole index, measured power spectrum, error, instrument noise part, “cosmic variance” part]
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')


#results from NM -  6 params, p0=OG
covA = np.array([[ 3.03437526e-01,  2.88692599e-05, -8.52825259e-05,
        -1.62274841e-05,  4.58660410e-14,  2.61727721e-04],
       [ 2.88692599e-05,  3.74275141e-08,  7.41685997e-09,
        -2.27803931e-08,  1.32060061e-16,  1.92002848e-07],
       [-8.52825259e-05,  7.41685997e-09,  3.66439436e-07,
        -2.57645586e-07,  1.49431865e-15, -2.94419107e-07],
       [-1.62274841e-05, -2.27803931e-08, -2.57645586e-07,
         2.66991796e-06,  5.36293890e-15, -8.93517788e-07],
       [ 4.58660410e-14,  1.32060061e-16,  1.49431865e-15,
         5.36293890e-15,  9.60675754e-23,  6.78532356e-15],
       [ 2.61727721e-04,  1.92002848e-07, -2.94419107e-07,
        -8.93517788e-07,  6.78532356e-15,  1.43225066e-05]])


#covA = np.array([[ 8.23871531e+00,  1.37020326e-03, -1.50471821e-02,
#         2.31846631e-01,  8.19425682e-10,  4.24816638e-02],
#       [ 1.37020326e-03,  4.46312117e-07, -1.62360713e-06,
#         5.75638828e-05,  2.19723116e-13,  1.10183871e-05],
#       [-1.50471821e-02, -1.62360713e-06,  3.43375583e-05,
#        -3.78047786e-04, -1.25344410e-12, -6.34653884e-05],
#       [ 2.31846631e-01,  5.75638828e-05, -3.78047786e-04,
#         2.07900746e-02,  7.86708515e-11,  1.80924500e-03],
#       [ 8.19425682e-10,  2.19723116e-13, -1.25344410e-12,
#         7.86708515e-11,  2.99046542e-19,  6.80197028e-12],
#       [ 4.24816638e-02,  1.10183871e-05, -6.34653884e-05,
#         1.80924500e-03,  6.80197028e-12,  3.40806635e-04]])

startVal = np.array([8.91853e+01, 2.66745e-02, 8.8083e-02, 3.8518e-01, 3.8058e-09 ,1.1090e+00])

startVal_tau = np.array([8.91853e+01, 2.66745e-02, 8.8083e-02, 0.04, 3.8058e-09 ,1.1090e+00])

#scaling step size based off of sigma to increase acceptance rate
#startSig = 0.5*np.array([2.679e+00, 6.3660e-04, 3.777e-03, 1.497e-02, 1.10e-10, 1.415e-02])

startSig = np.array([0.1,1e-3,1e-2,1e-3, 1e-11, 1e-2])
#param_scalings=np.asarray([0.1,1e-3,1e-2,1e-3, 1e-11, 1e-2])

# covA = np.eye(6)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def cmbfun(x,pars):
    #call the cmb function with the current parameters
    y = wmap_camb_example.get_spectrum(pars, len(x))
    return y

def our_chisq(data,pars, tau=False):
    #we need a function that calculates chi^2 for us for the MCMC
    #routine to call
    x=data[0]
    y=data[1]
    noise=data[2]
    
    #set chi to inf if tau is negative to not get these terms
    
#    tau bound - 0.0544 ± 0.0073
    if tau:
#        add ((tau-0.0544)/0.0073)^2 
        if (0.0544 - 0.0073) < pars[3] < (0.0544 + 0.0073):
            model=cmbfun(x,pars)[2:]
            chisq=np.sum( (y-model)**2/noise**2)
        else:
            chisq = np.inf
    else:
        if 0 <= pars[3]:
            model=cmbfun(x,pars)[2:]
            chisq=np.sum( (y-model)**2/noise**2) #+ ((pars[3]-0.0544)/0.0073)**2
        else:
            chisq = np.inf

    return chisq


def run_mcmc(pars, data, par_step, chifun, nstep=5000, tau=False):
    
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    chi_cur=chifun(data, pars, tau)
    L = np.linalg.cholesky(covA);   
#    d = np.dot(L, np.random.randn(L.shape[0])) 
#    print(d)
    accept_rate = np.zeros(nstep)
    naccept = 0
    i = 0
    
#    for i in range(nstep):
    while i < nstep:
        
#        pars_trial=pars+np.random.randn(npar)*par_step 
#        pars_trial=pars + par_step*np.dot(L, np.random.randn(L.shape[0])) #makes h too small?
        pars_trial=pars + 0.9*np.dot(L, np.random.randn(L.shape[0])) #for using corvar for steps
#        print(i, pars_trial)
        
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
                
            chain[i,:]=pars
            chivec[i]=chi_cur
            accept_rate[i] = naccept / (i+1)
#            print(accept_rate[i])
            
            if i%100==0:
                print("{:.2f}, AR: {:.4f}; ".format(i/nstep, accept_rate[i]), end = '')
            #got to next step
            i = i+1
            
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
#pars_guess=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
pars_guess=np.asarray([67,0.02,0.1,0.05,2e-9,0.96])
pars_guess_1 = [7.94057e+01, 2.4145e-02, 9.9560e-02, 2.7444e-01, 3.0890e-09, 1.0304e+00]

pars_guess_tau = [7.94057e+01, 2.4145e-02, 9.9560e-02, 4.7444e-01, 3.0890e-09, 1.0304e+00]

data=[wmap[:,0], wmap[:,1], wmap[:,2]]

#chain,chivec=run_mcmc(pars_guess, data, par_sigs, our_chisq, nstep=4000)
chain,chivec, accept_rate=run_mcmc(pars_guess_1, data, startSig, our_chisq, nstep=600, tau=False)

#cut off burn in
pars_sigs_new=np.std(chain,axis=0)
pars_new=np.mean(chain,axis=0)
pars_ave=np.mean(accept_rate)
#
print(pars_ave, pars_new, pars_sigs_new)

np.savetxt("MCMCrun_chains1.txt", chain)
np.savetxt("MCMCrun_chi1.txt", chivec)
np.savetxt("MCMCrun_AR1.txt", accept_rate)
'''

chain, chivec, accept_rate = run_mcmc(pars_guess, data, startSig, our_chisq, nstep=5000, tau=False)

pars_sigs_new=np.std(chain,axis=0)
pars_new=np.mean(chain,axis=0)
pars_ave=np.mean(accept_rate)

print(pars_ave, pars_new, pars_sigs_new)

#problem if my starting tau is quite far out of tau range)
#runs much faster? but weird chains, seems to plateau lots
np.savetxt("MCMCrun_chains_tau.txt", chain)
np.savetxt("MCMCrun_chi_tau.txt", chivec)
np.savetxt("MCMCrun_AR_tau.txt", accept_rate)
'''