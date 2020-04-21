import numpy as np
import data
from scipy import linalg, optimize
import matplotlib.pyplot as plt

def get_offset(pdf,threshold):
    cdf = np.cumsum(pdf)/np.sum(pdf)
    good = (1-cdf)<threshold
    offset = np.flatnonzero(good)[0]
    return offset

def generate_pdf(ifr=0.006):
    nD = 180
    mu1 = 3.#5.1
    vr1 = 0.86
    sigma1 = vr1**2*mu1
    mu2 = 14.#18.8
    vr2 = 0.45
    sigma2 = vr2**2*mu2
    alpha1 = (mu1/sigma1)**2
    alpha2 = (mu2/sigma2)**2
    beta1 = sigma1**2/mu1
    beta2 = sigma2**2/mu2
    N =100000000
    samples = np.random.gamma(alpha1,beta1,N)+np.random.gamma(alpha2,beta2,N)
    counts, bins, patches = plt.hist(samples,bins=np.arange(nD),align='mid',log=True)
    pdf = counts/sum(counts)
    np.savetxt('time_to_death.txt',pdf)

def infer_infections(deaths, pdf, ifr, nTrials=10, neglect=0):
    #data_dates, cum_cases, cum_deaths = data.load_time_series(region)
    #deaths = np.insert(np.diff(cum_deaths),0,cum_deaths[0])
    if neglect:
        deaths = deaths[:-neglect]

    #ifr = 0.007
    nOffDays = 0  # Initial days with zero chance of death
    nDays = len(deaths)   

    #counts = pdf[:nDays]
    #pdf = counts/sum(counts)*ifr
    pdf = pdf[:nDays]*ifr

    infections = np.zeros((nDays-nOffDays));  # Infection cases time series (to be estimated)
    
    I = np.eye(nDays-nOffDays)
    z = np.zeros(nDays-nOffDays)
    dd  = deaths[nOffDays:];        # Trim the daily death vector to match A/A0
    b = np.hstack((dd,z))

    for iTry in range(nTrials):
        row = np.zeros(nDays); row[0] = pdf[0]
        M0  = linalg.toeplitz(pdf, row);   # Convolution matrix without perturbation
        if nOffDays>0:
            A0  = M0[nOffDays:, :-nOffDays];   # Trim convolution matrix if needed
        else:
            A0  = M0[nOffDays:, :];   # Trim convolution matrix if needed
        maxpertfac = 0.1
        pdf_pert = pdf*(1+np.random.uniform(-1,1,pdf.shape)*maxpertfac); # Perturb each day's fatality with uniformly distributed noise
        #pdf_pert = 2*pdf*np.random.uniform(0,1,pdf.shape)
        row = np.zeros(nDays); row[0] = pdf_pert[0]
        M   = linalg.toeplitz(pdf_pert, row); # Perturbed convolution matrix
        if nOffDays>0:
            A   = M[nOffDays:, :-nOffDays]; # Trim convolution matrix if needed
        else:
            A   = M[nOffDays:, :]; # Trim convolution matrix if needed
        regs = np.linspace(1e-4,1e-2) # Regularization parameter search range
        errs = np.zeros_like(regs)     # Measure of fit for each reg param value
        for i, reg in enumerate(regs):
            L = np.vstack((A,reg*I)); 
            inIreg, _ = optimize.nnls(L,b)
            errs[i] = 10*np.abs( ifr*np.sum(inIreg)-np.sum(dd)) + np.sum(np.abs(np.abs(A@inIreg-dd)))
        reg = regs[np.argmin(errs)]
        L = np.vstack((A,reg*I)); 
        newinf, _ = optimize.nnls(L,b)
        infections += newinf/nTrials
    
    return infections, M0
