import numpy as np
import matplotlib
from FyeldGenerator import generate_field

"""
Quick caller function to make this cleaner
INPUT
	gridSize: np.array with the size of each dim of the grid
	dx: size of cells, not well tested for != 1.0
	n: number of particles
	x: position of particles
	dim: dimention (2 or 3)
	max_m: maximum mass to be created
OUTPUT
	masses: n len array of each particles mass
"""
def GetMass(gridSize, dx, n, x, dim, max_m, cosm="mine"):

	Probs, x_arr = P_k_dist(gridSize, dx, n, dim, cosm=cosm)
	# print(Probs.shape)

	masses = setMass(n, x, Probs, dx, max_m, dim)
	return masses


"""
Produces the probabilty grid that will be used to get masses. 
Two methods:
cosm="mine" (default): Will use the P(x) model of the scale-invariant power 
	spectrum, seems to work well in 2D but gets streaky in 3D (may be a problem in the IFT)
cosm="other": Uses FyeldGenerator's generate_field to get the grid. See functions near end of file.

"""
def P_k_dist(gridSize, dx, n, dim, cosm="mine"):

    x = np.arange(0, gridSize[0], dx)
    
    #using my attempt at getting the density k^-3 field, it looks decent in 2d, but not 3d
    if cosm=="mine":
        N = gridSize[0]
        
        #0-N/2, -N/2 - (-1)
        N = gridSize[0] + 1 #only going to work with even grid sizes rip
        a = np.arange(0, N/2, dx)
        b = np.arange(-N/2+1, -1, dx)

        k_grid = np.concatenate((a, b)) * 2 * np.pi / N

        k_moreD = []

        #get a grid of k_3d = sqrt(k^2 + k^2 + k^2 ) (or sim for 2D)
        for i in range(0, len(k_grid)):
            kk = []
            for j in range(0, len(k_grid)):
                if dim==3:
                    k3 = []
                    for k in range(0, len(k_grid)):
                        k3.append(np.sqrt(k_grid[i]**2 + k_grid[j]**2))
                    kk.append(k3)
                else:
                    kk.append(np.sqrt(k_grid[i]**2 + k_grid[j]**2))
            k_moreD.append(kk)

        #quick switch to an np array
        k_moreD = np.array(k_moreD)

        #actually do the power law
        pk = 1/np.abs(k_moreD**3)

        #replace any infs by 0.0
        nans = np.argwhere(np.isinf(pk))

        for i in nans:
            if dim==2:
                pk[i[0]][i[1]] = 0.0
            else:
                pk[i[0]][i[1]][i[2]] = 0.0

        pk_sq = np.sqrt(pk)

        #mulitply by some white noise and IFT it back 
        whitenoise = np.fft.fft(np.random.normal(0,1, (gridSize/dx).astype(int) ))
        pk_noise = pk_sq*whitenoise
        kgrid_inv = np.fft.irfftn(pk_noise, s=pk_noise.shape)
        
    #using the FyeldGenerator for better results in 3D
    else:
        kgrid_inv = gen_field(gridSize, dx, dim)
    
    #rescaling it from a density to a probability
    #I think this is probably wrong, but it keeps the structure and I don't know what else to do
    minP = np.min(kgrid_inv)
    maxP = np.max(kgrid_inv)

    q_s = kgrid_inv/(maxP - minP)

    m = np.min(q_s)
    q_s_up = q_s - m #is now in range (0, 1)

    return q_s_up, x

"""
Taking in a grid of probabilties (P), uses this to get the mass of each particle.
"""
def setMass(n, x, P, dt, max_m, dim):
	m = np.zeros(n)

	for i in range(n):
	    #need to figure out which cell each particle is in
	    #then pick out the prob in that cell to use here
	    #don't need to worry about particles outside the cells as I 
	    #just set them

	    cell_n = np.floor(x[i]/(dt)).astype('int64') 
	    # print(cell_n)

	    if dim==2:
	        P_val = P[cell_n[0]][cell_n[1]]
	    else:
	        P_val = P[cell_n[0]][cell_n[1]][cell_n[2]]
	    
	    #what kind of distribution are we then using?
	    #we need mass to be positive, just use a normal distribution for now        
	    mass = np.random.uniform(0.1, P_val,1)*max_m #putting lower limit as I really don't want 0 mass particles
	    m[i] = mass
	    
	return m

"""
Helper that generates power-law power spectrum
taken from https://github.com/cphyc/FyeldGenerator
to see if this gets better results than my own 
homemade one.
"""
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk

# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

def gen_field(gridSize, dx, dim):
    if dim==2:
        shape = (gridSize[0], gridSize[1])
    else:
        shape = (gridSize[0], gridSize[1], gridSize[2])

    x_arr = np.arange(0, gridSize[0], dx)

    field = generate_field(distrib, Pkgen(3), shape)
    return field