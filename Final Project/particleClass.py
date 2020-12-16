import numpy as np
import matplotlib

class Particles:
    def __init__(self,x,v,m, gridSize, grid_dx=1, BC_per=False):
        #set up our grid - gridSize should be [x, y, z]
        x_g = np.arange(0, gridSize[0], grid_dx)
        y_g = np.arange(0, gridSize[1], grid_dx)
        z_g = np.arange(0, gridSize[2], grid_dx)
        self.grid = np.meshgrid(x_g, y_g, z_g) #the (x,y,z) coordinate for each cell
        self.grid_shape = np.array(gridSize) #shape of grid
        self.dt = grid_dx #length of cell
        self.f_grid = np.meshgrid(self.grid[0][0]*0, self.grid[0][0]*0, self.grid[0][0]*0) #to save the components of the force at each cell
        
        self.BC = BC_per #true for periodic BC, false for not
        self.density = np.meshgrid(self.grid[0][0]*0) #grid of density
        self.potential = np.meshgrid(self.grid[0][0]*0) #grid of potential
        
        self.x=x.copy() #position of each particle
        self.v=v.copy() #velocity of each particle
        try:
            self.m=m.copy() #mass of each particle
        except:
            self.m=m
            
        self.f=np.empty(x.shape) #to save each component of the force on each particle
        self.a=np.empty(x.shape) #to save each component of the acceleration of each particle
        
        self.n=self.x.shape[0] #number of particles
        self.kill_p = np.zeros(self.x.shape[0], dtype=bool) #set to 1 (True) if particle is killed


    ############ methods for calculations for each particle (as seen in class) #############
    #################### you can skip this for the sake of this project ####################
    def r(self):
        return np.sqrt(np.sum(self.x**2, axis=1))
        
    def get_forces(self, soft=0.01, do_pot=False):
        self.f[:]=0
        for i in range(self.n):
            for j in range(self.n):
                if i!=j:
                    dx=-self.x[i,:]+self.x[j,:]
                    drsqr=np.sum(dx**2)
                    
                    if drsqr<soft**2:
                        drsqr = soft**2
                
                    r3=1/(drsqr*np.sqrt(drsqr))
                    
                    self.f[i,:]=self.f[i,:]+dx*self.m[j]*r3
        
    #this is for calculting forces with position values (xx) that are not ever saved
    def get_forces_2(self, xx, soft=0.01, do_pot=False):
        self.f[:]=0
        for i in range(self.n):
            for j in range(self.n):
                if i!=j:
                    dx=-xx[i,:]+xx[j,:]
                    drsqr=np.sum(dx**2)
                    
                    if drsqr<soft**2:
                        drsqr = soft**2
                
                    r3=1/(drsqr*np.sqrt(drsqr))
                    self.f[i,:]=self.f[i,:]+dx*self.m[j]*r3

    def get_acc(self, soft=0.01, do_pot=False):
        self.a[:]=0
        for i in range(self.n):
            for j in range(self.n):
                if i!=j:
                    dx=-self.x[i,:]+self.x[j,:]
                    drsqr=np.sum(dx**2)
                    
                    if drsqr<soft**2:
                        drsqr = soft**2
                
                    r3=1/(drsqr*np.sqrt(drsqr))
                    
                    self.a[i,:]=self.a[i,:]+dx*r3 #going to be the same really as F as m=1 rn


    def take_step_leapfrog(self, dt, soft=0.01):
        self.get_acc()
        v_h = self.v + self.a*dt
        self.x = self.x + v_h*dt
        self.v = v_h #so the V saved will be a half step behind x
        
    #############################################
    
    ############ methods for calculations with convolution (for final project) #############
    

    """
    acc = grad phi

    calculate the potential using the Green's function!
    So you can use the Green's function of the laplacian : -1/(4*pi*r) where r = sqrt(x^2 + y^2 + z^2)
    Since (taking 4*pi*G=1): laplacian(phi) = rho ,

    solve phi = ifft(fft(G_laplacian) * fft(rho))  where G_laplacian is just the Green's function of the laplacian
    """
    
    #gets radius from a point for each cell in our grid
    def r_grid_from_point(self, point):
        t = np.zeros( (len(self.grid[0]), len(self.grid[1]), len(self.grid[2])) ) 
         
        # could also just do this, but I want my point in the center so i don't lose most of it cause that seems 
        # problematic - well this works as long as the box is a cube
        grid_arr = np.array(self.grid)
        t = np.sum((grid_arr + (self.dt)/2.0 -point[0])**2,axis=0)

        return t
    

    #Green's function for the laplacian for a particle in the center of the grid
    def G_laplacian(self, soften=0.01):
        #can we just use a non-real particle at the center to make this easier? Yes i think so
        
        centerPoint = [self.grid_shape[0]/2.0, self.grid_shape[1]/2.0, self.grid_shape[2]/2.0]
        R_arr = self.r_grid_from_point(centerPoint)
        R_arr[R_arr<soften] = soften
        
        green = -1/(4*np.pi*R_arr)
        
        green += np.flip(green,0)
        green += np.flip(green,1)
        green += np.flip(green,2)
        
        return green
    

    #uses a histogram to get the density of each cell
    def get_density(self):
        
        #use a histogram to do way easier
        
        #get our bin edges - need one extra point compared to our grid as we need all the edges
        dt = self.dt
        x_g = np.arange(0, len(self.grid[0])*dt + dt, dt) #these can't deal with particles in -ve space -> BC will deal with this?
        y_g = np.arange(0, len(self.grid[1])*dt + dt, dt)
        z_g = np.arange(0, len(self.grid[2])*dt + dt, dt)
        
        #now fill the histgram, with the masses as the weights
        Hist, edges = np.histogramdd(self.x, bins=[x_g,y_g,y_g], weights=self.m)
        #default behaviour for histogram is to "push" points that land on a bin edge to the right.
            
        #get total mass per grid cube, then just divide by the volume of each cube
        density = Hist/dt**3
        self.density = density
     

    """
	Convolves the test green function for 1 particle at the center of the grid with the density
	to get the grid potential. The gradient is then taken to get the grid force, adding padding here
	if BC are non-periodic.

	Returns phi, f_grid, grad_x, grad_y, grad_z for trouble-shooting
    """
    def get_forces_conv(self, soft=0.01, do_pot=False):
        self.get_density()
        rho = self.density
        
        #get green's function
        green = self.G_laplacian()

        #convolution to get the potential
        phi = convFunction(green, rho, dim=len(self.x[0]))  
        self.potential = phi
        
        #doing our gradient to get the forces, need to take BC in to account here    
        if self.BC:
    
            #just df/dx = (f(x+dt)-f(x-dt))/2dt for now - as this rolls over really it's for periodic
            grad_x = (np.roll(phi,1,axis=0)-np.roll(phi,-1,axis=0))/(2*self.dt)
            grad_y = (np.roll(phi,1,axis=1)-np.roll(phi,-1,axis=1))/(2*self.dt)
            grad_z = (np.roll(phi,1,axis=2)-np.roll(phi,-1,axis=2))/(2*self.dt) 
            
        else: #need to add 0 padding outside and then after grad get correct size back
            #just df/dx = (f(x+dt)-f(x-dt))/2dt for now
#             numpy.pad(phi, [1,1])
            p = 1
            phi_pad = np.pad(phi, [p, p]) #just pad 1 on each side
            
            grad_x = (np.roll(phi_pad,1,axis=0)-np.roll(phi_pad,-1,axis=0))/(2*self.dt)
            grad_y = (np.roll(phi_pad,1,axis=1)-np.roll(phi_pad,-1,axis=1))/(2*self.dt)
            grad_z = (np.roll(phi_pad,1,axis=2)-np.roll(phi_pad,-1,axis=2))/(2*self.dt) 
            
            #re-size
            grad_x = grad_x[p:-p, p:-p, p:-p]
            grad_y = grad_y[p:-p, p:-p, p:-p]
            grad_z = grad_z[p:-p, p:-p, p:-p]
       
        grad_x = -rho*grad_x
        grad_y = -rho*grad_y
        grad_z = -rho*grad_z
    
        #should stack all the forces so this is 4d -> [f_x, f_y, f_z] for each grid cell
        self.f_grid = np.stack((grad_x, grad_y, grad_z), axis=3) 
        
        return phi, self.f_grid, grad_x, grad_y, grad_z


    """
	Get the accelerating for each particle from the force of the cell it is found in 
	by looping through all particles.

    """
    def get_acc_conv(self, soft=0.01, do_pot=False):
        #first calculate the force in each grid cell
        self.get_forces_conv()
        out = 0
        
        for i in range(self.n):
            #need to figure out which cell each particle is in
            #then pick out the force in that cell to use here
            
            cell_n = np.floor(self.x[i,:]/(self.dt)).astype('int64') 
        
            #got to watch out when particles leave our cube and deal with them
            if np.any(cell_n < 0) or np.any(cell_n >= len(self.grid[0])) or self.kill_p[i]:
#                 print("particle out!")
                out += 1
                
                if self.BC: #could I just use a mod for periodic?
                    cell_n = np.floor(self.x[i,:]/(self.dt)).astype('int64') % len(self.grid[0])
                    
                    f = self.f_grid[cell_n[0]][cell_n[1]][cell_n[2]]
                    if self.m[i] > 0:
                        self.a[i,:]=f/self.m[i]
                    else:
                        self.a[i,:] = f*0.0
                    
                else:
                    self.m[i] = 0.0
                    self.kill_p[i] = 1
                    self.a[i,:]= [0.0,0.0,0.0]
                    
            else: #non-edge particles
                f = self.f_grid[cell_n[0]][cell_n[1]][cell_n[2]]
                        
                self.a[i,:]=f/self.m[i]
            
#         print("particles out from acc matching:", out)
            
        return self.a
    
    """ 
	Checks for particles that are not with our grid. If periodic uses a mod to place
	them in approximatly the opposing cell, if non periodic, particles are removed (by marking
	them killed, chaning their mass to 0 and placing them outside our grid).
    """
    def check_BC(self):
        
        out = 0
        
        for i in range(self.n):           
            
            cell_n = np.floor(self.x[i,:]/(self.dt)).astype('int64') 
            #got to watch out when particles leave our cube and deal with them
            
            if np.any(cell_n < 0) or np.any(cell_n >= len(self.grid[0])):
#                 print("particle out!")
                out += 1
                
                if self.BC: #just use a mod for periodic
                #a little rough, but will bring particles in areas too big to the low number cells
                #and vise versa for negative positions
#                     print("moving", self.x[i,:])
                    self.x[i,:] = self.x[i,:] % self.grid_shape[0]
#                     print("to ", self.x[i,:])
                    
                else: #need to actually remove them
                    self.x[i,:] = [-5,-5,-5] #move well outside our grid
                    self.v[i,:] = self.x[i,:]*0 #set to 0
                    self.m[i] = 0.0 #no mass means no acc
                    self.kill_p[i] = 1 #mark as killed
                    
#         print("particles out:", out)


	#our actual leapfrog stepper - pretty straight forward
    def take_step_leapfrog_conv(self, dt, soft=0.01):
        self.get_acc_conv() #get our acc

        v_h = self.v + self.a*dt
        self.x = self.x + v_h*dt
        self.check_BC() #check for run away particles
        
        self.v = v_h #so the V saved will be a half step behind x
        
    #calculate the kinetic, potential and total energy
    def get_energy(self):
        #simple 1/2 mv^2 for each particle for kinetic energy
        e_K = np.sum( 0.5*self.m*(np.sum(self.v**2, axis=1))**2)
        
        #get potential energy using our density, making sure to not double count (I think)
        e_P = -0.5*np.sum(self.density*self.potential)
        
        #add them together for total energy
        e_total = e_K + e_P
        return [e_total, e_K, e_P]
    


#helper functions

#simple convolution of 2 arrays of dim
def convFunction(arr1, arr2, dim=3):
 
    arrFT1 = np.fft.rfftn(arr1)
    arrFT2 = np.fft.rfftn(arr2)
    
    phi = np.fft.irfftn(arrFT1 * arrFT2, s=arr1.shape) # need to have s to get odd length arrays

    for i in range(dim):
            phi = 0.5*(np.roll(phi,1,axis=i)+phi)    
    return phi