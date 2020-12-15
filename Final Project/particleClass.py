import numpy as np
import matplotlib

class Particles:
    def __init__(self,x,v,m, gridSize, grid_dx=1, BC_per=False):
        #set up our grid - gridSize should be [x, y, z]
        x_g = np.arange(0, gridSize[0], grid_dx)
        y_g = np.arange(0, gridSize[1], grid_dx)
        z_g = np.arange(0, gridSize[2], grid_dx)
        self.grid = np.meshgrid(x_g, y_g, z_g)
        self.grid_shape = np.array(gridSize)
        self.dt = grid_dx
        self.f_grid = np.meshgrid(self.grid[0][0]*0, self.grid[0][0]*0, self.grid[0][0]*0)
        
        self.BC = BC_per #true for periodic BC, false for not
        self.density = np.meshgrid(self.grid[0][0]*0)
        self.potential = np.meshgrid(self.grid[0][0]*0)
        
        self.x=x.copy()
        self.v=v.copy()
        try:
            self.m=m.copy()
        except:
            self.m=m
            
        self.f=np.empty(x.shape)
        self.a=np.empty(x.shape)
        
        self.n=self.x.shape[0]
        self.kill_p = np.zeros(self.x.shape[0], dtype=bool) #set to 1 (True) if particle is killed
         
        #maybe save energy?
        
    ############ methods for calculations for each particle (as seen in class) #############
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
    
    ############ methods for calculations with convolution (for project) #############
    

    """
    acc = grad phi

    calculate the potential using the Green's function!
    So you can use the Green's function of the laplacian : -1/(4*pi*r) where r = sqrt(x^2 + y^2 + z^2)
    Since (taking 4*pi*G=1): laplacian(phi) = rho ,

    solve phi = ifft(fft(G_laplacian) * fft(rho))  where G_laplacian is just the Green's function of the laplacian
    """
    
    def r_grid_from_point(self, point):
        t = np.zeros( (len(self.grid[0]), len(self.grid[1]), len(self.grid[2])) ) 
         
        # could also just do this, but I want my point in the center so i don't lose most of it cause that seems 
        # problematic - well this works as long as the box is a cube
        grid_arr = np.array(self.grid)
        t = np.sum((grid_arr + (self.dt)/2.0 -point[0])**2,axis=0)

        return t
    
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
     
    def get_forces_conv(self, soft=0.01, do_pot=False):
        self.get_density() #I think this looks decent
        rho = self.density
        
        green = self.G_laplacian()
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
            
            grad_x = grad_x[p:-p, p:-p, p:-p]
            grad_y = grad_y[p:-p, p:-p, p:-p]
            grad_z = grad_z[p:-p, p:-p, p:-p]
            #re-size somehow
       
        grad_x = -rho*grad_x
        grad_y = -rho*grad_y
        grad_z = -rho*grad_z
    
        #should stack all the forces so this is 4d -> [f_x, f_y, f_z] for each grid cell
        self.f_grid = np.stack((grad_x, grad_y, grad_z), axis=3) 
        
        return phi, self.f_grid, grad_x, grad_y, grad_z


    def get_acc_conv(self, soft=0.01, do_pot=False):
        #first calculate the force in each grid cell
        self.get_forces_conv()
        out = 0
        
        for i in range(self.n):
            #need to figure out which cell each particle is in
            #then pick out the force in that cell to use here
            
#             print("position? ", self.x[i,:])
            
            cell_n = np.floor(self.x[i,:]/(self.dt)).astype('int64') 
        
            #got to watch out when particles leave our cube and deal with them
            if np.any(cell_n < 0) or np.any(cell_n >= len(self.grid[0])) or self.kill_p[i]:
#                 print("particle out!")
                out += 1
                
                if self.BC: #could I just use a mod for periodic?
                    cell_n = np.floor(self.x[i,:]/(self.dt)).astype('int64') % len(self.grid[0])
                    
                    f = self.f_grid[cell_n[0]][cell_n[1]][cell_n[2]]
                    self.a[i,:]=f/self.m[i]
                    
                else:
#                     cell_n = [0,0,0]
                    self.m[i] = 0.0
                    self.kill_p[i] = 1
                    self.a[i,:]= [0.0,0.0,0.0]
                    
            else: #non-edge particles
                f = self.f_grid[cell_n[0]][cell_n[1]][cell_n[2]]
            
#             print("f, cell#:", f, cell_n)
            
                self.a[i,:]=f/self.m[i]
            
#         print("particles out from acc matching:", out)
            
        return self.a
    
    def check_BC(self):
        
        out = 0
        
        for i in range(self.n):           
#             print("position? ", self.x[i,:])
            
            cell_n = np.floor(self.x[i,:]/(self.dt)).astype('int64') 
            #got to watch out when particles leave our cube and deal with them
            
            if np.any(cell_n < 0) or np.any(cell_n >= len(self.grid[0])): #could I just use a mod for periodic
#                 print("particle out!")
                out += 1
                
                if self.BC: #just use a mod for periodic
                #a little rough, but will bring particles in areas too big to the low number cells
                #and vise versa for negative positions
#                     print("moving", self.x[i,:])
                    self.x[i,:] = self.x[i,:] % self.grid_shape[0]
#                     print("to ", self.x[i,:])
                    
                else: #need to actually remove them somehow?
                    self.x[i,:] = [-5,-5,-5] #[0,0,0] #can I somehow remove them without messing up something else?
                    self.v[i,:] = self.x[i,:]
                    self.m[i] = 0.0
                    self.kill_p[i] = 1
                    
#         print("particles out:", out)


    def take_step_leapfrog_conv(self, dt, soft=0.01):
        self.get_acc_conv()
        v_h = self.v + self.a*dt
        self.x = self.x + v_h*dt
        self.check_BC()
        
        self.v = v_h #so the V saved will be a half step behind x
        
    def get_energy(self):
        #simple 1/2 mv^2 for each particle for kinetic energy
        e_K = np.sum( 0.5*self.m*(np.sum(self.v**2, axis=1))**2)
        
        #get potential energy using our density, making sure to not double count (I think)
        e_P = -0.5*np.sum(self.density*self.potential)
        
        #add them together for total energy
        e_total = e_K + e_P
        return [e_total, e_K, e_P]
    
#helper fucntions
def convFunction(arr1, arr2, dim=3):
 
    arrFT1 = np.fft.rfftn(arr1)
    arrFT2 = np.fft.rfftn(arr2)
    
    phi = np.fft.irfftn(arrFT1 * arrFT2, s=arr1.shape) # need to have s to get odd length arrays

    for i in range(dim):
            phi = 0.5*(np.roll(phi,1,axis=i)+phi)    
    return phi