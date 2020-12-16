import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#import particleClass as system
from particleClass import *

n=200000 #runs really quite slow with even just 10 000 points
gridSize = [20, 20, 20]

#single particle at rest
x=np.random.uniform(3.0, gridSize[0]-3.0,(n,3)) #need to place closer to center
v=np.random.randn(n,3)*0 #at rest?

# print(x)

m=np.ones(n) #mass

dt_t = 1.0

parts=Particles(x,v,m, gridSize, grid_dx=dt_t, BC_per=False)

dt=0.002

fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")

ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_zlabel("z",fontsize=14)

ax.set_xlim3d(0, gridSize[0])
ax.set_ylim3d(0,gridSize[1])
ax.set_zlim3d(0,gridSize[2])

save_E_w = []

for k in range(30):
    for l in range(1):
        parts.take_step_leapfrog_conv(dt)
    
    scatt = ax.scatter(parts.x[:,0], parts.x[:,1], parts.x[:,2], color="royalblue",marker=".", s=0.08)#alpha=0.7, )
    
    ax.set_title("frame {}".format(k),fontsize=20)
    plt.savefig('3dfinal_noP_{}.jpg'.format(k), dpi=1200)
    scatt.remove() #clears prev points

    save_E_w.append(parts.get_energy())
    print("Step: {}, E total: {}".format(k, save_E_w[-1][0]))
    
save_E_w = np.array(save_E_w)
steps = np.linspace(0, len(save_E_w[:,0])-1, len(save_E_w[:,0]))

fig = plt.figure()
plt.plot(steps, save_E_w[:,0], label="Total E")
plt.plot(steps, save_E_w[:,1], label="Kinetic E", ls="--") 
plt.plot(steps, save_E_w[:,2], label="Potential E") #hmm this is constant - should it be?

plt.title("Energy over time")
plt.xlabel("time step")
plt.ylabel("Energy")
plt.legend()
plt.savefig('Energy_noP.jpg')
