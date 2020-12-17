import numpy as np
import matplotlib

#for pop out plots - windows
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#import particleClass as system
from particleClass import *

n=200000 #runs really quite slow with even just 10 000 points
gridSize = [25, 25, 25]

#single particle at rest
x=np.random.uniform(0, gridSize[0],(n,3))
v=np.random.randn(n,3) #at rest?

m=np.ones(n) #mass

dt_t = 1.0

parts=Particles(x,v,m, gridSize, grid_dx=dt_t, BC_per=True)

dt=0.0015

fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
# ax = Axes3D(fig)
ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_zlabel("z",fontsize=14)

ax.set_xlim3d(0, gridSize[0])
ax.set_ylim3d(0,gridSize[1])
ax.set_zlim3d(0,gridSize[2])

save_E = []

for k in range(50):
    for l in range(1):
        parts.take_step_leapfrog_conv(dt)
   
    scatt = ax.scatter(parts.x[:,0], parts.x[:,1], parts.x[:,2], color="royalblue",marker=".", s=0.003)#alpha=0.7, )
    
    ax.set_title("frame {}".format(k),fontsize=20)
    plt.savefig('Q3_P2_{}.jpg'.format(k), dpi=1200)
    scatt.remove() #clears prev points

    save_E.append(parts.get_energy())
    print("Step: {}, E total: {}".format(k, save_E[-1][0]))
    
save_E = np.array(save_E)
steps = np.linspace(0, len(save_E[:,0])-1, len(save_E[:,0]))

fig = plt.figure()
plt.plot(steps, save_E[:,0], label="Total E")
plt.plot(steps, save_E[:,1], label="Kinetic E", ls="--") 
plt.plot(steps, save_E[:,2], label="Potential E") #hmm this is constant or very low

plt.title("Energy over time")
plt.xlabel("time step")
plt.ylabel("Energy")
plt.legend()
plt.savefig('Energy_P2.jpg')

fig = plt.figure()
# plt.plot(steps, save_E[:,0], label="Total E")
# plt.plot(steps, save_E[:,1], label="Kinetic E", ls="--") 
plt.plot(steps, save_E[:,2], label="Potential E") #hmm this is constant or very low

plt.title("Energy over time")
plt.xlabel("time step")
plt.ylabel("Energy")
plt.legend()
plt.savefig('Energy_P_justPo2.jpg')