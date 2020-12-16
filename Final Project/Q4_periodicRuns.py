import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#import particleClass as system
from particleClass import *
from massInit import GetMass

n=300000 #runs really quite slow with even just 10 000 points
gridSize = [20, 20, 20]
dt_t = 1.0

#single particle at rest
x=np.random.uniform(1, gridSize[0]-1,(n,3))
v=np.random.randn(n,3)*0
max_m = 2.0

m = GetMass(np.array(gridSize), dt_t, n, x, 3, max_m, cosm="mine")

parts=Particles(x,v,m, gridSize, grid_dx=dt_t, BC_per=True)

dt=0.01

fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")

ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_zlabel("z",fontsize=14)

ax.set_xlim3d(0, gridSize[0])
ax.set_ylim3d(0,gridSize[1])
ax.set_zlim3d(0,gridSize[2])

save_E = []

for k in range(2):
    for l in range(1):
        parts.take_step_leapfrog_conv(dt)
   
    scatt = ax.scatter(parts.x[:,0], parts.x[:,1], parts.x[:,2], c=m ,marker=".", s=0.005)
    if k==0:
        fig.colorbar(scatt, ax=ax)
    ax.set_title("frame {}".format(k),fontsize=20)
    plt.savefig('Q4_try1_{}.jpg'.format(k), dpi=1200)
    scatt.remove() #clears prev points

    save_E.append(parts.get_energy())
    print("Step: {}, E total: {}".format(k, save_E[-1][0]))