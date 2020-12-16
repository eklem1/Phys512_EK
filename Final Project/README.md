# Phys 512 Final Project

**particleClass.py**: Main particle class, initializes and progresses NBody simulation. Has two different sections, basic structure we saw in class with forces calculated for each particle, this is run using take_step_leapfrog(). For this project however, the convolution method is used, calculating the potential at each grid point and then using this to get the force and therefor accelerating for each particle.

### Q1 & Q2
**Q1_2_testing.ipynb**: Code to run #1 and #2. Results:
   1. Q1_output_1particle.gif 
   2. Q2_output_2particle.gif 

### Q3
I think my energy is wrong, as for both the potential energy seems almost constant, and as the kinetic energy is much larger the total energy just follows that. However looking at plots of the potential it does looks fairly correct, so prehaps I'm just calculating it incorrectly. I also did not make my grids very big as Python would not let me allocate the memory needed for them.

**Q3_nonperiodic.py**: Code to run #3, with non-periodic boundary conditions. Particles starting away from edges so they do not escape too fast. Results:
 - smt
 - Energy_noP.jpg : energy plot
The energy is definitely not constant.

**Q3_periodic.py**: Code to run #3, with periodic boundary conditions. Results:
 - smt
 - Energy_P.jpg : energy plot


### Q4
**Q4_work.ipynb**: Code and plots used to develope the scale-invariant power spectrum.

**massInit.py**: Functions made in above notebook cleaned up to be used easier. Initializes masses for n particles in 2D or 3D, with a given maximum mass.

**Q4_periodicRuns.py**: Code to run #4, initializes masses with massInit with periodic boundary conditions. Results:
 - Q4_output_other.gif : PS from FyeldGenerator
 - Q4_output_mine.gif : homemade PS 


 I do get some nice clumps of matter here, similar to how our actual universe looks. However they do dissolve into less defined shapes if run longer.


## plots
Contains all important resulting plots and gifs.
