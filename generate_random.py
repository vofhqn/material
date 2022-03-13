from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
import numpy as np
calc=LennardJones(rc=500)
def relax(xyz):
    #N=len(xyz)
    N = xyz.shape[0]
    atm=Atoms('Ar'+str(N),positions=xyz)
    atm.calc=calc
    atm.get_potential_energy()
    dyn = BFGS(atm, logfile=None)
    dyn.run(fmax=0.0001, steps=200)
    return(atm.get_positions(),atm.get_potential_energy())

N = 30 #num of particles
D = 10000
maxV = 5.0
randomdata = np.random.uniform(-maxV, maxV, (D, N, 3))
basinpos = []
finalenergy = np.zeros(D)
for i in range(D):
    if i % 100 == 0:
        print("solved", i)
    ret = relax(randomdata[i, :, :])
    basinpos.append(ret[0])
    finalenergy[i] = ret[1]
basinpos = np.array(basinpos)
np.savez("data.npz", initpos = randomdata, basinpos = basinpos, energy = finalenergy)
