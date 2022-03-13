import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
calc=LennardJones(rc=500)
def relax(xyz):
    #N=len(xyz)
    N = xyz.shape[0]
    atm=Atoms('Ar'+str(N),positions=xyz)
    atm.calc=calc
    atm.get_potential_energy()
    dyn = BFGS(atm, logfile=None)
    dyn.run(fmax=0.0001, steps=200)
    return(atm.get_positions(),atm.get_potential_energy(), dyn.nsteps)


def generate_random_data(N, minsep, maxV, cur = 0, curloc = None, maxTry = 100):
    if cur == N:
        return curloc, True
    count = 0
    while count < maxTry:
        count += 1
        loc = np.random.uniform(-maxV, maxV, (1, 3))
        if cur > 0:
            diff = curloc - loc 
            distance = np.linalg.norm(diff, axis = 1)
            if np.min(distance) < minsep:
                continue
        nxtloc = loc
        if cur > 0:
            nxtloc = np.append(curloc, loc, axis = 0)
        res, success = generate_random_data(N, minsep, maxV, cur + 1, nxtloc)
        if success:
            return res, True

    return None, False

# data  = generate_random_data(10, 4, 2)
# data, success = data[0], data[1]
# print(data)
# for i in range(10):
#     diff = data - data[i, :]      
#     print(np.linalg.norm(diff, axis = 1))
            

