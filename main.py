import torch 
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

def search(nns, natoms, cur_pos, bonus, maxV, candidate = 100):
    if cur_pos.shape[0] == natoms:
        return 
    
    nxt_cand = np.random.uniform(-maxV, maxV, (candidate, 3))
    data = np.zeros((candidate, cur_pos.shape[0] + 1, 3))
    for _ in range(nxt_cand):
        data[_, :-1, :] = cur_pos
        data[_, -1, :] = nxt_cand[_]
    values = nns[cur_pos.shape[0] + 1](torch.tensor(data).to("cuda"))
    b = bonus(data)
    

        
