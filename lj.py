#https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
#https://wiki.fysik.dtu.dk/ase/index.html
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
calc=LennardJones(rc=500)

def relax(xyz):
    N=len(xyz)
    atm=Atoms('Ar'+str(N),positions=xyz)
    atm.calc=calc
    atm.get_potential_energy()
    dyn = BFGS(atm, logfile=None)
    dyn.run(fmax=0.0001)
    return(atm.get_positions(),atm.get_potential_energy())


#example
# xyz=[(0,0,0),(0,0,1),(1,2,1)]
# results = relax(xyz)
# print(results[0], results[1])


