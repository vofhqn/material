from utils import *

N, minsep, maxV = 20, 1.5, 3
cur_energy = 0
search_steps = 2000
minenerges = np.zeros(search_steps + 1)
minenerges[0] = 1e8
totalsteps = np.zeros(search_steps + 1)
for t in range(search_steps):
    init = generate_random_data(N, minsep, maxV)
    loc, success = init[0], init[1]
    if t > 0 and t % 10 == 0:
        print("step", t ,minenerges[t], totalsteps[t])
    #print(loc, success)
    if not success:
        continue
    #successfully generate a initial positioning with minsep

    finalposition, finalenergy, steps = relax(loc)
    minenerges[t + 1] = min(finalenergy, minenerges[t])
    totalsteps[t + 1] = steps + totalsteps[t]

print(minenerges)
print(totalsteps)

np.savez("random_N20.npz", energy=minenerges, steps=totalsteps)