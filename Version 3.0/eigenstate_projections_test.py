import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

from procedure import Procedure
import measurables

proc = Procedure()
proc.setParameters(L = 3 , J = 1, hx = 0.2)


times = np.arange(0,proc.T+proc.dt, proc.dt)

T = proc.T
N_cycles = 2

data = proc.runProcedure(N_cycles=N_cycles, measure=Procedure.measure_eigenstate_projections,
                        setup_state_for_next_cycle=Procedure.pass_full_density_matrix,
                        coupling_decrease="use_default", using_density_matrices=True)

print(data[0,0,:])

energies = np.ones((len(data[:,0,0])//2, len(data[0,:,0]) * len(data[0,0,:])))
products = np.ones((len(data[:,0,0])//2, len(data[0,:,0]) * len(data[0,0,:])))
for i in range(len(data[0,:,0])):
    for j in range(len(data[0,0,:])):
        measurements = data[:,i,j]
        for k in range(0, len(measurements), 2):
            energies[k//2, i * len(data[0,0,:]) + j] = measurements[k]
            products[k//2, i * len(data[0,0,:]) + j] = measurements[k+1]

for i in range(len(energies[:,0])):
    plt.plot(energies[i,:])
    #plt.plot(products[i,:])

plt.show()

for i in range(len(energies[:,0])):
    #plt.plot(energies[i,:])
    plt.plot(products[i,:])

print(np.sum(products, axis = 0))


plt.show()