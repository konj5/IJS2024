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
def smooth_step_coupling_decrease(t):
    assert T == 50

    def g(t): return 1 / (1 + np.exp(t))
    ta = 5.7
    t0 = 0
    t1 = T
    a = 0.6
    def f(t): return g(a * ((t-ta)-t1)) + g(-a * ((t+ta)-t0)) - 1
    return f(t)    

T = proc.T
N_cycles = 25

data = proc.runProcedure(N_cycles=N_cycles, measure=Procedure.measure_eigenstate_projections,
                        setup_state_for_next_cycle=Procedure.pass_full_density_matrix,
                        coupling_decrease=smooth_step_coupling_decrease, using_density_matrices=True)

data2 = proc.runProcedure(N_cycles=N_cycles, measure=Procedure.measure_chain_eigenstate_projections,
                        setup_state_for_next_cycle=Procedure.pass_full_density_matrix,
                        coupling_decrease=smooth_step_coupling_decrease, using_density_matrices=True)

    
energies = data[0::2, :, :]
products = data[1::2, :, :]

N_cycles = len(energies[0,:,0])
N_per_cycle = len(energies[0,0,:])
N_states = len(energies[:,0,0])

#prepare for animation 

energy_n = np.zeros((N_cycles, N_states, N_per_cycle), dtype=float)
dotproducts_n = np.zeros((N_cycles, N_states, N_per_cycle), dtype=float)


for i in range(N_cycles):
    for j in range(N_states):
        energy_n[i,j,:] = energies[j,i, :]
        dotproducts_n[i,j,:] = products[j,i, :]

energy = energy_n
dotproducts = dotproducts_n


###

energies2 = data2[0::2, :, :]
products2 = data2[1::2, :, :]

N_cycles2 = len(energies2[0,:,0])
N_per_cycle2 = len(energies2[0,0,:])
N_states2 = len(energies2[:,0,0])

#prepare for animation 

energy_n2 = np.zeros((N_cycles2, N_states2, N_per_cycle2), dtype=float)
dotproducts_n2 = np.zeros((N_cycles2, N_states2, N_per_cycle2), dtype=float)


for i in range(N_cycles2):
    for j in range(N_states2):
        energy_n2[i,j,:] = energies2[j,i, :]
        dotproducts_n2[i,j,:] = products2[j,i, :]

energy2 = energy_n2
dotproducts2 = dotproducts_n2


plt.plot(products[20,0,:], label = "0")
plt.plot(products[20,1,:], label = "1")
plt.plot(products[20,2,:], label = "2")
plt.plot(products[20,3,:], label = "3")
plt.plot(products[20,4,:], label = "4")
plt.plot(products[20,5,:], label = "5")

plt.legend()
plt.show()

"""

from matplotlib import animation

fig, axs = plt.subplots(1,2)
fig.set_figheight(10)
fig.set_figwidth(20)

ax1 , ax2 = axs

ts = times
basegraphs1 = []
slidinggraphs1 = []
for i in range(N_states):
    xdata = ts[0:0]
    ydata = np.array(energy[0, i, 0:0])
    width = np.array(dotproducts[0, i, 0:0])
    slidinggraphs1.append(ax1.fill_between(xdata, ydata-width/2, ydata+width/2))

ax1.set_title("Cycle 1 (Whole system)")
ax1.set_ylabel("Energy")
ax1.set_xlabel("Time during single cycle")

ts = times
basegraphs2 = []
slidinggraphs2 = []
for i in range(N_states2):
    xdata = ts[0:0]
    ydata = np.array(energy2[0, i, 0:0])
    width = np.array(dotproducts2[0, i, 0:0])
    slidinggraphs1.append(ax2.fill_between(xdata, ydata-width/2, ydata+width/2))

ax2.set_title("Cycle 1 (TFIM chain)")
ax2.set_ylabel("Energy")
ax2.set_xlabel("Time during single cycle")

dt = 10
dts = []
t0 = 0

def animate(n):

    global dt, t0, dts
    dt = time.time()-t0
    dts.append(dt)
    if len(dts) > 10:
        dts = dts[-10: -1]
    t0 = time.time()

    global slidinggraphs1, basegraphs1

    ###

    I = n % N_per_cycle
    cycle = n // N_per_cycle

    ax1.clear()
    ax1.set_title(f"Cycle {cycle + 1}")

    basegraphs1 = []
    slidinggraphs1 = []
    for i in range(N_states):

        basegraphs1.append(ax1.plot(ts,energy[cycle,i,:], linestyle = "--", color = "gray"))


    for i in range(N_states):
        xdata = ts[0:I]
        ydata = np.array(energy[cycle, i, 0:I])
        width = np.array(dotproducts[cycle, i, 0:I])*(np.max(energy)-np.min(energy)) * 0.06
        slidinggraphs1.append(ax1.fill_between(xdata, ydata-width/2, ydata+width/2))

    ###

    I = n % N_per_cycle2
    cycle = n // N_per_cycle2

    ax2.clear()
    ax2.set_title(f"Cycle {cycle + 1}")

    basegraphs2 = []
    slidinggraphs2 = []
    for i in range(N_states2):

        basegraphs2.append(ax2.plot(ts,energy2[cycle,i,:], linestyle = "--", color = "gray"))


    for i in range(N_states2):
        xdata = ts[0:I]
        ydata = np.array(energy2[cycle, i, 0:I])
        width = np.array(dotproducts2[cycle, i, 0:I])*(np.max(energy2)-np.min(energy2))* 0.06
        slidinggraphs2.append(ax2.fill_between(xdata, ydata-width/2, ydata+width/2))

    ###

    space = "                                                      "
    space = space + space
    print(f"{n} \ {N_cycles * N_per_cycle}: time:{np.average(dts) * (N_cycles * N_per_cycle-n) / 60 : 0.2f}min" + space, end = "\r")

    
N_frames = N_cycles * N_per_cycle
#N_frames = 60
anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=N_frames,
                             interval=100)

anim.save("Ideal_L=3_hx=0.2_J=1_smooth-box-Jc.mp4",writer=animation.FFMpegWriter(fps=60))
print("")
print("Done")
#plt.show() 


"""
