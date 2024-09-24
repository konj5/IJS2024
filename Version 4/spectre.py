import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

from procedure import Procedure

proc = Procedure()
proc.setParameters()

data = proc.runProcedure()





#prepare for animation 

energy_n = np.zeros((N_cycles, N_states, N_per_cycle), dtype=float)
states_n = np.zeros((N_cycles, N_states, N_per_cycle), dtype=qt.Qobj)
dotproducts_n = np.zeros((N_cycles, N_states, N_per_cycle), dtype=float)


for i in range(N_cycles):
    for j in range(N_states):
        energy_n[i,j,:] = energy[i*N_per_cycle:(i+1)*N_per_cycle, j]
        states_n[i,j,:] = states[i*N_per_cycle:(i+1)*N_per_cycle, j]
        dotproducts_n[i,j,:] = dotproducts[i*N_per_cycle:(i+1)*N_per_cycle, j]

energy = energy_n
states = states_n
dotproducts = dotproducts_n


from matplotlib import animation

fig, ax = plt.subplots(1,1)

ts = np.linspace(0,1,N_per_cycle)
basegraphs = []
slidinggraphs = []
for i in range(N_states):
    xdata = ts[0:0]
    ydata = np.array(energy[0, i, 0:0])
    width = np.array(dotproducts[0, i, 0:0])
    slidinggraphs.append(ax.fill_between(xdata, ydata-width/2, ydata+width/2))

    basegraphs.append(ax.plot(ts,energy[0,i,:], label = index_to_statename(i), linestyle = "--"))

ax.set_title("Cycle 1")
ax.set_ylabel("Energy")
ax.set_xlabel("Time during single cycle")

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

    global slidinggraphs, basegraphs

    I = n % N_per_cycle
    cycle = n // N_per_cycle

    ax.clear()
    ax.set_title(f"Cycle {cycle + 1}")

    basegraphs = []
    slidinggraphs = []
    for i in range(N_states):

        basegraphs.append(ax.plot(ts,energy[cycle,i,:], label = index_to_statename(i), linestyle = "--"))


    for i in range(N_states):
        xdata = ts[0:I]
        ydata = np.array(energy[cycle, i, 0:I])
        width = np.array(dotproducts[cycle, i, 0:I])*5
        slidinggraphs.append(ax.fill_between(xdata, ydata-width/2, ydata+width/2))

    space = "                                                      "
    space = space + space
    print(f"{n} \ {N_cycles * N_per_cycle}: time:{np.average(dts) * (N_cycles * N_per_cycle-n) / 60 : 0.2f}min" + space, end = "\r")

    
N_frames = N_cycles * N_per_cycle
#N_frames = 60
anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=N_frames,
                             interval=100)

anim.save("xyz_spectre.mp4",writer=animation.FFMpegWriter(fps=60))
plt.show() 

