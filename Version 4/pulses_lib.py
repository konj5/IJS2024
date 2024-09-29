import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm

sns.set_style("whitegrid")

si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


class pulseDrive():
    
    def __init__(self, magnitude:float, direction:str, t0:float, tmax:float) -> None:
          self.mag = magnitude
          self.dir = direction
          self.t0 = t0
          self.tmax = tmax

    def __repr__(self):
        return f"{self.dir} pulse magnitude {self.mag}: between {self.t0}, {self.tmax}"
    
    def __str__(self):
        return f"{self.dir} pulse magnitude {self.mag}: between {self.t0}, {self.tmax}"

    def drive(self,t):
        tmid = (self.tmax+self.t0)/2
        sigma = (self.tmax-self.t0)/6
        if t > self.t0 and t < self.tmax:
            return 1/2*self.mag*1/np.sqrt(2*np.pi*sigma**2) *np.exp(-(t-tmid)**2/(2*sigma**2))
        else:
            return 0
        
def __TFIM_Hamiltonian(N, J, hx, hz):
    sx_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))

        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))

    J_list = np.ones(N) * J
    hx_list = np.ones(N) * hx
    hz_list = np.ones(N) * hz

    H_ising_chain = 0

    for n in range(N):
        H_ising_chain += -hx_list[n] * sx_list[n]
        H_ising_chain += -hz_list[n] * sz_list[n]

    for n in range(N):
        H_ising_chain += - J_list[n] * sz_list[n] * sz_list[((n+1)%N)]

    return H_ising_chain

def __multilist_to_pulseset(N:int, instruction: list, t0:float, tmax:float):
    x_set, z_set = [[] for i in range(N)], [[] for i in range(N)] 

    """Instruction format = [[direction:string, magnitude:float, list of spins indices:list, time_order:int]]"""
    time_indices = set()
    for instr in instruction:
        if instr[0] == "x":
            for i in instr[2]:
                x_set[i].append([instr[1], instr[3]])
                time_indices.add(instr[3])
        elif instr[0] == "z":
            for i in instr[2]:
                z_set[i].append([instr[1], instr[3]])
                time_indices.add(instr[3])
        else:
            raise ValueError("Direction must be x or z")

    #for i in range(min(time_indices), max(time_indices)+1):
        #if i not in time_indices:
        #   raise ValueError("Time indices must have no holes between them")

    for i in range(N):
        x_set[i] = sorted(x_set[i], key = lambda x: x[1])
        z_set[i] = sorted(z_set[i], key = lambda x: x[1])

    time_indices = [x for x in range(min(time_indices), max(time_indices)+1)]
    n = len(time_indices)
    ts = np.linspace(t0, tmax, n+1)
    tis_to_edgevals = dict()

    for i in range(min(time_indices), max(time_indices)+1):
        tis_to_edgevals[time_indices[i]] = (ts[i],ts[i+1])
    
    for i in range(N):
        for j in range(len(x_set[i])):
            x_set[i][j][1] = tis_to_edgevals[x_set[i][j][1]]
        for j in range(len(z_set[i])):
            z_set[i][j][1] = tis_to_edgevals[z_set[i][j][1]]

    return x_set, z_set

def multipulseset_to_function(pulse_set: list):
    fs = [None for i in range(len(pulse_set))]

    for i in range(len(pulse_set)):

        def f(t):
            val = 0
            for spin in pulse_set[i]:
                val += pulseDrive(spin[0], "irrelevant", spin[1][0], spin[1][1]).drive(t)
            return val
        
        fs[i] = f

    return fs

def multi_bit_rotation(N, J, hx, hz, psi0, instruction_list:list, t0:float, tmax:float):
    tlist = np.linspace(t0,tmax,100)

    x_set, z_set = __multilist_to_pulseset(N,instruction_list, t0, tmax)
    f_xs = multipulseset_to_function(x_set); f_zs = multipulseset_to_function(z_set)


    sx_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))

        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))

    #Create hamultonian
    H = [__TFIM_Hamiltonian(N, J, hx, hz)]

    for i in range(N):
        H.append([sx_list[i], f_xs[i]])
        H.append([sz_list[i], f_zs[i]])

    result = qt.mesolve(H, psi0, tlist, args=[])
    return result.states[-1]


def generate_single_spin_state(phi:float, theta:float):
    return np.cos(theta/2) * qt.basis(2,0) + np.sin(theta/2) * np.exp(1j * phi) * qt.basis(2,1)

def make_operator_lists(N:int):
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))

        op_list[n] = sy
        sy_list.append(qt.tensor(op_list))

        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))

    return sx_list, sy_list, sz_list

def printLists(names, lists):
    n = len(lists)
    N = len(lists[0])

    line0 = "{:13}".format(" ")
    for name in names:
        line0 += "{:10}".format(name) + "{:1}".format(" | ")
    
    print(line0)

    for i in range(N):
        line = "{:10}".format(f"Spin {i}:") + " | " 
        for lst in lists:
            line += "{:10f}".format(lst[i]) + "{:1}".format(" | ")
        print(line)

def reconstruct_state_L1(state):
    N = int(np.round(np.log2(state.shape[0])))
    sx_list, sy_list, sz_list = make_operator_lists(N)
    exactZ, exactY , exactX = [], [], []
    for i in range(N):
        exactX.append(qt.expect(sx_list[i], state))
        exactY.append(qt.expect(sy_list[i], state))
        exactZ.append(qt.expect(sz_list[i], state))

    substates = []
    for i in range(N):
        theta = np.arccos(exactZ[i])
        if exactZ[i] != 1.0:
            phi = np.arccos(exactX[i] / np.sqrt(1-exactZ[i]**2))
        else:
            phi = 0
        
        substates.append(generate_single_spin_state(phi,theta))
    newstate = qt.tensor(substates)
    return newstate

def reconstruct_state_L1_LSQ(state):
    def find_state(sx,sy,sz):
        from scipy.optimize import least_squares

        def f(x):
            theta = x[0]
            phi = x[1]

            return np.array([np.sin(theta)*np.cos(phi) - sx, np.sin(theta)*np.sin(phi) - sy, np.cos(theta) - sz])
        


        sol = least_squares(f, x0=[0,0])
        
        return generate_single_spin_state(sol.x[1], sol.x[0])
    
    N = int(np.round(np.log2(state.shape[0])))
    sx_list, sy_list, sz_list = make_operator_lists(N)
    exactZ, exactY , exactX = [], [], []
    for i in range(N):
        exactX.append(qt.expect(sx_list[i], state))
        exactY.append(qt.expect(sy_list[i], state))
        exactZ.append(qt.expect(sz_list[i], state))

    substates = []
    for i in range(N):
        substates.append(find_state(exactX[i], exactY[i], exactZ[i]))
    newstate = qt.tensor(substates)
    return newstate