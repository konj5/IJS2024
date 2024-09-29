import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm.notebook import tqdm
import time, timeit

si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

def tfim_magnetisation(L):

            sx_list = []
            sy_list = []
            sz_list = []

            N = 2*L
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
            
            M = 0
            
            for n in range(L):
                M+=sz_list[n]
            
            return M

def tfim_magnetisation_x(L):

            sx_list = []
            sy_list = []
            sz_list = []

            N = 2*L
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
            
            M = 0
            
            for n in range(L):
                M+=sx_list[n]
            
            return M

def bath_magnetisation(L):

            sx_list = []
            sy_list = []
            sz_list = []

            N = 2*L
            for n in range(N):
                op_list = []
                for m in range(N):
                    op_list.append(si)

                op_list[n] = sz
                sz_list.append(qt.tensor(op_list))
            
            M = 0
            
            for n in range(L, 2*L):
                M+=sz_list[n]
            
            return M


def bath_magnetisation_x(L):

    sx_list = []
    sy_list = []
    sz_list = []

    N = 2*L
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))
    
    M = 0
    
    for n in range(L, 2*L):
        M+=sx_list[n]
    
    return M



def tfim_hamiltonian(L, tfim_params):

    sx_list = []
    sy_list = []
    sz_list = []

    N = 2*L

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))

        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))

    # Ising model hamiltonian construction
    J, hx, hz = tfim_params

    J_list = np.ones(L) * J
    hx_list = np.ones(L) * hx
    hz_list = np.ones(L) * hz

    H_ising_chain = 0

    for n in range(L):
        H_ising_chain += -hx_list[n] * sx_list[n]
        H_ising_chain += -hz_list[n] * sz_list[n]

    for n in range(L):
        H_ising_chain += - J_list[n] * sz_list[n] * sz_list[((n+1)%L)]

    return H_ising_chain