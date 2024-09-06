import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm

sns.set_style("whitegrid")



# Pauli matricies

si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

end_of_cycle_measurement = False          # Meritev spinov v sigma^z bazi po koncu vsakega cikla
pass_directly_with_added_errors = False 
remake_product_state = True
remake_product_state_LSQ = False 
open_dynamics = False          # Odprt sistem z sigma^z collapse operatorji na vsakem spinu
couplings_decrease = True      # Linearno nižanje  sklopitve iz cikla v cikel
coupling = "lin"
hb_decrease = False           #  Linearno nižanje hb polja iz cikla v cikel

#### Simulation parameters ####
dt = 0.1
T = 50.0
times = np.arange(0.0, T+dt, dt)

N_cycles = 25     # Stevilo ciklov algoritma


#### TFIM system parameters ####
L = 3  # length of chain
N = 2*L # total number of sites

J = 1
hx = 0.2
hz = 0

#### Coupling and bath parameters ####
Jc = 0.8
Bi = 4.0
Bf = 0.0

hb = 0.8

tfim_parameters = [J, hx, hz]
bath_parameters = [hb, Bi, Bf, T]
coupling_parameters = [Jc, T]

time_dependant_functions_coeffs = {'Jc': Jc, 'T': T, 'Bi': Bi, 'Bf': Bf}

def bath_z_field_zeeman_drive(t, args):
    """Function B(t)"""
    TT = args['T']
    B_i = args['Bi']
    B_f = args['Bf']

    return (((B_f - B_i)/(TT)) * t + B_i)




def coupling_drive(t, args):
    """Function g(t)"""
    TT = args['T']
    J_c = args['Jc']

    return J_c        



plt.plot(times, [coupling_drive(t, time_dependant_functions_coeffs) for t in times], label=r'$J_c$', lw=3.0)
plt.plot(times, bath_z_field_zeeman_drive(times, time_dependant_functions_coeffs), label=r'$B_z$', lw=3.0)
plt.xlabel('t', fontsize = 18)
plt.ylabel('parameter', fontsize = 18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=18)
plt.grid(linestyle='--', alpha=0.4)

#  Open system collapse operators 


def collapse_operators(L, gamma0=0.05):

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


    gamma_list = np.ones(N) * gamma0

    coll_ops = [gamma_list[i] * sz_list[i] for i in range(N)]

    return coll_ops

# Operators

def tfim_sigmax_magnetisation(L):

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

    Mx = 0
    
    for n in range(L):
        Mx+=sx_list[n]
    
    return Mx
    


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

def run_cycle(L, tfim_params, bath_params, coupling_params, psi0, tlist):


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


    # bath x-field hamiltonian construction
    hb, Bi, Bf, T = bath_params

    hb_bath_list = np.ones(L) * hb
    
    H_bath_x_field = 0

    for m in range(L):
        H_bath_x_field += hb_bath_list[m] * sx_list[m+L]


    # bath z-field hamiltonian construction
     
    H_bath_z_field = 0
    
    for m in range(L):
        H_bath_z_field -= sz_list[m+L]


    def bath_z_field_zeeman_drive(t, args):
        """Function B(t)"""
        TT = args['T']
        B_i = args['Bi']
        B_f = args['Bf']

        return (((B_f - B_i)/(TT)) * t + B_i)


    # bath_system coupling hamiltonian construction

    jc, T = coupling_params

    H_coupling = 0
    
    for n in range(L):
        H_coupling += sz_list[n] * sz_list[n+L]

    
    def coupling_drive(t, args):
        """Function Jc(t)"""
        TT = args['T']
        j_c = args['Jc']

        return j_c 

    # Constructing the whole hamiltonian

    time_dependant_functions_coeffs = {'Jc': jc, 'T': T, 'Bi': Bi, 'Bf': Bf}
    
    H = [H_ising_chain, H_bath_x_field, [H_bath_z_field, bath_z_field_zeeman_drive], [H_coupling, coupling_drive]]

    if open_dynamics:
        col_ops = collapse_operators(L)
    else:
        col_ops = []

    # Running the time evolution

    for i in tqdm(range(1), desc = "mesolve", leave=False):
        result = qt.mesolve(H, psi0, tlist, c_ops=col_ops, e_ops=[], args=time_dependant_functions_coeffs)
    
    eigens = []
    for i in tqdm(range(len(tlist)), desc = "eigens", leave=False):
        H = H_ising_chain + H_bath_x_field + H_bath_z_field * bath_z_field_zeeman_drive(tlist[i], time_dependant_functions_coeffs) + H_coupling * coupling_drive(tlist[i], time_dependant_functions_coeffs)
        eigens.append(H.eigenstates()[1])

    return result, eigens

# calculating the ground state energy of the TFIM

H_TFIM = tfim_hamiltonian(L, tfim_parameters)
M_TFIM = tfim_magnetisation(L)

eigen_energies = H_TFIM.eigenenergies()

E0 = eigen_energies[0]
print("TFIM ground energy: ", E0)

def run_procedure(error_percentage):

    # Setting up the inital states

    bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])
    bath_fully_polarized_density_matrix = bath_fully_polarized_state * bath_fully_polarized_state.dag()

    random_tfim_state = qt.tensor([qt.rand_ket(2) for i in range(L)])
    neel_state = qt.tensor([qt.basis(2,i%2) for i in range(L)])

    initial_state = qt.tensor([neel_state, bath_fully_polarized_state])
    #initial_state = qt.tensor([random_tfim_state, bath_fully_polarized_state])
    initial_state = initial_state * initial_state.dag()

    # Setting for the changing of parameters cycle to cycle

    if couplings_decrease:
        if coupling == "lin":
            f = lambda i:-0.5*Jc*i/N_cycles + Jc 
        if coupling == "fd":
            epsilon = 10**-3
            b = 1/N_cycles * np.log((2-epsilon)/epsilon)
            f = lambda i: 2/(1+np.exp(b*i))
        
        if coupling == "exp":
            epsilon = 10**-3
            b = 1/N_cycles * np.log(epsilon)
            f = lambda i: np.exp(b*i)

        couplings = [Jc * f(i) for i in range(N_cycles)]

    else:
        couplings = [Jc for i in range(N_cycles)]

    if hb_decrease:
        hb_fields = [-0.5*hb*i/N_cycles + hb for i in range(N_cycles)]
    else:
        hb_fields = [hb for i in range(N_cycles)]


    end_of_process_tfim_energies = []
    whole_process_energies = []
    whole_process_entropies = []
    whole_process_tfim_magnetisations = []
    whole_process_bath_magnetisations = []
    whole_process_tfim_magnetisations_x_direction = []
    whole_process_bath_magnetisations_x_direction = []
    whole_process_states = []
    whole_process_eigens = []
    
    # Main loop
    
    for k in tqdm(range(N_cycles), leave = False):
        
        coupling_parameters = [couplings[k], T]
        bath_parameters = [hb_fields[k], Bi, Bf, T]
        
        

        
        
        cycle_result, cycle_eigens = run_cycle(L, tfim_parameters, bath_parameters, coupling_parameters, initial_state, times)     # Runing a single cycle
        cycle_states = cycle_result.states
        cycle_tfim_energy = qt.expect(H_TFIM, cycle_states)
        whole_process_energies.append(cycle_tfim_energy)
        
        whole_process_states.append(cycle_states)
        whole_process_eigens.append(cycle_eigens)
    

        tfim_cycle_states = [st.ptrace([i for i in range(L)]) for st in cycle_states]
        whole_process_entropies.append([qt.entropy_vn(st * st.dag()) for st in tfim_cycle_states])

        whole_process_tfim_magnetisations.append(qt.expect(tfim_magnetisation(L), cycle_states) / L)
        whole_process_tfim_magnetisations_x_direction.append(qt.expect(tfim_magnetisation_x(L), cycle_states) / L)
        whole_process_bath_magnetisations.append(qt.expect(bath_magnetisation(L), cycle_states) / L)
        whole_process_bath_magnetisations_x_direction.append(qt.expect(bath_magnetisation_x(L), cycle_states) / L)

        
        end_of_process_tfim_energies.append(cycle_tfim_energy[-1])

        end_cycle_state = cycle_states[-1]
        end_cycle_state = end_cycle_state.tidyup()

        if end_of_cycle_measurement:

            mag, after_measure_state = qt.measurement.measure_observable(end_cycle_state, tfim_magnetisation(L))
            tfim_part_of_the_end_cycle_state_density_matrix = after_measure_state.ptrace([i for i in range(L)])
            tfim_part_of_the_end_cycle_state = qt.Qobj((tfim_part_of_the_end_cycle_state_density_matrix.diag()), dims=[[2 for i in range(L)], [1 for i in range(L)]])

            initial_state = qt.tensor([tfim_part_of_the_end_cycle_state, bath_fully_polarized_state])

        elif remake_product_state: 
            tfim_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
            
            from pulses_lib import reconstruct_state_L1
            reconstructed_state = reconstruct_state_L1(tfim_part_of_the_end_cycle_state_density_matrix)

            initial_state = qt.tensor([reconstructed_state, bath_fully_polarized_density_matrix])

        elif remake_product_state_LSQ: 
            tfim_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
            
            from pulses_lib import reconstruct_state_L1_LSQ
            reconstructed_state = reconstruct_state_L1_LSQ(tfim_part_of_the_end_cycle_state_density_matrix)

            initial_state = qt.tensor([reconstructed_state, bath_fully_polarized_density_matrix])

        elif pass_directly_with_added_errors:
            m_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
            random_state = qt.tensor([qt.rand_ket(2) for i in range(L)])
            random_state = random_state * random_state.dag()

            keep_amount = 1-error_percentage

            tfim_new = np.sqrt(keep_amount) * m_part_of_the_end_cycle_state_density_matrix + np.sqrt(1-keep_amount) * random_state

            initial_state = qt.tensor([tfim_new, bath_fully_polarized_density_matrix])

        else:

            tfim_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
            initial_state = qt.tensor([tfim_part_of_the_end_cycle_state_density_matrix, bath_fully_polarized_density_matrix])

    return whole_process_states, whole_process_eigens, whole_process_energies


    
states, eigens, energies = run_procedure(-2)

valss = []
for i in tqdm(range(len(eigens)), leave=False, desc = "i"):
    for k in tqdm(range(len(states[i])), leave=False, desc = "k"):
        vals = []
        state = states[i][k]
        eigen = eigens[i][k]
        
        for j in range(len(eigen)):
            #print(state.dag() * eigen[j])
            #print(f"{type(state * eigen[j])}   {i} {k} {j}")
            m_eigen = eigen[j] * eigen[j].dag()
            vals.append(np.abs((state * m_eigen).tr()))
    
        valss.append(vals)
    
        
valss = np.array(valss)
print(valss.shape)

energies_new = []
for i in range(len(energies)):
    for j in range(len(energies[i])):
        energies_new.append(energies[i][j])
energies = energies_new

from matplotlib import animation

def barlist(n): 
    return valss[n,:]

fig, (fig1, fig2) = plt.subplots(2,1, figsize = (20,15))

ts = np.linspace(0,N_cycles * T, len(energies), endpoint=True)
energy = fig1.plot(ts, energies)
for i in np.arange(0, N_cycles*T, T):
    fig1.axvline(i, np.min(energies), np.max(energies), linestyle = "dashed", color = "gray")
    
epoint = fig1.scatter([ts[0]], [energies[0]], color = "red")
fig1.set_xlabel("Time")
fig1.set_ylabel("Energy")


x=range(0, len(valss[0,:]))
barcollection = fig2.bar(x,barlist(1))
fig2.set_xlabel("Eigenstate index")
fig2.set_ylabel("$|\langle \psi|n\\rangle|^2$")



import time

dt = 10
dts = []
t0 = 0

def animate(i):
    global dt, t0, dts
    dt = time.time()-t0
    dts.append(dt)
    if len(dts) > 10:
        dts = dts[-10: -1]
    t0 = time.time()
    y=barlist(i+1)
    print("                                                                                        ",end="\r")
    print(f"{i} \ {len(valss[:,0])}: time:{np.average(dts) * (len(valss[:,0])-i) / 60 : 0.2f}min", end = "\r")
    for k, b in enumerate(barcollection):
        b.set_height(y[k])
        
    epoint.set_offsets((ts[i], energies[i]))
    
N = len(valss[:,0])
#N = 40
anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=N,
                             interval=100)

anim.save('animation_from_py.mp4',writer=animation.FFMpegWriter(fps=60))
plt.show() 