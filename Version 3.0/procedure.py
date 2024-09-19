import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

import measurables

sns.set_style("whitegrid")

class Hamiltonian():
    #Razlaga uporabljenih funkcij
    #couling_decrease: Normirana funkcija, ki se pomnoži z amplitudo Jc, da dobimo odvisnost Jc iz cikla v cikel
    #Jc_drive: Normirana funkcija, ki se pomnoži z amplitudo Jc, da dobimo odvisnost Jc znotraj cikla
    #bath_z_field_zeeman_drive: Funkcija, ki poda eksaktno odvisnost za bath_z_field znotraj cikla


    def __init__(self, L:int, N_cycles:int, tfim_parameters:list, bath_parameters:list, coupling_parameters:list, coupling_decrease:callable = "use_default", Jc_drive:callable = "use_default", bath_z_field_zeeman_drive:callable = "use_default") -> None:
        self.L = L
        self.tfim_parameters = tfim_parameters # [J, hx, hz]
        self.bath_parameters = bath_parameters # [hb, Bi, Bf, T]
        self.coupling_parameters = coupling_parameters #  [Jc, T]

        def default_coupling_decrease(cycle_number):
            return 1 - 0.5 * cycle_number / N_cycles 

        self.coupling_decrease = coupling_decrease if coupling_decrease != "use_default" else default_coupling_decrease # funkcija, ki opiše kako Jc pada iz cikla v cikel

        def default_bath_z_field_zeeman_drive(t):
            T = coupling_parameters[1]
            B_i = bath_parameters[1]
            B_f = bath_parameters[2]

            return (((B_f - B_i)/(T)) * t + B_i)

        self.bath_z_field_zeeman_drive = bath_z_field_zeeman_drive if bath_z_field_zeeman_drive != "use_default" else default_bath_z_field_zeeman_drive # funkcija, ki opiše kako se z polje v kopeli spreminja znotraj posameznega cikla

        self.Jc_drive = Jc_drive if Jc_drive != "use_default" else lambda t: 1# funkcija, ki opiše kako se Jc spreminja znotraj posameznega cikla

        #Ustvarimo spinske operatorje
        si = qt.qeye(2)
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()
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

        self.sx_list = sx_list
        self.sy_list = sy_list
        self.sz_list = sz_list

        #Ustvarimo časovno popolnoma neodvisen hamiltonian verige
        J, hx, hz = tfim_parameters

        J_list = np.ones(L) * J
        hx_list = np.ones(L) * hx
        hz_list = np.ones(L) * hz

        H_ising_chain = 0

        for n in range(L):
            H_ising_chain += -hx_list[n] * sx_list[n]
            H_ising_chain += -hz_list[n] * sz_list[n]

        for n in range(L):
            H_ising_chain += - J_list[n] * sz_list[n] * sz_list[((n+1)%L)]

        self.H_ising_chain = H_ising_chain

        #Ustvarimo časovno popolnoma neodvisen del hamiltoniana kopeli
        H_bath_x_field = 0

        hb, Bi, Bf, T = bath_parameters
        hb_bath_list = np.ones(L) * hb
        for m in range(L):
            H_bath_x_field += hb_bath_list[m] * sx_list[m+L]
        
        self.H_bath_x_field = H_bath_x_field

        #Ustvarimo od cikla neodvisen hamiltonian kopeli

        H_bath_z_field = 0
            
        for m in range(L):
            H_bath_z_field -= sz_list[m+L]

        self.H_bath_z_field = H_bath_z_field

        #Ustvarimo časovni neodvisen del sklopitve
        H_coupling = 0
        for n in range(L):
            H_coupling += sz_list[n] * sz_list[n+L]

        self.H_coupling = H_coupling

    def getHamiltonian(self, cycleNumber:int):
        #Dobimo vrednost Jc za dani cikel
        Jc0, T = self.coupling_parameters
        Jc = self.coupling_decrease(cycleNumber) * Jc0

        #Sestavimo hamiltonian, ki gre v sesolve
        H = [self.H_ising_chain, self.H_bath_x_field, [self.H_bath_z_field, self.bath_z_field_zeeman_drive], [self.H_coupling, lambda t: self.Jc_drive(t) * Jc]]
        
        return H
    

#razred, ki definira kakšno proceduro izvajamo
class Procedure:
    
    #Ustvari prazen objekt
    def __init__(self) -> None:
        pass
    
    #Nastavi parametre precedure
    def setParameters(self, 
                 L:int = 3,
                 J:float = 1.0,
                 hx:float = 0.2,
                 hz:float = 0.0,
                 Jc:float = 0.8,
                 Bi:float = 4.0,
                 Bf:float = 0.0,
                 hb:float = 0.8,
                 T:float = 50,
                 dt:float = 0.1,
                 ):
        
        self.L = L
        self.N = 2*L
        self.T = T
        self.tfim_parameters = [J, hx, hz]
        self.bath_parameters = [hb, Bi, Bf, T]
        self.coupling_parameters = [Jc, T]
        self.dt = dt
        self.time_dependant_functions_coeffs = {'Jc': Jc, 'T': T, 'Bi': Bi, 'Bf': Bf}

    @staticmethod
    def default_get_startstate(L: int):
        neel_state = qt.tensor([qt.basis(2,i%2) for i in range(L)])
        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])

        initial_state = qt.tensor([neel_state, bath_fully_polarized_state])
        return initial_state
    
    @staticmethod
    def extract_chain(state:qt.Qobj, L:int):
        L = proc.L
        state_density_matrix = state * state.dag()
        chain_density_matrix = state_density_matrix.ptrace([i for i in range(L)])

        return chain_density_matrix

        
    @staticmethod
    def pass_directly(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        chain_density_matrix = Procedure.extract_chain(state, L)
        
        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])
        bath_fully_polarized_density_matrix = bath_fully_polarized_state * bath_fully_polarized_state.dag()

        return qt.tensor([chain_density_matrix, bath_fully_polarized_density_matrix])
    
    @staticmethod
    def measure_energy(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        #chain_density_matrix = Procedure.extract_chain(state, L)
        
        energy = qt.expect(measurables.tfim_hamiltonian(L, proc.tfim_parameters), state)
        return [energy]
    
    #Razlaga funkcij, ki jih runProcedure zahteva kot argumente

    #measure(state:qt.Qobj, proc:Procedure): funkcija ki prevzame stanje verige in kopeli iz vrne enodimenzionalen list ali np.ndarray, ki vsebuje meritve, ki nas zanimajo. Te morajo biti številke
    #setup_state_for_next_cycle(state:qt.Qobj, proc:Procedure): funkcija ki vzame končno stanje verige in kopeli, ter vrne začetno stanje obeh, za naslednji cikel
    #get_startstate(L:int): funkcija, ki vrne stanje verige in kopeli, na začetku prvega cikla

    def runProcedure(self, N_cycles:int, measure:callable, setup_state_for_next_cycle:callable = pass_directly, get_startstate:callable = default_get_startstate, 
                      coupling_decrease:callable="use_default", bath_z_field_zeeman_drive:callable="use_default", Jc_drive:callable="use_default"):
        
        #list vseh časov znotraj enega cikla, ko poberemo podatke
        ts = np.arange(0,self.T+self.dt, self.dt)

        #ustvarimo hamiltonian
        hamiltonian = Hamiltonian(self.L, N_cycles, self.tfim_parameters, self.bath_parameters, self.coupling_parameters, 
                                  coupling_decrease="use_default", bath_z_field_zeeman_drive="use_default", Jc_drive="use_default")

        #podatkovna struktura, v katero bomo vpisovali podatke [meritve, cikli, čaz znotraj cikla]
        data = None

        #Nastavimo začetno stanje
        state = get_startstate(self.L)
        for i in tqdm(range(N_cycles), desc = "Cycle"):
            for j in tqdm(range(len(ts)-1), desc = "sesolve", leave=False):
                #Evolucija stanja za en dt naprej
                result = qt.sesolve(H=hamiltonian.getHamiltonian(cycleNumber=i), psi0 = state, tlist=[ts[j], ts[j+1]])
                state = result.states[-1]

                #Meritev željenih opazljivk
                measurements = measure(state, self)

                #Meritve shranimo (Inicializiramo podatkovno strukturo če še ni)
                if data is None:
                    data = np.zeros((len(measurements), N_cycles, len(ts)), dtype=np.complex64)
                data[:,i,j] = np.array(measurements)

            #Stanje podamo v naslednji cikel
            state = setup_state_for_next_cycle(state, self)

        return data
    

proc = Procedure()
proc.setParameters(L=10)
print(proc.runProcedure(N_cycles=1, measure=Procedure.measure_energy))

