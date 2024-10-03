import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

import measurables
from pulses_lib import reconstruct_state_L1 as reconstruct_state_xz, reconstruct_state_L1_LSQ as reconstruct_state_xyz

sns.set_style("whitegrid")

class Hamiltonian():
    #Razlaga uporabljenih funkcij
    #couling_decrease: Normirana funkcija, ki se pomnoži z amplitudo Jc, da dobimo odvisnost Jc iz cikla v cikel
    #Jc_drive: Normirana funkcija, ki se pomnoži z amplitudo Jc, da dobimo odvisnost Jc znotraj cikla
    #bath_z_field_zeeman_drive: Funkcija, ki poda eksaktno odvisnost za bath_z_field znotraj cikla


    def __init__(self, L:int, N_cycles:int, chain_hamiltonian:qt.Qobj, bath_parameters:list, coupling_parameters:list, coupling_decrease:callable = "use_default", Jc_drive:callable = "use_default", bath_z_field_zeeman_drive:callable = "use_default") -> None:
        self.L = L
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
        self.H_ising_chain = chain_hamiltonian

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

    #Metoda, ki sestavi in vrne časovno odvisni hamiltonian
    def getHamiltonian(self, cycleNumber:int):
        #Dobimo vrednost Jc za dani cikel
        Jc0, T = self.coupling_parameters
        Jc = self.coupling_decrease(cycleNumber) * Jc0

        #Sestavimo hamiltonian, ki gre v sesolve
        H = [self.H_ising_chain, self.H_bath_x_field, [self.H_bath_z_field, self.bath_z_field_zeeman_drive], [self.H_coupling, lambda t: self.Jc_drive(t) * Jc]]
        
        return H
    
    #Metoda, ki sestavi in vrne hamiltonian v določenem trenutku
    def getInstantaneousHamiltonian(self, cycleNumber:int, t:float):
        #Dobimo vrednost Jc za dani cikel
        Jc0, T = self.coupling_parameters
        Jc = self.coupling_decrease(cycleNumber) * Jc0

        #Sestavimo hamiltonian, ki gre v sesolve
        H = self.H_ising_chain + self.H_bath_x_field + self.H_bath_z_field * self.bath_z_field_zeeman_drive(t) + self.H_coupling * self.Jc_drive(t) * Jc
        return H

    #Metoda, ki vrne hamiltonian verige
    eigenenergies, eigenstates = None, None
    def getChainHamiltonianEigens(self):
        #Poskrbimo, da diagonalizacije ne izvajamo več kot enkrat
        if self.eigenenergies is None:

            eigenenergies, eigenstates = self.H_ising_chain.H_ising_chain.eigenstates()
            self.eigenenergies, self.eigenstates = eigenenergies, eigenstates
            return eigenenergies, eigenstates

        return self.eigenenergies, self.eigenstates
        

    


#razred, ki definira kakšno proceduro izvajamo
class Procedure:
    
    #Ustvari prazen objekt
    def __init__(self) -> None:
        self.setParameters()
    
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
                 error_percentage:float = 0,
                 ):
        
        self.L = L
        self.N = 2*L
        self.T = T
        self.tfim_parameters = [J, hx, hz]
        self.bath_parameters = [hb, Bi, Bf, T]
        self.coupling_parameters = [Jc, T]
        self.dt = dt
        self.time_dependant_functions_coeffs = {'Jc': Jc, 'T': T, 'Bi': Bi, 'Bf': Bf}
        self.error_percentage = error_percentage
        self.v3 = None

    #Shrani hamiltonian trenutnega procesa
    def set_current_hamiltonian(self, hamiltonian:Hamiltonian):
        self.hamiltonian = hamiltonian
        

    #Shrani čas trenutnega procesa
    def set_current_time(self, t:float):
        self.t = t

    #Shrani trenutni cikel trenutnega procesa
    def set_current_cycle_number(self, cycle_number:int):
        self.cycle_number = cycle_number

    #Privzeto začetno stanje
    @staticmethod
    def default_get_start_statevector(L:int):
        neel_state = qt.tensor([qt.basis(2,i%2) for i in range(L)])
        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])

        initial_state = qt.tensor([neel_state, bath_fully_polarized_state])
        return initial_state
    
    @staticmethod
    def default_get_start_density_matrix(L:int):
        vector = Procedure.default_get_start_statevector(L)
        return vector * vector.dag()
    
    ###Več možnih metod, za podajanje stanja med cikli
    @staticmethod
    def pass_full_density_matrix(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        chain_density_matrix = qt.ptrace(state, [i for i in range(L)])
        
        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])
        bath_fully_polarized_density_matrix = bath_fully_polarized_state * bath_fully_polarized_state.dag()

        return qt.tensor([chain_density_matrix, bath_fully_polarized_density_matrix])
    
    @staticmethod
    def pass_density_matrix_with_errors(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        chain_density_matrix = qt.ptrace(state, [i for i in range(L)])

        random_state = qt.tensor([qt.rand_ket(2) for i in range(L)])
        random_density_matrix = random_state * random_state.dag()

        chain_density_matrix = np.sqrt(1-proc.error_percentage) * chain_density_matrix + np.sqrt(proc.error_percentage) * random_density_matrix
        
        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])
        bath_fully_polarized_density_matrix = bath_fully_polarized_state * bath_fully_polarized_state.dag()

        return qt.tensor([chain_density_matrix, bath_fully_polarized_density_matrix])
    
    
    @staticmethod
    def pass_measure_z_direction(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        mag, after_measure_state = qt.measurement.measure_observable(state, measurables.tfim_magnetisation(L))
        tfim_part_of_the_end_cycle_state_density_matrix = after_measure_state.ptrace([i for i in range(L)])
        tfim_part_of_the_end_cycle_state = qt.Qobj((tfim_part_of_the_end_cycle_state_density_matrix.diag()), dims=[[2 for i in range(L)], [1 for i in range(L)]])

        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])
        bath_fully_polarized_density_matrix = bath_fully_polarized_state * bath_fully_polarized_state.dag()

        return qt.tensor([tfim_part_of_the_end_cycle_state, bath_fully_polarized_state])
    
    @staticmethod
    def pass_remake_product_state_with_xz(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        state = state * state.dag()
        tfim_part_of_the_end_cycle_state_density_matrix = state.ptrace([i for i in range(L)])

        reconstructed_state = reconstruct_state_xz(tfim_part_of_the_end_cycle_state_density_matrix)

        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])

        return qt.tensor([reconstructed_state, bath_fully_polarized_state])
    
    @staticmethod
    def pass_remake_product_state_with_xyz(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        state = state * state.dag()
        tfim_part_of_the_end_cycle_state_density_matrix = state.ptrace([i for i in range(L)])
    
        reconstructed_state = reconstruct_state_xyz(tfim_part_of_the_end_cycle_state_density_matrix)

        bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])

        return qt.tensor([reconstructed_state, bath_fully_polarized_state])
    



    #Metoda za izračun absolutne vrednosti kvadrata skalarnega produkta stanj
    @staticmethod
    def abs_squared_scalarproduct(a,b):
        prod = a.dag() * b
        if type(prod) is not qt.Qobj:
            return np.abs(prod)**2
        else:
            return np.abs((a*(b*b.dag())).tr())

    #Nekaj primerov metod measure

    @staticmethod
    def measure_energy(state:qt.Qobj, proc:"Procedure"):
        L = proc.L
        #chain_density_matrix = Procedure.extract_chain(state, L)
        
        energy = qt.expect(measurables.tfim_hamiltonian(L, proc.tfim_parameters), state)
        return [energy]
        
    
    @staticmethod
    def measure_normalization_density_matrix(state:qt.Qobj, proc:"Procedure"):
        return [(state*state.dag()).tr()]
    
    @staticmethod
    def measure_normalization_state_vector(state:qt.Qobj, proc:"Procedure"):
        return [state.dag()*state]
    
    
    @staticmethod
    def measure_eigenstate_projections(state:qt.Qobj, proc:"Procedure"):
        if proc.v3 == False: raise Exception("Lastna stanja z operatorji niso implementirana") 
        hamiltonian = proc.hamiltonian
        H = hamiltonian.getInstantaneousHamiltonian(proc.cycle_number, proc.t)
        eigenenergies, eigenstates = H.eigenstates()

        data = []
        
        for i in range(len(eigenstates)):
            data.append(eigenenergies[i])
            data.append(proc.abs_squared_scalarproduct(state, eigenstates[i]))

        #Način shranjevanja: energija1, produkt1, energija2, produkt2, energija3, ....
        return data
    
    @staticmethod 
    def measure_chain_eigenstate_projections(state:qt.Qobj, proc:"Procedure"):
        if proc.v3 == False: raise Exception("Lastna stanja z operatorji niso implementirana") 
        hamiltonian = proc.hamiltonian
        eigenenergies, eigenstates = hamiltonian.getChainHamiltonianEigens()

        tfim_part_of_the_end_cycle_state_density_matrix = state.ptrace([i for i in range(proc.L)])

        data = []
        
        for i in range(len(eigenstates)):
            data.append(eigenenergies[i])
            data.append(proc.abs_squared_scalarproduct(tfim_part_of_the_end_cycle_state_density_matrix, eigenstates[i]*eigenstates[i].dag()))

        #Način shranjevanja: energija1, produkt1, energija2, produkt2, energija3, ....
        return data

        

    
    #Razlaga funkcij, ki jih runProcedure zahteva kot argumente

    #measure(state:qt.Qobj, proc:Procedure): funkcija ki prevzame stanje verige in kopeli iz vrne enodimenzionalen list ali np.ndarray, ki vsebuje meritve, ki nas zanimajo.
    #setup_state_for_next_cycle(state:qt.Qobj, proc:Procedure): funkcija ki vzame končno stanje verige in kopeli, ter vrne začetno stanje obeh, za naslednji cikel
    #get_startstate(L:int): funkcija, ki vrne stanje verige in kopeli, na začetku prvega cikla

    def runProcedure(self, N_cycles:int, measure:callable, chain_hamiltonian:qt.Qobj, setup_state_for_next_cycle:callable = pass_full_density_matrix, using_state_vectors:bool = False, using_density_matrices:bool = False, get_startstate:callable="use_default",
                      coupling_decrease:callable="use_default", bath_z_field_zeeman_drive:callable="use_default", Jc_drive:callable="use_default"):
        
        #list vseh časov znotraj enega cikla, ko poberemo podatke
        ts = np.arange(0,self.T+self.dt, self.dt)

        #ustvarimo hamiltonian
        hamiltonian = Hamiltonian(self.L, N_cycles, chain_hamiltonian, self.bath_parameters, self.coupling_parameters, 
                                  coupling_decrease="use_default", bath_z_field_zeeman_drive="use_default", Jc_drive="use_default")
        
        self.set_current_hamiltonian(hamiltonian)

        #podatkovna struktura, v katero bomo vpisovali podatke [meritve, cikli, čaz znotraj cikla]
        data = None

        #Ugotovi, ali delamo z vektorji stanj ali gostotnimi matrikami
        if get_startstate == "use_default":
            if using_state_vectors == using_density_matrices:
                if using_state_vectors == False: raise Exception("You have to choose either using_state_vectors or using_density_matrices")
                raise Exception("You can't choose both using_state_vectors and using_density_matrices")
            
            if using_state_vectors:
                get_startstate = Procedure.default_get_start_statevector
            else:
                get_startstate = Procedure.default_get_start_density_matrix
                

        #Nastavimo začetno stanje
        state = get_startstate(self.L)

        def isFunction(x):
            try:
                x(state,self)
                return True
            except:
                return False
            
        v3_or_v4 = isFunction(measure)
        print(v3_or_v4)

        for i in tqdm(range(N_cycles), desc = "Cycle"):

            if v3_or_v4:
                self.v3 = True

                for j in tqdm(range(len(ts)-1), desc = "sesolve" if using_state_vectors else "mesolve", leave=False):
                    #Evolucija stanja za en dt naprej
                    if using_state_vectors:
                        result = qt.sesolve(H=hamiltonian.getHamiltonian(cycleNumber=i), psi0 = state, tlist=[ts[j], ts[j+1]])
                    else:
                        result = qt.mesolve(H=hamiltonian.getHamiltonian(cycleNumber=i), rho0 = state, tlist=[ts[j], ts[j+1]])

                    #Posebej moramo zapisati še začetno stanje, ker ga sicer nebi zapisali
                    if j == 0:
                        state = result.states[0]

                        self.set_current_cycle_number(cycle_number=0)
                        self.set_current_time(t=ts[0])
                        measurements = measure(state, self)

                        #Prvič moramo še inicializirati podatkovno strukturo
                        if i == 0:
                            data = np.zeros((len(measurements), N_cycles, len(ts)), dtype=np.complex64)
                        data[:,i,0] = measurements

                    state = result.states[-1]


                    #Meritev željenih opazljivk
                    self.set_current_cycle_number(cycle_number=i)
                    self.set_current_time(t=ts[j+1])
                    measurements = measure(state, self)

                    #Meritve shranimo
                    data[:,i,j+1] = np.array(measurements)

            else:
                self.v3 = False

                #Izvedemo časovno evolucijo za en cikel
                if using_state_vectors:
                    result = qt.sesolve(H=hamiltonian.getHamiltonian(cycleNumber=i), psi0 = state, tlist=ts, e_ops=measure, progress_bar = "tqdm", options=qt.solver.Options(store_final_state = True, store_states = True ))
                else:
                    result = qt.mesolve(H=hamiltonian.getHamiltonian(cycleNumber=i), rho0 = state, tlist=ts, e_ops=measure, progress_bar = "tqdm", options=qt.solver.Options(store_final_state = True, store_states = True))

                #Shranimo meritve
                for j in range(len(result.expect)):
                    data[j,i,:] = result.expect[j]

            #Stanje podamo v naslednji cikel
            state = setup_state_for_next_cycle(state, self)

        return data
    
    

#Primer uporabe
if __name__ == "__main__":
    proc = Procedure()
    proc.setParameters(L=3)

    times = np.arange(0,proc.T+proc.dt, proc.dt)

    T = proc.T
    N_cycles = 30

    data = proc.runProcedure(N_cycles=N_cycles, chain_hamiltonian=measurables.tfim_hamiltonian(proc.L, proc.tfim_parameters), measure=Procedure.measure_energy,
                          setup_state_for_next_cycle=Procedure.pass_full_density_matrix,
                          coupling_decrease="use_default", using_density_matrices=True)

    H_TFIM = measurables.tfim_hamiltonian(proc.L, proc.tfim_parameters)


    eigen_energies = H_TFIM.eigenenergies()

    E0 = eigen_energies[0]
    print("TFIM ground energy: ", E0)

    cmap = plt.get_cmap('jet')
    COLORS = [cmap(i) for i in np.linspace(.01, .99, N_cycles)]

    for i in range(len(data[0,:,0])):
        plt.plot(times, data[0,i,:], color=COLORS[i])

    plt.axhline(y=E0, linestyle='--', color='black', label=r"$E_0$")

    norm = mpl.colors.Normalize(vmin=0, vmax=N_cycles)
    scalarmappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(scalarmappable, ax=plt.gca(), label="Cycle number")

    plt.xlabel("Time (during single cycle)", fontsize = 15)
    plt.ylabel("TFIM energy", fontsize = 15)
    plt.title("TFIM energies during cycles", fontsize = 16)
    plt.show()


    for i in range(len(data[0,0,:])):
        line = ""
        for j in range(N_cycles):
            line += f"{data[0,j,i]} "
        print(line)
