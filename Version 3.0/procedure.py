import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

sns.set_style("whitegrid")


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
                 end_of_cycle_measurement:bool = False,
                 pass_directly_with_added_errors:bool = False,
                 remake_product_state:bool = False,
                 remake_product_state_LSQ:bool = False):
        
        self.L = L
        self.N = 2*L
        self.tfim_parameters = [J, hx, hz]
        self.bath_parameters = [hb, Bi, Bf, T]
        self.coupling_parameters = [Jc, T]
        self.dt = dt
        self.time_dependant_functions_coeffs = {'Jc': Jc, 'T': T, 'Bi': Bi, 'Bf': Bf}
        self.end_of_cycle_measurement = end_of_cycle_measurement
        self.pass_directly_with_added_errors = pass_directly_with_added_errors 
        self.remake_product_state = remake_product_state
        self.remake_product_state_LSQ = remake_product_state_LSQ

    

    def runProcedure(self, N_cycles:int, measure:callable, setup_state_for_next_cycle:callable, ):
        
        #list vseh časov znotraj enega cikla, ko poberemo podatke
        ts = np.arange(0,self.T+self.dt, self.dt)

        #podatkovna struktura, v katero bomo vpisovali podatke
        data = None

        #Nastavimo začetno stanje
        state = 
        for i in range(N_cycles):
            for j in range(len(ts)):
                #Evolucija stanja za en dt naprej
                state = qt.sesolve(H=H, psi0 = state, tlist=[self.dt])

                #Meritev željenih opazljivk
                measurements = measure(state)

                #Meritve shranimo (Inicializiramo podatkovno strukturo če še ni)
                if data is None:
                    data = np.zeros((len(measurements), N_cycles, len(ts)))
                data[:,i,j] = measurements

            #Stanje podamo v naslednji cikel
            state = setup_state_for_next_cycle(state)

        return data
    












