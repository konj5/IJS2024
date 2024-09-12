import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

sns.set_style("whitegrid")

# Pauli matricies

si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()



open_dynamics = False          # Odprt sistem z sigma^z collapse operatorji na vsakem spinu
couplings_decrease = True      # Linearno nižanje  sklopitve iz cikla v cikel
coupling = "lin"
hb_decrease = False           #  Linearno nižanje hb polja iz cikla v cikel



class Procedure:
    
    def __init__(self, 
                 L = 3,
                 J = 1.0,
                 hx = 0.2,
                 hz = 0.0,
                 Jc = 0.8,
                 Bi = 4.0,
                 Bf = 0.0,
                 hb = 0.8,
                 N_cycles = 25,
                 T = 50,
                 dt = 0.1,
                 end_of_cycle_measurement = False,
                 pass_directly_with_added_errors = False,
                 remake_product_state = False,
                 remake_product_state_LSQ = False):
        
        self.L = L
        self.N = 2*L
        self.N_cycles = N_cycles
        self.tfim_parameters = [J, hx, hz]
        self.bath_parameters = [hb, Bi, Bf, T]
        self.coupling_parameters = [Jc, T]
        self.dt = dt
        self.time_dependant_functions_coeffs = {'Jc': Jc, 'T': T, 'Bi': Bi, 'Bf': Bf}
        self.end_of_cycle_measurement = end_of_cycle_measurement
        self.pass_directly_with_added_errors = pass_directly_with_added_errors 
        self.remake_product_state = remake_product_state
        self.remake_product_state_LSQ = remake_product_state_LSQ
                
    

    def run_with_mesolve(self):
        
        #repack the variables
        L = self.L
        N = self.N
        N_cycles = self.N_cycles
        tfim_parameters = self.tfim_parameters
        bath_parameters = self.bath_parameters
        coupling_parameters = self.coupling_parameters
        time_dependant_functions_coeffs = self.time_dependant_functions_coeffs
        J, hx, hz = tfim_parameters
        hb, Bi, Bf, T = bath_parameters
        Jc, T = coupling_parameters
        dt = self.dt
        times = np.arange(0.0, T+dt, dt)
        end_of_cycle_measurement = self.end_of_cycle_measurement
        pass_directly_with_added_errors = self.pass_directly_with_added_errors 
        remake_product_state = self.remake_product_state
        remake_product_state_LSQ = self.remake_product_state_LSQ

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
            eigensTFIM = []
            for i in tqdm(range(len(tlist)), desc = "eigens", leave=False):
                H = H_ising_chain + H_bath_x_field + H_bath_z_field * bath_z_field_zeeman_drive(tlist[i], time_dependant_functions_coeffs) + H_coupling * coupling_drive(tlist[i], time_dependant_functions_coeffs)
                eigens.append(H.eigenstates())
                
                H = H_ising_chain
                eigensTFIM.append(H.eigenstates())
                


            return result, eigens, eigensTFIM

        H_TFIM = tfim_hamiltonian(L, tfim_parameters)
        M_TFIM = tfim_magnetisation(L)

        eigen_energies = H_TFIM.eigenenergies()

        E0 = eigen_energies[0]


        def run_procedure(error_percentage=0):

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
            whole_process_eigensTFIM = []
            
            # Main loop
            
            for k in tqdm(range(N_cycles), leave = False):
                
                coupling_parameters = [couplings[k], T]
                bath_parameters = [hb_fields[k], Bi, Bf, T]
                
                

                
                
                cycle_result, cycle_eigens, cycle_eigensTFIM = run_cycle(L, tfim_parameters, bath_parameters, coupling_parameters, initial_state, times)     # Runing a single cycle
                cycle_states = cycle_result.states
                cycle_tfim_energy = qt.expect(H_TFIM, cycle_states)
                whole_process_energies.append(cycle_tfim_energy)
                
                whole_process_states.append(cycle_states)
                whole_process_eigens.append(cycle_eigens)
                whole_process_eigensTFIM.append(cycle_eigensTFIM)

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
                    reconstructed_density_matrix = reconstructed_state * reconstructed_state.dag()

                    initial_state = qt.tensor([reconstructed_density_matrix, bath_fully_polarized_density_matrix])

                elif remake_product_state_LSQ: 
                    tfim_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
                    
                    from pulses_lib import reconstruct_state_L1_LSQ
                    reconstructed_state = reconstruct_state_L1_LSQ(tfim_part_of_the_end_cycle_state_density_matrix)
                    reconstructed_density_matrix = reconstructed_state * reconstructed_state.dag()

                    initial_state = qt.tensor([reconstructed_density_matrix, bath_fully_polarized_density_matrix])

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

            return whole_process_states, whole_process_eigens, whole_process_energies, whole_process_eigensTFIM
        return run_procedure()
    
    def run_with_sesolve(self):
        
        #repack the variables
        L = self.L
        N = self.N
        N_cycles = self.N_cycles
        tfim_parameters = self.tfim_parameters
        bath_parameters = self.bath_parameters
        coupling_parameters = self.coupling_parameters
        time_dependant_functions_coeffs = self.time_dependant_functions_coeffs
        J, hx, hz = tfim_parameters
        hb, Bi, Bf, T = bath_parameters
        Jc, T = coupling_parameters
        dt = self.dt
        times = np.arange(0.0, T+dt, dt)
        end_of_cycle_measurement = self.end_of_cycle_measurement
        pass_directly_with_added_errors = self.pass_directly_with_added_errors 
        remake_product_state = self.remake_product_state
        remake_product_state_LSQ = self.remake_product_state_LSQ

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

        H_TFIM = tfim_hamiltonian(L, tfim_parameters)
        M_TFIM = tfim_magnetisation(L)

        eigen_energies = H_TFIM.eigenenergies()

        E0 = eigen_energies[0]


            
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

            #result = qt.mesolve(H, psi0, tlist, c_ops=col_ops, e_ops=[], args=time_dependant_functions_coeffs)
            result = qt.sesolve(H, psi0, tlist, e_ops=[], args=time_dependant_functions_coeffs)
        
            eigens = []
            eigensTFIM = []
            for i in tqdm(range(len(tlist)), desc = "eigens", leave=False):
                H = H_ising_chain + H_bath_x_field + H_bath_z_field * bath_z_field_zeeman_drive(tlist[i], time_dependant_functions_coeffs) + H_coupling * coupling_drive(tlist[i], time_dependant_functions_coeffs)
                eigens.append(H.eigenstates())
                
                H = H_ising_chain
                eigensTFIM.append(H.eigenstates())


            return result, eigens, eigensTFIM

    

        def run_procedure(error_percentage=0):

            # Setting up the inital states

            bath_fully_polarized_state = qt.tensor([qt.basis(2,0) for i in range(L)])
            bath_fully_polarized_density_matrix = bath_fully_polarized_state * bath_fully_polarized_state.dag()

            random_tfim_state = qt.tensor([qt.rand_ket(2) for i in range(L)])
            neel_state = qt.tensor([qt.basis(2,i%2) for i in range(L)])

            initial_state = qt.tensor([neel_state, bath_fully_polarized_state])
            #initial_state = qt.tensor([random_tfim_state, bath_fully_polarized_state])
            #initial_state = initial_state * initial_state.dag()

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
            whole_process_eigensTFIM = []
            
            # Main loop
            
            for k in tqdm(range(N_cycles), leave = False):
                
                coupling_parameters = [couplings[k], T]
                bath_parameters = [hb_fields[k], Bi, Bf, T]
                
                

                
                
                cycle_result, cycle_eigens, cycle_eigensTFIM = run_cycle(L, tfim_parameters, bath_parameters, coupling_parameters, initial_state, times)     # Runing a single cycle
                cycle_states = cycle_result.states
                cycle_tfim_energy = qt.expect(H_TFIM, cycle_states)
                whole_process_energies.append(cycle_tfim_energy)
                
                whole_process_states.append(cycle_states)
                whole_process_eigens.append(cycle_eigens)
                whole_process_eigensTFIM.append(cycle_eigensTFIM)

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

                    initial_state = qt.tensor([reconstructed_state, bath_fully_polarized_state])

                elif remake_product_state_LSQ: 
                    tfim_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
                    
                    from pulses_lib import reconstruct_state_L1_LSQ
                    reconstructed_state = reconstruct_state_L1_LSQ(tfim_part_of_the_end_cycle_state_density_matrix)

                    initial_state = qt.tensor([reconstructed_state, bath_fully_polarized_state])

                elif pass_directly_with_added_errors:
                    
                    raise Exception("Didn't care to implement this, sorry.")
                    
                    m_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
                    random_state = qt.tensor([qt.rand_ket(2) for i in range(L)])
                    random_state = random_state * random_state.dag()

                    keep_amount = 1-error_percentage

                    tfim_new = np.sqrt(keep_amount) * m_part_of_the_end_cycle_state_density_matrix + np.sqrt(1-keep_amount) * random_state

                    initial_state = qt.tensor([tfim_new, bath_fully_polarized_density_matrix])

                else:
                    
                    raise Exception("Ideal needs mesolve, sorry.")

                    tfim_part_of_the_end_cycle_state_density_matrix = end_cycle_state.ptrace([i for i in range(L)])
                    initial_state = qt.tensor([tfim_part_of_the_end_cycle_state_density_matrix, bath_fully_polarized_density_matrix])

            return whole_process_states, whole_process_eigens, whole_process_energies, whole_process_eigensTFIM
        return run_procedure()
    
    
if __name__ == "__main__": 
        Proc = Procedure(remake_product_state_LSQ=False, N_cycles=2, L = 1)
        
        #t0 = time.time()
        #data1 = Proc.run_with_sesolve()
        #print(time.time() - t0)
        t0 = time.time()
        data2 = Proc.run_with_mesolve()
        print(time.time() - t0)

        eigen = data2[1]

        print(len(eigen))
        print(len(eigen[0]))
        print(len(eigen[0][0]))
        print(len(eigen[0][0][0]))