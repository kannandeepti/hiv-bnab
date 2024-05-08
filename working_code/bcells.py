import copy
from typing import Self

import numpy as np
import utils
from parameters import Parameters


class Bcells(Parameters):


    def __init__(self, max_num_bcells: int):

        self.max_num_bcells = max_num_bcells

        # Specific to each type of bcell, must be set
        self.birth_rate = None
        self.death_rate = None

        self.mutation_state_array = np.zeros((self.max_num_bcells, self.num_res)) # (num_bcell, num_res)
        self.precalculated_dEs = np.zeros((self.max_num_bcells, self.num_res)) # (num_bcell, num_res)
        self.lineage = np.zeros(self.max_num_bcells) # (num_bcell), i think 0 indicates empty b cell
        self.target_epitope = np.zeros(self.max_num_bcells)  # (num_bcell)
        self.variant_affinities = np.zeros(self.max_num_bcells, self.n_var)  # (num_bcell, num_var)
        self.activated_time = np.zeros(self.max_num_bcells) # (num_bcell)
        self.num_mut = np.zeros(self.max_num_bcells) # (num_bcell) ???
        self.mutation_state1 = None#???
        self.mutation_state2 = None
        self.unique_clone_index = None

        self.array_names = [
            'mutation_state_array',
            'precalculated_dEs',
            'lineage',
            'target_epitope',
            'variant_affinities',
            'activated_time',
            'num_mut',
            'mutation_state1',
            'mutation_state2',
            'unique_clone_index'
        ]
    

    def replace_all_arrays(self, idx) -> None:
        for array_name in self.array_names:
            setattr(self, array_name, getattr(self, array_name)[idx])


    def add_bcells(self, bcells: Self) -> None:
        for array_name in self.array_names:
            new_array = np.concatenate(
                getattr(self, array_name), 
                getattr(bcells, array_name),
                axis=0
            )
            setattr(self, array_name, new_array)


    def get_sigma(self) -> list[np.ndarray]:
        """Get sigma: covariance matrix per epitope."""
        sigma = [[] for _ in self.n_ep]
        for ep in range(self.n_ep):
            if ep == 0:  # dom
                sigma[ep] = np.array([[1, 0.4, 0.4], [0.4, 1, 0], [0.4, 0, 1]]) #XXX

            else:
                sigma[ep] = np.diag(np.ones(self.n_var))
                for var in range(self.n_var - 1):  # non-WT var
                    # each row corresponds to a different epitope
                    rho = self.rho[ep - 1][var]
                    sigma[ep][0][var + 1] = rho
                    sigma[ep][var + 1][0] = rho
        return sigma
    

    def get_naive_bcells_arr(self) -> np.ndarray:
        """Get number of naive B cells in each fitness class."""
        # Find out number of B cells in each class
        max_classes = np.around(
            np.array([
                self.E1h - self.E0,
                self.E1h - self.dE12 - self.E0, 
                self.E1h - self.dE13 - self.E0
            ]) / self.dE
        ) + 1
        r = np.zeros(self.n_ep)  # slopes of geometric distribution

        for ep in range(self.n_ep):
            if max_classes[ep] > 1:
                func = lambda x: self.num_naive_precursors - (x ** max_classes[ep] - 1) / (x - 1)
                r[ep] = utils.fsolve_mult(func, guess=1.1)
            else:
                r[ep] = self.num_naive_precursors

        # n_ep x 11 array, number of naive B cells in each fitness class
        naive_bcells_arr = np.zeros(shape=(self.n_ep, self.num_class_bins))
        p = [1-self.p2-self.p3, self.p2, self.p3]
        for ep in range(self.n_ep):
            if max_classes[i] > 1:
                naive_bcells_arr[ep, :self.num_class_bins] = p[ep] * r[ep] ** (
                    max_classes[ep] - (np.arange(self.num_class_bins) + 1)
                )
            elif max_classes[ep] == 1:
                naive_bcells_arr[ep, 0] = p[ep] * self.num_naive_precursors

        return naive_bcells_arr
    

    def get_dE(
        self, 
        idx_new: int, 
        idx: int, 
        sigma: list[np.ndarray], 
        ep: int
    ) -> np.ndarray:
        """Get dE: affinity changes."""
        mu = np.zeros(self.n_var)
        sigma = self.mutation_pdf[1] ** 2 * sigma[ep]
        num = (idx_new - idx) * self.n_res
        X = self.mutation_pdf[0] + np.random.multivariate_normal(mu, sigma, num)
        dE = np.log10(np.exp(1)) * (np.exp(X) - self.mutation_pdf[2])
        return dE

    
    def get_activation(self, conc_array, variant_idx):
        """From concentration and affinities, calculate activation signal and activation."""
        conc_term = conc_array / self.C0
        aff_term = 10 ** (self.variant_affinities[:, variant_idx] - self.E0)
        activation_signal = (conc_term * aff_term) ** self.w2

        if self.w1 > 0:  # Alternative Ag capture model
            num = (self.w1 + 1) * activation_signal
            denom = self.w1 + activation_signal
            activation_signal = num / denom

        min_arr = np.minimum(activation_signal, 1)
        activated = min_arr > np.random.uniform(size=activation_signal.shape)
        return activation_signal, activated
    

    def get_birth_signal(
        self, 
        activation_signal, 
        activated, 
        tcell, 
    ) -> np.ndarray:
        """From activation signal, calculate Tcell help and birth rate."""
        activated_fitness = activated * activation_signal
        avg_fitness = activated_fitness[activated_fitness > 0].mean()
        tcell_help = tcell / activated.sum() / avg_fitness * activated_fitness
        birth_rate = self.birth_rate * (tcell_help / (1 + tcell_help))
        return birth_rate


    def get_seeding_idx(self, conc) -> np.ndarray:
        """Finds the logical indices of naive Bcells that will enter GC.
        
        shape should np.ndarray of size self.max_num_bcells containing 0s or 1s
        
        """
        conc_array = np.zeros(shape=self.max_num_bcells)
        for ep in range(self.n_ep):
            matching_epitope = self.target_epitope == (ep + 1)
            conc_array += conc[ep] * matching_epitope * (self.activated_time == 0)

        ## assuming WT at ind 0 is the correct variant to use XXX
        activation_signal, activated = self.get_activation(conc_array, 0)
    
        if activated.sum(): # at least one B cell is intrinsically activated
            lambda_ = self.get_tcell_help(
                activation_signal, activated, self.nmax, self.lambda_max)
            selected = np.random.uniform(size=activated.shape) < lambda_ * self.dt
            incoming_naive = activated & selected
            
        else:
            incoming_naive = np.array([])
            
        return incoming_naive
    

    def get_bcells_from_idx(self, idx) -> Self:
        new_bcells = copy.deepcopy(self)
        new_bcells.replace_all_arrays(idx)
        return new_bcells
    

    def get_seeding_bcells(self, conc) -> Self:
        seeding_idx = self.get_seeding_idx(conc)
        seeding_bcells = self.get_bcells_from_idx(seeding_idx)
        return seeding_bcells
        

    def get_birth_idx(self, conc, tcell) -> np.ndarray:
        """Get indices of Bcells that will undergo birth."""
        conc_array = np.zeros(shape=self.max_num_bcells)
        for ep in range(self.n_ep):
            conc_array += conc[ep] * (self.target_epitope == (ep + 1))

        ## assuming WT at ind 0 is the correct variant to use XXX
        activation_signal, activated = self.get_activation(conc_array, 0)

        if activated.sum(): # at least one B cell is intrinsically activated
            beta = self.get_birth_signal(activation_signal, activated, tcell)
            beta[np.isnan(beta)] = 0
            selected = np.random.uniform(size=activated.shape) < beta * self.dt
            birth_idx = activated & selected
        
        else:
            birth_idx = np.array([])

        return birth_idx
    

    def birth_mutate(self, conc, tcell, mutate: bool=True) -> Self:
        pass  # todo XXX


    def divide_birth(self) -> Self:
        pass  # todo XXX


    

    def divide_birth_into_mem_and_pc(
        self, 
        birth_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if birth_idx.sum():
            plasmas = np.random.uniform(size=birth_idx.shape) < self.prob_plasma
            plasma_idx, memory_idx = birth_idx[plasmas], birth_idx[~plasmas]
        else:
            plasma_idx, memory_idx = np.array([]), np.array([])
        return plasma_idx, memory_idx
    

    def get_death_idx(self) -> np.ndarray:
        death_idx = np.zeros(self.lineage.shape).astype(bool)
        live = self.lineage != 0
        killed = np.random.uniform(size=self.lineage) < self.death_rate * self.dt
        death_idx = live & killed
        return death_idx
    

    def kill(self) -> None:
        death_idx = self.get_death_idx(self.death_rate)
        # todo XXX







