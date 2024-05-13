import copy
from typing import Self
import numpy as np
import utils
from parameters import Parameters



class Bcells(Parameters):


    def __init__(self):
        super().__init__()

        self.birth_rate = self.bcell_birth_rate
        self.death_rate = None  # Specific to each type of bcell, must be set

        self.mutation_state_array = np.zeros(
            (self.initial_bcell_number, self.n_res), dtype=int
        )                                                                                           # (num_bcell, n_res)
        self.precalculated_dEs = np.zeros(
            (self.initial_bcell_number, self.n_res, self.n_var)
        )                                                                                           # (num_bcell, n_res, n_var)
        self.lineage = np.zeros(self.initial_bcell_number)                                          # (num_bcell), i think 0 indicates empty b cell
        self.target_epitope = np.zeros(self.initial_bcell_number)                                   # (num_bcell)
        self.variant_affinities = np.zeros((self.initial_bcell_number, self.n_var))                   # (num_bcell, num_var)
        self.activated_time = np.zeros(self.initial_bcell_number)                                   # (num_bcell)
        self.num_mut = np.zeros(self.initial_bcell_number)                                          # (num_bcell) ???
        self.gc_or_egc_derived = np.zeros(
            self.initial_bcell_number, dtype=int
        ) + utils.DerivedCells.UNSET.value                                                          # (num_bcell), 0=gc,, 1=egc, unset=-1
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
            'gc_or_egc_derived'
            'mutation_state1',
            'mutation_state2',
            'unique_clone_index'
        ]
    

    def replace_all_arrays(self, idx: np.ndarray) -> None:
        for array_name in self.array_names:
            setattr(self, array_name, getattr(self, array_name)[idx])


    def filter_all_arrays(self, idx: np.ndarray) -> None:
        other_idx = utils.get_other_idx(np.arange(self.lineage.size), idx)
        for array_name in self.array_names:
            setattr(self, array_name, getattr(self, array_name)[other_idx])


    def add_bcells(self, bcells: Self) -> None:
        for array_name in self.array_names:
            new_array = np.concatenate(
                getattr(self, array_name), 
                getattr(bcells, array_name),
                axis=0
            )
            setattr(self, array_name, new_array)


    def tag_gc_or_egc_derived(self, pop_value: int) -> None:
        self.gc_or_egc_derived = np.zeros(
            shape=self.lineage.shape, dtype=int
        ) + pop_value
    

    def get_naive_bcells_arr(self) -> np.ndarray:
        """Get number of naive B cells in each fitness class.""" #XXX
        # Find out number of B cells in each class
        max_classes = np.around(
            np.array([
                self.E1h - self.E0,
                self.E1h - self.dE12 - self.E0, 
                self.E1h - self.dE13 - self.E0
            ]) / self.class_size
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
            if max_classes[ep] > 1:
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
        ep: int
    ) -> np.ndarray:
        """Get dE: affinity changes."""
        mu = np.zeros(self.n_var)
        sigma = self.mutation_pdf[1] ** 2 * self.sigma[ep]
        num = (idx_new - idx) * self.n_res
        X = self.mutation_pdf[0] + np.random.multivariate_normal(mu, sigma, num)
        dE = np.log10(np.exp(1)) * (np.exp(X) - self.mutation_pdf[2])
        return dE

    
    def get_activation(self, conc_array: np.ndarray, variant_idx: int) -> tuple[np.ndarray, np.ndarray]:
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
        activation_signal: np.ndarray, 
        activated: np.ndarray, 
        tcell: float,
        birth_rate: float
    ) -> np.ndarray:
        """From activation signal, calculate Tcell help and birth rate."""
        activated_fitness = activated * activation_signal
        avg_fitness = activated_fitness[activated_fitness > 0].mean()
        tcell_help = tcell / activated.sum() / avg_fitness * activated_fitness
        birth_signal = birth_rate * (tcell_help / (1 + tcell_help))
        return birth_signal


    def get_seeding_idx(self, conc: np.ndarray) -> np.ndarray:
        """Finds the logical indices of naive Bcells that will enter GC.
        
        shape should np.ndarray of size num_bcells containing 0s or 1s
        
        """
        conc_array = np.zeros(shape=self.lineage.shape)
        for ep in range(self.n_ep):
            matching_epitope = self.target_epitope == (ep + 1)
            conc_array += conc[ep] * matching_epitope * (self.activated_time == 0)

        ## assuming WT at ind 0 is the correct variant to use XXX
        activation_signal, activated = self.get_activation(conc_array, 0)
    
        if activated.sum(): # at least one B cell is intrinsically activated
            lambda_ = self.get_birth_signal(
                activation_signal, activated, self.nmax, self.gc_entry_birth_rate
            )
            selected = np.random.uniform(size=activated.shape) < lambda_ * self.dt
            incoming_naive = activated & selected
            
        else:
            incoming_naive = np.array([])
            
        return incoming_naive
    

    def get_bcells_from_idx(self, idx) -> Self:
        new_bcells = copy.deepcopy(self)
        new_bcells.replace_all_arrays(idx)
        return new_bcells
    

    def get_seeding_bcells(self, conc: np.ndarray) -> Self:
        seeding_idx = self.get_seeding_idx(conc)
        seeding_bcells = self.get_bcells_from_idx(seeding_idx)
        return seeding_bcells
        

    def get_birth_idx(self, conc: np.ndarray, tcell: float) -> np.ndarray:
        """Get indices of Bcells that will undergo birth."""
        conc_array = np.zeros(shape=self.lineage.shape)
        for ep in range(self.n_ep):
            conc_array += conc[ep] * (self.target_epitope == (ep + 1))

        ## assuming WT at ind 0 is the correct variant to use XXX
        activation_signal, activated = self.get_activation(conc_array, 0)

        if activated.sum(): # at least one B cell is intrinsically activated
            beta = self.get_birth_signal(
                activation_signal, activated, tcell, self.birth_rate
            )
            beta[np.isnan(beta)] = 0
            selected = np.random.uniform(size=activated.shape) < beta * self.dt
            birth_idx = activated & selected
        
        else:
            birth_idx = np.array([])

        return birth_idx
    

    def get_daughter_bcells(self, conc: np.ndarray, tcell: float) -> Self:
        birth_idx = self.get_birth_idx(conc, tcell)
        daughter_bcells = self.get_bcells_from_idx(birth_idx)
        return daughter_bcells


    def get_mutated_bcells_from_idx(self, idx: np.ndarray) -> Self:
        mutated_bcells = self.get_bcells_from_idx(idx)
        mutated_residues = np.random.randint(self.n_res, size=idx.size)

        # Find which bcells are mutated already
        original_mutation_states = mutated_bcells.mutation_state_array[
            np.arange(idx.size), mutated_residues
        ]
        nonmutated_idx = np.where(original_mutation_states == 0)
        mutated_idx = np.where(original_mutation_states == 1)
        if nonmutated_idx[0].size + mutated_idx[0].size != idx.size:
            raise ValueError('Mutation state array may contain nonbinary values.')
        
        # Invert values in mutation_state_array
        mutated_bcells.mutation_state_array[nonmutated_idx, mutated_residues] += 1
        mutated_bcells.mutation_state_array[mutated_idx, mutated_residues] -= 1

        # Adjust affinities
        mutated_bcells.variant_affinities += mutated_bcells.precalculated_dEs[
            nonmutated_idx, mutated_residues, :
        ]
        mutated_bcells.variant_affinities -= mutated_bcells.precalculated_dEs[
            mutated_idx, mutated_residues, :
        ]

        if utils.any(mutated_bcells.variant_affinities > self.max_affinity):
            raise ValueError('Affinity impossibly high.')
        
        return mutated_bcells


    def divide_bcells(self, pop_value: int, mutate: bool=True) -> tuple[Self]:
        """Divide bcells into memory, plasma, nonexported_bcells."""
        # Exported indices
        birth_bcells_idx = np.arange(len(self.lineage))
        output_idx = utils.get_sample(birth_bcells_idx, p=self.output_prob)
        plasma_idx = utils.get_sample(output_idx, p=self.output_pc_fraction)
        memory_idx = utils.get_other_idx(output_idx, plasma_idx)

        # Nonexported indices
        nonoutput_idx = utils.get_other_idx(birth_bcells_idx, output_idx)
        death_idx = utils.get_sample(nonoutput_idx, p=self.mutation_death_prob)
        nondeath_idx = utils.get_other_idx(nonoutput_idx, death_idx)
        silent_mutation_idx = utils.get_sample(
            nondeath_idx, 
            p=self.mutation_silent_prob * len(nonoutput_idx) / len(nondeath_idx)
        )
        affinity_change_idx = utils.get_other_idx(nondeath_idx, silent_mutation_idx)

        # Make the bcells from indices
        memory_bcells = self.get_bcells_from_idx(memory_idx)
        plasma_bcells = self.get_bcells_from_idx(plasma_idx)
        nonmutated_bcells = self.get_bcells_from_idx(silent_mutation_idx)
        
        if mutate:
            mutated_bcells = self.get_mutated_bcells_from_idx(affinity_change_idx)
            nonmutated_bcells.add_bcells(mutated_bcells)

        # Tag bcells as GC or EGC-derived
        memory_bcells.tag_gc_or_egc_derived(pop_value)
        plasma_bcells.tag_gc_or_egc_derived(pop_value)
        nonmutated_bcells.tag_gc_or_egc_derived(pop_value)

        return memory_bcells, plasma_bcells, nonmutated_bcells
    

    def get_death_idx(self) -> np.ndarray:
        death_idx = np.zeros(self.lineage.shape).astype(bool)
        live = self.lineage != 0
        killed = np.random.uniform(size=self.lineage) < self.death_rate * self.dt
        death_idx = live & killed
        return death_idx
    

    def kill(self) -> None:
        death_idx = self.get_death_idx(self.death_rate)
        self.filter_all_arrays(death_idx)







