import copy
from typing import Self, Optional
import numpy as np
import utils
from parameters import Parameters



class Bcells(Parameters):


    def __init__(self, initial_number: Optional[int]=None):
        """Initialize attributes.
        
        All the parameters from Parameters are included. The birth/death rates and
        bcell field arrays are set as well.

        Args:
            initial_number: Initial number of bcells.

        Attributes:
            birth_rate: bcell birth rate
            death_rate: bcell death rate. Specific to each type of bcell.
            initial_number: Initial number of bcells.
            array_names: Names of the bcell field arrays. This is used for modifying
                all bcell field arrays at once (replace_all_arrays, filter_all_arrays)
        """
        super().__init__()
        self.birth_rate = self.bcell_birth_rate
        self.death_rate = None
        self.initial_number = (
            initial_number if initial_number else self.initial_bcell_number
        )
        self.reset_bcell_fields()

        self.array_names = [
            'mutation_state_array',
            'precalculated_dEs',
            'lineage',
            'target_epitope',
            'variant_affinities',
            'activated_time',
            'gc_or_egc_derived'
        ]
    
    def reset_bcell_fields(self) -> None:
        """Initialize bcell field arrays.
        
        Attributes:
            mutation_state_array: np.ndarray (shape=(n_cell, n_res)) indicating 0
                for unmutated residue and 1 for mutated residue.
            precalculated_dEs: np.ndarray (shape=(n_cell, n_res, n_var)) containing
                the change in affinity (dE) for a particular variant given a mutation
                in a residue.
            lineage: np.ndarray (shape=(n_cell)) containing the lineage of the bcell.
            target_epitope: np.ndarray (shape=(n_cell)) containing the targeted epitope.
            variant_affinites: np.ndarray (shape=(n_cell, n_var)) containing the
                binding affinity of the bcell with a particular variant.
            activated_time: np.ndarray (shape=(n_cell)) containing the time the bcell
                was produced.
            gc_or_egc_derived: np.ndarray (shape=(n_cell)) containing whether the bcell
                was derived from the GC or EGC. See utils.DerivedCells for tag values.
        """
        self.mutation_state_array = np.zeros(
            (self.initial_number, self.n_res), dtype=int
        )                                                                                 
        self.precalculated_dEs = np.zeros(                                              
            (self.initial_number, self.n_res, self.n_var)
        )
        self.lineage = np.zeros(self.initial_number, dtype=int)                   
        self.target_epitope = np.zeros(self.initial_number, dtype=int)                              
        self.variant_affinities = np.zeros((self.initial_number, self.n_var))             
        self.activated_time = np.zeros(self.initial_number)                             
        self.gc_or_egc_derived = np.zeros(
            self.initial_number, dtype=int
        ) + utils.DerivedCells.UNSET.value                                        


    def replace_all_arrays(self, idx: np.ndarray) -> None:
        """For all bcell field arrays, keep only bcells indicated by idx.
        
        Args:
            idx: indices indicating which bcells to keep.
        """
        for array_name in self.array_names:
            setattr(self, array_name, getattr(self, array_name)[idx])


    def filter_all_arrays(self, idx: np.ndarray) -> None:
        """For all bcell field arrays, take out cells indicated by idx.
        
        Args:
            idx: indices indicating which bcells to remove.
        """
        other_idx = utils.get_other_idx(np.arange(self.lineage.size), idx)
        for array_name in self.array_names:
            setattr(self, array_name, getattr(self, array_name)[other_idx])


    def add_bcells(self, bcells: Self) -> None:
        """Add bcells to self by appending all the bcell field arrays.
        
        Args:
            bcells: the other Bcell population.
        """
        for array_name in self.array_names:
            new_array = np.concatenate([
                getattr(self, array_name), 
                getattr(bcells, array_name),
            ], axis=0)
            setattr(self, array_name, new_array)


    def tag_gc_or_egc_derived(self, tag_value: int) -> None:
        """Tag all the bcells with tag_value.
        
        Args:
            tag_value: Indicates if bcells are from GC or EGC. See utils.DerivedCells
            for values.
        """
        self.gc_or_egc_derived = np.zeros(
            shape=self.lineage.shape, dtype=int
        ) + tag_value
    

    def get_dE(
        self, 
        idx_new: int, 
        idx: int, 
        ep: int
    ) -> np.ndarray:
        """Get dE affinity changes by sampling from multivariate log-normal.
        
        Args:
            idx_new: End index for bcells
            idx: Start index for bcells
            ep: Epitope

        Returns:
            dE: affinity change values
                np.ndarray (shape=(len(idx_new)-len(idx))*n_res, n_var).
        """
        mu = np.zeros(self.n_var)
        sigma = self.mutation_pdf[1] ** 2 * self.sigma[ep]
        num = (idx_new - idx) * self.n_res
        X = self.mutation_pdf[0] + np.random.multivariate_normal(mu, sigma, num)
        dE = -np.log10(np.exp(1)) * (np.exp(X) - self.mutation_pdf[2])
        return dE

    
    def get_activation(
        self, conc_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """From concentration and affinities, calculate activation signal and activation.
        
        Args:
            conc_array: the effective Ag concentration for the epitope that the bcell
            is targeting. np.ndarray (shape=(n_cells, n_ag)).
            variant_idx: the variant Ag idx that is being used to calculate captured Ag.

        Returns:
            activation_signal: Amount of Ag captured. np.ndarray (shape=(n_cells)).
            activated: Whether bcell is activated. np.ndarray (shape=(n_cells)).
        """
        conc_term = conc_array / self.C0
        aff_term = 10 ** (self.variant_affinities[:, :self.n_ag] - self.E0)
        activation_signal = (conc_term * aff_term) ** self.w2

        if self.w1 > 0:  # Alternative Ag capture model
            num = (self.w1 + 1) * activation_signal
            denom = self.w1 + activation_signal
            activation_signal = num / denom

        activation_signal = self.activation_condense_fn(activation_signal)
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
        """From activation signal, calculate tcell help and birth rate.
        
        Args:
            activation_signal: Amount of Ag captured. np.ndarray (shape=(n_cells)).
            activated: Whether bcell is activated. np.ndarray (shape=(n_cells)).
            tcell: Current tcell amount.
            birth_rate: Bcell birth rate.

        Returns:
            birth_signal: The birth_signal of each bcell, used to calculate which
                bcells create daughter cells later. np.ndarray (shape=(n_cells)).
        """
        activated_fitness = activated * activation_signal
        avg_fitness = activated_fitness[activated_fitness > 0].mean()
        tcell_help = tcell / activated.sum() / avg_fitness * activated_fitness
        birth_signal = birth_rate * (tcell_help / (1 + tcell_help))
        return birth_signal


    def get_seeding_idx(self, conc: np.ndarray) -> np.ndarray:
        """Finds the indices of Bcells that will enter GC.
        
        Args:
            conc: Effective Ag concentration for each epitope.
                np.ndarray (shape=(n_ep, n_ag))
            
        Returns:
            incoming_naive: Indices of Bcells that will enter the GC.
                np.ndarray (shape=(n_cells))
        """
        conc_array = np.zeros(shape=(self.lineage.size, self.n_ag))
        for ep in range(self.n_ep):
            matching_epitope = self.target_epitope == ep
            conc_array += (
                conc[ep] * 
                matching_epitope[:, np.newaxis] * 
                (self.activated_time == 0)[:, np.newaxis]
            )

        activation_signal, activated = self.get_activation(conc_array)
    
        if activated.sum(): # at least one B cell is intrinsically activated
            lambda_ = self.get_birth_signal(
                activation_signal, activated, self.nmax, self.gc_entry_birth_rate
            )
            selected = np.random.uniform(size=activated.shape) < lambda_ * self.dt
            incoming_naive = activated & selected
            
        else:
            incoming_naive = np.array([])
            
        return np.nonzero(incoming_naive)[0]
        

    def get_birth_idx(self, conc: np.ndarray, tcell: float) -> np.ndarray:
        """Finds the indices of Bcells that will undergo birth.
        
        Args:
            conc: Effective Ag concentration for each epitope.
                np.ndarray (shape=(n_ep, n_ag))
            tcell: Current tcell amount.
            
        Returns:
            incoming_naive: Indices of Bcells that will undergo birth.
                np.ndarray (shape=(n_cells))
        """
        conc_array = np.zeros(shape=(self.lineage.size, self.n_ag))
        for ep in range(self.n_ep):
            conc_array += conc[ep] * (self.target_epitope == ep)[:, np.newaxis]

        activation_signal, activated = self.get_activation(conc_array)

        if activated.sum(): # at least one B cell is intrinsically activated
            beta = self.get_birth_signal(
                activation_signal, activated, tcell, self.birth_rate
            )
            beta[np.isnan(beta)] = 0
            selected = np.random.uniform(size=activated.shape) < beta * self.dt
            birth_idx = activated & selected
        
        else:
            birth_idx = np.array([])

        return np.nonzero(birth_idx)[0]


    def get_death_idx(self) -> np.ndarray:
        """Finds the indices of Bcells to die.
        
        Returns:
            death_idx: Indices of Bcells that will die.
        """
        death_idx = np.zeros(self.lineage.shape).astype(bool)
        killed = np.random.uniform(size=self.lineage.size) < self.death_rate * self.dt
        death_idx = np.nonzero(killed)[0]
        return death_idx
    

    def kill(self) -> None:
        """Finds indices of bcells to die and removes them.
        
        If bcell population goes to 0, then the bcell field arrays are
        reset so that they have 1 bcell. This is to avoid errors when indexing
        an empty array.
        """
        death_idx = self.get_death_idx()
        self.filter_all_arrays(death_idx)
        if self.lineage.size == 0:
            self.reset_bcell_fields()


    def get_bcells_from_idx(self, idx: np.ndarray) -> Self:
        """Get a copy of Bcells selected based on idx.
        
        Args:
            idx: indices to select bcells.

        Returns:
            new_bcells: the copies of new Bcells.
        """
        new_bcells = copy.deepcopy(self)

        # Avoid error in replace_all_arrays
        if new_bcells.lineage.size == 0:
            return new_bcells
        
        new_bcells.replace_all_arrays(idx)
        return new_bcells
    
    
    def get_filtered_by_tag(self, tag_value: int) -> Self:
        """Get a copy of Bcells filtered by their tag_value.
        
        Args:
            tag_value: whether to look for GC or EGC-derived cells.

        Returns:
            Copies of the filtered Bcells.
        """
        filtered_idx = np.where(self.gc_or_egc_derived == tag_value)[0]
        return self.get_bcells_from_idx(filtered_idx)
    
    
    def get_seeding_bcells(self, conc: np.ndarray) -> Self:
        """Get a copy of GC-seeding Bcells.

        Args:
            conc: Effective Ag concentration for each epitope.
                np.ndarray (shape=(n_ep, n_ag))

        Returns:
            Copies of the seeding Bcells.
        """
        seeding_idx = self.get_seeding_idx(conc)
        return self.get_bcells_from_idx(seeding_idx)
    

    def get_daughter_bcells(self, conc: np.ndarray, tcell: float) -> Self:
        """Get a copy of the daughter bcells after selection and birth.
        
        Args:
            conc: Effective Ag concentration for each epitope.
                np.ndarray (shape=(n_ep, n_ag))
            tcell: Current tcell amount.

        Returns:
            Copies of the daughter Bcells.
        """
        birth_idx = self.get_birth_idx(conc, tcell)
        return self.get_bcells_from_idx(birth_idx)


    def get_mutated_bcells_from_idx(self, idx: np.ndarray) -> Self:
        """Get copies of bcells based on idx and mutate them.
        
        Args:
            idx: indices of the bcells to mutate.

        Returns:
            Copies of the bcells after mutating.
        """
        mutated_bcells = self.get_bcells_from_idx(idx)
        mutated_residues = np.random.randint(self.n_res, size=idx.size)

        # Find which bcells are mutated already
        original_mutation_states = mutated_bcells.mutation_state_array[
            np.arange(idx.size), mutated_residues
        ]
        nonmutated_idx = np.where(original_mutation_states == 0)[0]
        mutated_idx = np.where(original_mutation_states == 1)[0]
        residues_nonmutated = mutated_residues[nonmutated_idx]
        residues_mutated = mutated_residues[mutated_idx]

        if nonmutated_idx.size + mutated_idx.size != idx.size:
            raise ValueError('Mutation state array may contain nonbinary values.')
        
        # Invert values in mutation_state_array
        mutated_bcells.mutation_state_array[nonmutated_idx, residues_nonmutated] += 1
        mutated_bcells.mutation_state_array[mutated_idx, residues_mutated] -= 1

        # Adjust affinities
        mutated_bcells.variant_affinities[
            nonmutated_idx
        ] += mutated_bcells.precalculated_dEs[nonmutated_idx, residues_nonmutated]

        mutated_bcells.variant_affinities[
            mutated_idx
        ] -= mutated_bcells.precalculated_dEs[mutated_idx, residues_mutated]

        if np.any(mutated_bcells.variant_affinities > self.max_affinity):
            raise ValueError('Affinity impossibly high.')
        
        return mutated_bcells


    def differentiate_bcells(
        self, 
        output_prob: float, 
        output_pc_fraction: float,
        tag_value: int, 
        mutate: bool=True
    ) -> tuple[Self]:
        """Divide bcells into memory, plasma, nonexported_bcells.
        
        Args:
            tag_value: Indicates if bcells are from GC or EGC. See utils.DerivedCells
            output_prob: Probability of a birthed bcell being exported.
            output_pc_fraction: Fraction of exported bcells that become plasma cells.
            mutate: Whether to include mutated cells. Not used in the EGC.

        Returns:
            memory_bcells: Memory bcells that will be exported.
            plasma_bcells: Plasma bcells that will be exported.
            nonexported_bcells. Nonexported bcells. Includes mutated bcells if
                mutate is True.
        
        """
        # Exported indices
        birth_bcells_idx = np.arange(len(self.lineage))
        output_idx = utils.get_sample(birth_bcells_idx, p=output_prob)
        plasma_idx = utils.get_sample(output_idx, p=output_pc_fraction)
        memory_idx = utils.get_other_idx(output_idx, plasma_idx)

        # Nonexported indices
        nonoutput_idx = utils.get_other_idx(birth_bcells_idx, output_idx)
        death_idx = utils.get_sample(nonoutput_idx, p=self.mutation_death_prob)
        nondeath_idx = utils.get_other_idx(nonoutput_idx, death_idx)
        p = min((
            self.mutation_silent_prob * 
            len(nonoutput_idx) / 
            (len(nondeath_idx) + self.epsilon)
        ), 1)
        silent_mutation_idx = utils.get_sample(nondeath_idx, p=p)
        affinity_change_idx = utils.get_other_idx(nondeath_idx, silent_mutation_idx)

        # Make the bcells from indices
        memory_bcells = self.get_bcells_from_idx(memory_idx)
        plasma_bcells = self.get_bcells_from_idx(plasma_idx)
        nonexported_bcells = self.get_bcells_from_idx(silent_mutation_idx)
        
        if mutate:
            mutated_bcells = self.get_mutated_bcells_from_idx(affinity_change_idx)
            nonexported_bcells.add_bcells(mutated_bcells)

        # Tag bcells as GC or EGC-derived
        memory_bcells.tag_gc_or_egc_derived(tag_value)
        plasma_bcells.tag_gc_or_egc_derived(tag_value)
        nonexported_bcells.tag_gc_or_egc_derived(tag_value)

        return memory_bcells, plasma_bcells, nonexported_bcells







