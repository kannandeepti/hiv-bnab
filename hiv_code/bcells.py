r"""
Bcells class
=============

This module defines the `Bcells` class, which defines a B cell population and is
used to model B cell population dynamics, including
mutation, selection, activation, differentiation, and affinity maturation.

The `Bcells` class extends `Parameters` to represent a particular B cell population
(i.e. naive, GC, GC-derived memory, EGC-derived memory, GC-derived plasma, or
EGC-derived plasma cells). Each `Bcells` object contains **B cell field arrays**
of length number of cells in the population, specifying for each cell the
mutation state, target epitope, precursor lineage, GC of origin, time of activation,
and affinity to antigenic variants.

---

### **Core Attributes**
The `Bcells` class maintains **NumPy arrays and sparse matrices** to store B cell properties.

| **Attribute**            | **Type**                 | **Shape** | **Description** |
|--------------------------|-------------------------|-----------|----------------|
| `mutation_state_array`   | `csr_matrix (sparse)`   | `(n_cells, n_residues)` | Mutation states of each residue of each B cell receptor (0 = unmutated, 1 = mutated). |
| `gc_lineage`             | `np.ndarray`            | `(n_cells,)` | Index of GC to which each B cell is assigned (*0, ..., `n_gc`*) |
| `lineage`                | `np.ndarray`            | `(n_cells,)` | Precursor lineage identifier for each B cell (*0, ..., `n_naive_precursors`*)|
| `target_epitope`         | `np.ndarray`            | `(n_cells,)` | Epitope each B cell targets (*0, ..., n_ep*). |
| `variant_affinities`     | `np.ndarray`            | `(n_cells, n_variants)` | Binding affinities (:math:`-log_{10}(M)`)|
| `activated_time`         | `np.ndarray`            | `(n_cells,)` | Last activation time of each B cell. |

---
### Notes:
Sparse matrices (csr_matrix) are used to optimize memory usage for large populations.

"""

import copy
from typing import Self

import numpy as np
import scipy
from scipy.sparse import csr_matrix, lil_matrix

from . import utils
from .parameters import Parameters


class Bcells(Parameters):
    """Class defining a B cell population.

    Attributes:
        mutation_state_array (csr_matrix): Binary matrix of size
            *(n_cells, n_residues)*; 0 = germline, 1 = mutated.
        gc_lineage (np.ndarray[int] of shape (n_cells,)): GC identifier per cell.
        lineage (np.ndarray[int] of shape (n_cells,)): Unique lineage index per cell.
        target_epitope (np.ndarray[int] of shape (n_cells,)): Epitope targeted by each cell.
        memory_reentry_tag (np.ndarray[int] of shape (n_cells,)): 1 if a memory cell re-entered the
            naïve pool, else 0.
        variant_affinities (np.ndarray[float] of shape shape (n_cells, n_variants)): Binding affinity (:math:`-\log_{10} K_d`) to each
            variant.
        activated_time (np.ndarray[float] of shape (n_cells,)): Time each cell
            was activated (i.e. seeded if GC B cell or differentiated if
            plasma/memory B cell).
    """

    def __init__(
        self, updated_params_file: str | None = None, initial_number: int | None = None
    ):
        """
        Initialize a Bcells instance.

        Args:
            updated_params_file (str): Optional JSON file that overrides the default
                parameters inherited from :class:`Parameters`.
            initial_number (int): Initial population size.  If *None*, uses
                ``self.initial_bcell_number`` from the parameter set.

        Notes:
        * ``self.birth_rate`` is set directly from the inherited parameter
          ``bcell_birth_rate``; ``self.death_rate`` is left ``None`` and should
          be assigned by the simulation driver via
          ``Simulation.set_death_rates()``.
        * All per-cell field arrays are created (or re-created) by
          :py:meth:`reset_bcell_fields`.
        * ``self.bcell_field_keys`` lists every attribute that must stay
          row-aligned; helper utilities iterate over this list when slicing or
          concatenating populations.

        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)

        self.birth_rate = self.bcell_birth_rate
        self.death_rate = None  # set in Simulation class
        self.initial_number = (
            initial_number if initial_number else self.initial_bcell_number
        )

        # Initialize all B cell arrays
        self.reset_bcell_fields()

        # Arrays we collectively manipulate in e.g. replace_all_arrays
        self.bcell_field_keys = [
            "mutation_state_array",
            "gc_lineage",
            "lineage",
            "target_epitope",
            "memory_reentry_tag",  # 1 if memory cell that re-enters naive pool, 0 otherwise
            "variant_affinities",
            "activated_time",
        ]

    def reset_bcell_fields(self) -> None:
        """
        Initialize or reset all B cell field arrays.

        Sets:
          - mutation_state_array as a sparse matrix of zeros with shape
            `(initial_number, n_residues)`.
          - gc_lineage, lineage, and target_epitope, activated_time as zero-filled
            arrays of shape `(initial_number,)`.
          - variant_affinities as a zero matrix of shape
            `(initial_number, n_variants)`.

        """
        # Initialize mutation state array as a sparse matrix
        self.mutation_state_array = csr_matrix(
            np.zeros((self.initial_number, self.n_residues), dtype=int)
        )

        # Initialize other B cell field arrays
        self.gc_lineage = np.zeros(self.initial_number, dtype=int)
        self.lineage = np.zeros(self.initial_number, dtype=int)
        self.target_epitope = np.zeros(self.initial_number, dtype=int)
        self.memory_reentry_tag = np.zeros(self.initial_number, dtype=int)
        self.variant_affinities = np.zeros((self.initial_number, self.n_variants))
        self.activated_time = np.zeros(self.initial_number)

    # ------------------------------------
    # Subsetting & Merging
    # ------------------------------------
    def subset_bcell_fields(self, indices: np.ndarray) -> None:
        """
        Replace each B cell field array with its subset defined by the provided indices.

        This function iterates over the keys in `self.Bcell_field_keys` and updates each corresponding
        attribute by selecting only the elements at positions given in `indices`.

        Args:
            indices (np.ndarray[int]): Array of indices to retain in all field arrays.

        """
        for key in self.bcell_field_keys:
            array = getattr(self, key)
            setattr(self, key, array[indices])

    def exclude_bcell_fields(self, indices: np.ndarray) -> None:
        """
        Remove B cells at the specified indices from all field arrays.

        This function computes the complementary indices (i.e. the indices not in `indices`)
        and then calls `subset_bcell_fields` with these complementary indices to update all field arrays.

        Args:
            indices (np.ndarray[int]): Array of indices representing B cells to be removed.
        """
        keep_indices = utils.get_other_idx(np.arange(self.lineage.size), indices)
        self.subset_bcell_fields(keep_indices)

    def add_bcells(self, other: Self) -> None:
        """
        Merge the B cell data from `other` instance into self.

        This method concatenates each field array (listed in `self.Bcell_field_keys`) from the
        current instance with the corresponding field array from the other instance. Sparse arrays
        (csr_matrix) are merged using vertical stacking, while dense NumPy arrays are concatenated
        along axis 0.

        Args:
            other (Self): Another Bcells instance to merge into this instance.

        """
        for key in self.bcell_field_keys:
            current_field = getattr(self, key)
            other_field = getattr(other, key)

            if isinstance(current_field, csr_matrix):
                assert len(current_field.shape) == 2
                # Merge sparse matrices via vertical stacking
                new_field = scipy.sparse.vstack([current_field, other_field])
            elif isinstance(current_field, np.ndarray):
                # Merge dense arrays along the first axis
                new_field = np.concatenate([current_field, other_field], axis=0)
            else:
                raise ValueError(
                    f"Unsupported field type for {key}: {type(current_field)}"
                )

            setattr(self, key, new_field)

    def get_bcells_from_indices(self, indices: np.ndarray) -> Self:
        """
        Return a new BCells instance containing only the cells at the specified indices.

        Note that a deep copy is necessary because the field arrays (e.g., mutation_state_array, gc_lineage,
        lineage, target_epitope, variant_affinities, activated_time) are mutable objects. Without a deep copy,
        changes to the subset would reflect back in the original instance.

        Args:
            indices (np.ndarray): Array of indices indicating which B cells to retain.

        Returns:
            Self: A new BCells instance containing only the selected cells.

        """
        new_bcells = copy.deepcopy(self)

        # Avoid error in replace_all_arrays
        if new_bcells.lineage.size == 0:
            return new_bcells

        new_bcells.subset_bcell_fields(indices)
        return new_bcells

    # ------------------------------------
    # Set attributes
    # ------------------------------------
    def set_activation_time(self, current_time: float, shift: bool = True) -> None:
        """
        Set the activation time for all B cells.

        This method fills the `activated_time` array with the provided `current_time`.
        If `shift` is True, a temporal offset of 0.5 * dt is added to improve numerical alignment.

        Args:
            current_time (float): The current simulation time.
            shift (bool, optional): Whether to add a 0.5 * dt offset. Defaults to True.
        """
        offset = 0.5 * self.dt if shift else 0
        self.activated_time.fill(current_time + offset)

    def set_memory_reentry_tag(self) -> None:
        """Tag all B cells in this population as memory cells that are re-entering naive pool."""
        self.memory_reentry_tag = np.ones(self.lineage.size, dtype=int)

    # ------------------------------------
    # Positive Selection
    # ------------------------------------
    def get_activation_signal(
        self, conc_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute activation signal for each B cell j targeting epitope i,
        which is a proxy for the amount of antigen internalized:

        ..math::
            A_j = ((C_i / C_0) * 10^{min(E_j, E_{sat}) - E_0})^{K}

        Each cell is then probabilistically activated:

        ..math::
            P_{activated} = min(A_j, 1)

        Args:
            conc_array (np.ndarray of shape (n_cells,)):
                Concentration of epitope targeted by each B cell

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - activation_signal: Array of activation signals (shape: (n_cells,)).
                - activated_mask: Boolean array indicating activation status for each cell (shape: (n_cells,)).
        """

        # Compute concentration term by normalizing with Ag_norm (C0)
        conc_term = conc_array / self.C0

        # Compute affinity term:
        # - Clip log10 affinities (for the targeted variants) to a maximum (E_sat)
        energies = np.clip(self.variant_affinities[:, 0], None, self.Esat)
        aff_term = np.power(10, energies - self.E0)
        activation_signal = np.power(conc_term * aff_term, self.stringency)

        # Determine activation status:
        # For each cell, generate a random number uniformly in [0, 1). If the cell's activation signal
        # (clipped between 0 and 1) is greater than the random number, the cell is activated.
        activated_mask = np.random.uniform(size=activation_signal.size) < np.clip(
            activation_signal, 0, 1
        )

        return activation_signal, activated_mask

    def get_birth_signal(
        self,
        activation_signal: np.ndarray,
        activated_mask: np.ndarray,
        tcell: float,
        birth_rate: float,
    ) -> np.ndarray:
        """
        Calculate the positive selection (birth) rate for each B cell.

        ..math::
            \beta_j = \beta_0 * \frac{(A_j/\langle A \rangle) * tcell/N_{activated}}{1 + (A_j/\langle A \rangle) * tcell/N_{activated}}

        Args:
            activation_signal (np.ndarray of shape (n_cells,)):
                Activation signal for each B cell
            activated_mask (np.ndarray of shape (n_cells,)):
                Boolean array indicating activation status for each cell
            tcell (float): Number of T cells available for help
            birth_rate (float): Baseline division rate for B cells

        Returns:
            birth_signal (np.ndarray of shape (n_cells,)):
                Birth rate for each B cell
        """

        # Validate input shapes
        if activation_signal.shape != activated_mask.shape:
            raise ValueError(
                "Shapes of `activation_signal` and `activated_mask` do not match."
            )

        # Calculate antigen captured by activated B cells
        antigen_captured = activation_signal * activated_mask

        # Handle edge cases where no cells are activated
        if antigen_captured.sum() == 0 or activated_mask.sum() == 0:
            return np.zeros_like(activation_signal)

        # Compute average antigen captured by activated cells
        avg_antigen_captured = antigen_captured[antigen_captured > 0].mean()

        # numerator
        tcell_help = (antigen_captured / avg_antigen_captured) * (
            tcell / activated_mask.sum()
        )
        birth_signal = birth_rate * tcell_help / (1 + tcell_help)
        return birth_signal

    # ------------------------------------
    # Seeding & Division
    # ------------------------------------
    def get_seeding_indices(self, conc: np.ndarray, tcell: float) -> np.ndarray:
        """
        Select indices of B cells eligible for germinal center (GC) entry in two steps

        1. Compute activation signals for each B cell using the antigen concentration and affinity.
        2. Calculate an entry probability (birth signal) based on the activation signal and available T-cell help.

        Args:
            conc (np.ndarray of shape (n_ep,)):
                Concentration of each epitope displayed on FDCs after epitope masking
            tcell (float): Number of T cells available for help

        Returns:
            np.ndarray: Indices of B cells selected for GC seeding
        """
        # Initialize the concentration array for each B cell
        conc_array = np.zeros(shape=(self.lineage.size,))
        for ep in range(self.n_ep):
            matching_epitope = self.target_epitope == ep
            conc_array += conc[ep] * matching_epitope

        # Calculate activation signals and determine activated cells
        activation_signal, activated = self.get_activation_signal(conc_array)

        # Determine which B cells will enter the GC (at least one B cell enters)
        if activated.any():
            # Compute birth signals for activated cells
            lambda_ = self.get_birth_signal(
                activation_signal, activated, tcell, self.gc_entry_birth_rate
            )
            # Stochastically select B cells based on birth signals
            selected = np.random.uniform(size=activated.shape) < lambda_ * self.dt
            incoming_naive = activated & selected
        else:
            incoming_naive = np.array([])
        # Return indices of selected B cells
        return np.nonzero(incoming_naive)[0]

    def get_birth_indices(self, conc: np.ndarray, tcell: float) -> np.ndarray:
        """
        Select indices of positively selected B cells for division.
        Same as `get_seeding_indices` but with the GC B cell birth rate instead of the GC entry birth rate.

        Args:
            conc (np.ndarray of shape (n_ep,)):
                Concentration of each epitope displayed on FDCs after epitope masking
            tcell (float): Number of T cells available for help

        Returns:
            np.ndarray: Indices of B cells selected for GC seeding
        """
        # Initialize the concentration array for each B cell
        conc_array = np.zeros(shape=(self.lineage.size,))
        for ep in range(self.n_ep):
            # conc[ep] is a float, target_epitope is (nbcells,)
            conc_array += conc[ep] * (self.target_epitope == ep)

        # Calculate activation signals and determine activated cells
        activation_signal, activated = self.get_activation_signal(conc_array)

        if activated.sum():  # at least one B cell is intrinsically activated
            # Compute birth signals for activated cells
            beta = self.get_birth_signal(
                activation_signal, activated, tcell, self.birth_rate
            )
            # Handle potential NaN values in birth signals
            beta[np.isnan(beta)] = 0
            # Stochastically select cells for birth based on probabilities
            selected = np.random.uniform(size=activated.shape) < beta * self.dt
            birth_idx = activated & selected
        else:
            birth_idx = np.array([])

        # Return indices of selected B cells
        return np.nonzero(birth_idx)[0]

    def get_seeding_bcells(self, antigen_conc: np.ndarray, tcell: float) -> Self:
        """
        Return a new BCells instance containing only cells eligible for GC entry.

        Args:
            antigen_conc (np.ndarray of shape (n_ep,)):
                Concentration of each epitope displayed on FDCs after epitope masking
            tcell (float): Number of T cells available for help

        Returns:
            Self: A new BCells instance with only GC-seeding cells.
        """
        seeding_idx = self.get_seeding_indices(antigen_conc, tcell)
        return self.get_bcells_from_indices(seeding_idx)

    def get_daughter_bcells(self, antigen_conc: np.ndarray, tcell: float) -> Self:
        """
        Return a new BCells instance containing only the daughter (dividing) B cells.

        Args:
            antigen_conc (np.ndarray of shape (n_ep,)):
                Concentration of each epitope displayed on FDCs after epitope masking
            tcell (float): Number of T cells available for help

        Returns:
            Self: A new BCells instance containing only the daughter (dividing) B cells
        """
        # Identify indices of daughter B cells
        dividing_indices = self.get_birth_indices(antigen_conc, tcell)

        # Create and return a copy containing only daughter B cells
        return self.get_bcells_from_indices(dividing_indices)

    # ------------------------------------
    # Removal / Kill
    # ------------------------------------
    def get_death_indices(self) -> np.ndarray:
        """
        Compute indices of B cells that will undergo apoptosis during the current time step.
        Each B cell dies with probability equal to (death_rate × dt).

        Returns:
            np.ndarray: Indices of B cells to be removed
        """
        death_idx = np.zeros(self.lineage.shape).astype(bool)
        killed = np.random.uniform(size=self.lineage.size) < self.death_rate * self.dt
        death_idx = np.nonzero(killed)[0]
        return death_idx

    def kill(self) -> None:
        """
        Remove B cells slated for apoptosis during the current time step.
        """
        # Step 1: Identify cells for death.
        death_idx = self.get_death_indices()
        # Step 2: Exclude the dead cells from all field arrays.
        self.exclude_bcell_fields(death_idx)
        # Step 3: If the population is empty, reset the fields.
        if self.lineage.size == 0:
            self.reset_bcell_fields()

    # ------------------------------------
    # Mutation
    # ------------------------------------
    def get_dE(self, n_cells: int, ep: int) -> np.ndarray:
        """
        Pre-compute the possible affinity change (ΔE) that `n_cells` B cells can
        experience upon mutation to any of its residues.

        This method samples ΔE values from a log-normal distribution. It assumes that
        the underlying normal distribution has zero mean and a covariance given by:

            σ = (mut_PDF[1])² * epitope_covariance_matrix[ep]

        The sampled values X are transformed into affinity changes using:

            ΔE = -log10(e) * (exp(X) - mut_PDF[2])

        Args:
            n_cells (int): Number of B cells to mutate.
            ep (int): Index of the target epitope.

        Returns:
            dE (np.ndarray of shape (n_cells * n_residues, n_variants)):
                Affinity changes upon mutation to each B cell residue.
        """
        mu = np.zeros(self.n_variants)
        sigma = self.mutation_pdf[1] ** 2 * self.epitope_covariance_matrix[ep]
        num_samples = (n_cells) * self.n_residues

        # Sample from the underlying normal distribution and add the specified offset.
        X = self.mutation_pdf[0] + np.random.multivariate_normal(mu, sigma, num_samples)
        # Transform the samples to log-scale affinity changes.
        dE = -np.log10(np.exp(1)) * (np.exp(X) - self.mutation_pdf[2])
        return dE

    def mutate_bcells_by_indices(
        self, indices: np.ndarray, precalculated_dEs: np.ndarray
    ) -> Self:
        """
        Return a new BCells instance with mutations applied to the cells at the specified indices.

        For each cell selected by `indices`, a random residue is chosen for mutation. If the cell
        is unmutated at that residue (state 0), its mutation state is flipped to 1 and its binding
        affinity is increased by the corresponding value from `precalculated_dEs`. Conversely, if the
        cell is already mutated (state 1), its state is reverted to 0 and its affinity is decreased.

        Args:
            indices (np.ndarray of shape (n_cells,)): Indices of B cells to mutate
            precalculated_dEs (np.ndarray of shape (n_gc, n_cell, n_residues, n_variants)): Precomputed affinity changes (ΔE) for mutations

        Returns:
            Self: A new BCells instance with the selected cells mutated.
        """
        # Create a deep copy of the selected cells to avoid modifying the original instance.
        mutated_bcells = self.get_bcells_from_indices(indices)

        # Randomly choose a residue for each selected cell.
        chosen_residues = np.random.randint(self.n_residues, size=indices.size)

        # Retrieve the mutation states for the chosen residues.
        if indices.size == 0:
            original_mutation_states = np.squeeze(
                mutated_bcells.mutation_state_array[
                    np.arange(indices.size), chosen_residues
                ].toarray()
            )
        else:
            original_mutation_states = np.array(
                mutated_bcells.mutation_state_array[
                    np.arange(indices.size), chosen_residues
                ]
            ).squeeze()

        original_mutation_states = np.atleast_1d(original_mutation_states)

        # Determine which cells are currently unmutated (state 0) vs. mutated (state 1).
        nonmutated_cells = np.where(original_mutation_states == 0)[0]
        already_mutated_cells = np.where(original_mutation_states == 1)[0]

        # Sanity check: Ensure all selected cells have a binary mutation state.
        if nonmutated_cells.size + already_mutated_cells.size != indices.size:
            raise ValueError("Mutation state array contains non-binary values.")

        # Get residue indices for non-mutated and already mutated cells.
        residues_nonmutated = chosen_residues[nonmutated_cells]
        residues_mutated = chosen_residues[already_mutated_cells]

        # Get lineage and GC lineage information for mutation adjustment.
        lineage_nonmutated = mutated_bcells.lineage[nonmutated_cells]
        lineage_mutated = mutated_bcells.lineage[already_mutated_cells]

        gc_lineage_nonmutated = mutated_bcells.gc_lineage[nonmutated_cells]
        gc_lineage_mutated = mutated_bcells.gc_lineage[already_mutated_cells]

        # Flip mutation state: unmutated cells become mutated and vice versa.
        mutated_bcells.mutation_state_array[nonmutated_cells, residues_nonmutated] = 1
        mutated_bcells.mutation_state_array[already_mutated_cells, residues_mutated] = 0

        # Adjust affinities:
        # - Increase for newly mutated cells.
        # - Decrease for cells that are reverting.
        mutated_bcells.variant_affinities[nonmutated_cells] += precalculated_dEs[
            gc_lineage_nonmutated, lineage_nonmutated, residues_nonmutated
        ]

        mutated_bcells.variant_affinities[already_mutated_cells] -= precalculated_dEs[
            gc_lineage_mutated, lineage_mutated, residues_mutated
        ]

        # Ensure that no affinity exceeds the maximum allowable value.
        if np.any(mutated_bcells.variant_affinities > self.max_affinity):
            raise ValueError("Affinity impossibly high.")

        return mutated_bcells

    def differentiate_bcells(
        self,
        export_prob: float,
        PC_frac: float,
        precalculated_dEs: np.ndarray,
        mutate: bool = True,
    ) -> tuple[Self, Self, Self]:
        """
        Partition B cells into memory, plasma, and non-exported groups.

        The process is as follows:
        1. **Export Selection:**
            - Randomly select cells for export with probability `output_prob`.
            - From exported cells, select a fraction (`PC_frac`) to differentiate into plasma cells;
            the remaining exported cells become memory cells.
        2. **Non-Export Handling:**
            - The cells that are not exported (nonexported) are further culled by a death probability
            (p_lethal). The survivors are divided into two groups:
                a. Cells that undergo silent mutations.
                b. Cells that are eligible for affinity-altering mutations.
        3. **Mutation (if enabled):**
            - If `mutate` is True, apply affinity changes to the cells in group (b) using `precalculated_dEs`.
        4. **Merge Non-Exported:**
            - Combine the silently mutated cells with the affinity-mutated cells to form the final non-exported group.

        Args:
            export_prob (float): Probability that a newly birthed B cell is exported.
            PC_frac (float): Fraction of exported cells that become plasma cells.
            precalculated_dEs (np.ndarray of shape (n_gc, n_cell, n_residues, n_variants)): Precomputed affinity changes (ΔE) for mutations
            mutate (bool, optional): Whether to apply affinity-altering mutations to non-exported cells.
                                    Defaults to True.

        Returns:
            tuple[Self, Self, Self]: A tuple containing three BCells instances:
                - Memory cells (exported but not plasma).
                - Plasma cells (exported and differentiating into plasma cells).
                - Non-exported cells (remaining cells, including mutated ones if mutate=True).
        """
        # Step 1: Sample exported B cells
        birth_bcells_idx = np.arange(len(self.lineage))
        export_idx = utils.get_sample(birth_bcells_idx, p=export_prob)
        plasma_idx = utils.get_sample(export_idx, p=PC_frac)
        memory_idx = utils.get_other_idx(export_idx, plasma_idx)

        # Step 2: Sample non-exported B cells
        nonexport_idx = utils.get_other_idx(birth_bcells_idx, export_idx)
        death_idx = utils.get_sample(nonexport_idx, p=self.mutation_death_prob)
        survivors_idx = utils.get_other_idx(nonexport_idx, death_idx)

        # Calculate probability for silent mutations
        silent_prob = min(
            (
                self.mutation_silent_prob
                * len(nonexport_idx)
                / (len(survivors_idx) + self.epsilon)
            ),
            1,
        )
        silent_mutation_idx = utils.get_sample(survivors_idx, p=silent_prob)
        affinity_change_idx = utils.get_other_idx(survivors_idx, silent_mutation_idx)

        # Step 3: Retrieve B cell populations
        memory_bcells = self.get_bcells_from_indices(memory_idx)
        plasma_bcells = self.get_bcells_from_indices(plasma_idx)
        nonexported_bcells = self.get_bcells_from_indices(silent_mutation_idx)

        # Step 4: Mutate affinity-changing B cells if applicable
        if mutate:
            mutated_bcells = self.mutate_bcells_by_indices(
                affinity_change_idx, precalculated_dEs
            )
        else:
            mutated_bcells = self.get_bcells_from_indices(affinity_change_idx)

        # Combine mutated B cells with the non-exported group
        nonexported_bcells.add_bcells(mutated_bcells)

        return memory_bcells, plasma_bcells, nonexported_bcells

    # ------------------------------------
    # Analysis
    # ------------------------------------
    def get_num_above_aff(self) -> np.ndarray:
        """
        Get the number of B cells with binding affinities exceeding predefined thresholds.

        For each epitope and for each threshold in `self.history_affinity_thresholds`, this method computes
        the number of B cells (separately for each antigen variant) whose binding affinities
        (from self.variant_affinities) exceed the threshold.

        Returns:
            num_above_aff: np.ndarray (shape=(n_var, n_ep, n_thresholds))

        """
        # Initialize output array: counts[variant, epitope, threshold]
        counts = np.zeros(
            (self.n_variants, self.n_ep, len(self.history_affinity_thresholds))
        )

        # Loop over each threshold and each epitope.
        for threshold_idx, affinity_threshold in enumerate(
            self.history_affinity_thresholds
        ):
            for ep in range(self.n_ep):
                # Select cells targeting the current epitope.
                cell_affinities = self.variant_affinities[self.target_epitope == ep]
                # Count, for each variant, the number of cells with affinity above the threshold.
                n_cells = (cell_affinities > affinity_threshold).sum(axis=0)
                counts[:, ep, threshold_idx] = n_cells

        return counts

    def get_num_in_aff_bins(self) -> np.ndarray:
        """Get the number of B cells with affinities in different affinity bins.

        Affinity bins are specified in `self.affinity_bins`.

        Returns:
            num_in_aff: np.ndarray (shape=(n_var, n_ep, n_affinities))
        """
        num_in_aff = np.zeros((self.n_variants, self.n_ep, len(self.affinity_bins)))
        for aff_idx, affinity_bin in enumerate(self.affinity_bins):
            for ep in range(self.n_ep):
                variant_affinities = self.variant_affinities[self.target_epitope == ep]
                if aff_idx == 0:  # first bin
                    affinities_in_range = variant_affinities <= affinity_bin
                else:
                    affinity_prev_bin = self.affinity_bins[aff_idx - 1]
                    affinities_in_range = (variant_affinities > affinity_prev_bin) & (
                        variant_affinities <= affinity_bin
                    )
                n_cells = (affinities_in_range).sum(axis=0)
                num_in_aff[:, ep, aff_idx] = n_cells
        return num_in_aff

    def get_total_num(self) -> np.ndarray:
        """Get the total number of B cells in this population.
        Last dimension of returned array indicates whether the B cell is naive (0) or was a memory cell that re-entered (1)

        Returns:
            total_num : np.ndarray (shape=(n_var, n_ep, 2))
        """
        total_num = np.zeros((self.n_variants, self.n_ep, 2))
        for ep in range(self.n_ep):
            for tag in [0, 1]:
                cells_per_ep = (self.target_epitope == ep) & (
                    self.memory_reentry_tag == tag
                )
                n_cells = cells_per_ep.sum(axis=0)
                total_num[:, ep, tag] = n_cells
        return total_num
