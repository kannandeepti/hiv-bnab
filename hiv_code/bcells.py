"""
### Germinal Center (GC) B Cells Module
=======================================

This module defines the `Bcells` class, a core component of the **germinal center (GC) 
simulation framework**. It models B cell **population dynamics, mutation, selection, 
activation, differentiation, and affinity maturation** in response to antigenic selection.

---

### **Overview**
The `Bcells` class extends `Parameters` to manage **B cell field arrays** and track 
evolutionary changes during the germinal center reaction. It integrates with other modules 
such as `utils` and `parameters` to provide a **realistic and flexible representation of 
B cell affinity maturation**.

---

### **Key Features**
- **B Cell Population Management:** 
  - Tracks birth, death, and differentiation processes.
- **Affinity Mutation & Selection:** 
  - Models silent mutations, affinity-affecting mutations, and antigen selection pressure.
- **Germinal Center Seeding & Expansion:** 
  - Determines which naive B cells enter GCs.
- **T-Cell Help Integration:** 
  - Simulates interactions between B cells and T follicular helper (Tfh) cells.
- **Efficient Memory Handling:** 
  - Uses sparse matrices (`csr_matrix`) to store mutation states and optimize computation.
- **Probability-Based Differentiation:** 
  - Stochastically classifies B cells into **memory, plasma, or non-exported populations**.
- **Tracking and Filtering:** 
  - Provides utilities for selecting and modifying subsets of B cells.

---

### **Usage**
To instantiate a `Bcells` object, specify an initial population size or load parameters from a file. 
The class provides multiple methods to manipulate and evolve B cell populations dynamically.

---

#### **Example Usage**
```python
from bcells import Bcells

# Initialize B cell population with default parameters
bcell_population = Bcells(initial_number=1000)

# Reset B cell fields to their initial states
bcell_population.reset_bcell_fields()

# Add new B cells from another instance
additional_bcells = Bcells(initial_number=500)
bcell_population.add_bcells(additional_bcells)

# Differentiate B cells into memory, plasma, and non-exported cells
memory_bcells, plasma_bcells, nonexported_bcells = bcell_population.differentiate_bcells(
    output_prob=0.3,
    output_pc_fraction=0.5,
    precalculated_dEs=precomputed_affinity_changes
)

# Retrieve B cells with high affinity for further analysis
high_affinity_bcells = bcell_population.count_high_affinity_cells()

---

### **Integration with Germinal Center Simulation**
The `Bcells` class is **designed to integrate seamlessly** with the **germinal center 
simulation pipeline**. It interacts with:
- **Antigen modeling**: Determines how B cells bind and respond to antigenic stimuli.
- **Affinity maturation processes**: Simulates evolutionary changes in B cell populations.
- **Immune response simulations**: Tracks memory and plasma cell differentiation over time.

**Methods like** `get_seeding_bcells()`, `get_daughter_bcells()`, and `differentiate_bcells()` 
allow researchers to track B cell **evolution at different stages** of GC selection.

---

### **Key Components**
| **Component**           | **Description** |
|-------------------------|----------------|
| `reset_bcell_fields()`  | Initializes or resets all B cell field arrays. |
| `get_bcells_from_idx()` | Extracts a subset of B cells based on indices. |
| `get_seeding_bcells()`  | Identifies naive B cells eligible for GC entry. |
| `get_daughter_bcells()` | Selects B cells that undergo division and mutation. |
| `differentiate_bcells()` | Classifies B cells into **memory, plasma, or non-exported**. |
| `count_high_affinity_cells()`   | Counts B cells exceeding affinity thresholds. |

---

### **Expected Data Structures**
The `Bcells` class maintains **NumPy arrays and sparse matrices** to store B cell properties.

| **Attribute**            | **Type**                 | **Shape** | **Description** |
|--------------------------|-------------------------|-----------|----------------|
| `mutation_state_array`   | `csr_matrix (sparse)`   | `(n_cells, n_residues)` | Tracks mutation states (0 = unmutated, 1 = mutated). |
| `gc_lineage`             | `np.ndarray`            | `(n_cells,)` | Germinal center lineage identifier for each B cell. |
| `lineage`                | `np.ndarray`            | `(n_cells,)` | Unique lineage identifier for each B cell. |
| `target_epitope`         | `np.ndarray`            | `(n_cells,)` | Epitope each B cell targets. |
| `variant_affinities`     | `np.ndarray`            | `(n_cells, n_variants)` | Log-transformed binding affinities. |
| `activated_time`         | `np.ndarray`            | `(n_cells,)` | Last activation time of each B cell. |

---

### **Mathematical Model**
1. **Activation Probability:** 
Activation Signal = (Antigen Concentration * Affinity) ^ Scaling Factor
- Higher **antigen concentration** and **affinity** increase activation probability.

2. **Birth Probability:**
Birth Probability = (Activation Strength * T-cell Help) / (1 + Activation Strength * T-cell Help)
- Higher **T-cell help** results in increased **B cell expansion**.

3. **Affinity Maturation Model:**
New Affinity = Old Affinity + Mutation Effect (dE)
- B cells **undergo mutations**, increasing or decreasing affinity.

4. **Differentiation Probabilities:**
P(Plasma) = Output Probability * Plasma Fraction P(Memory) 
= Output Probability * (1 - Plasma Fraction) P(Non-exported) = 1 - Output Probability

### **Error Handling**
- **Shape Mismatches**: Ensures array dimensions remain **consistent** during B cell processing.
- **Negative Affinity Values**: Checks that **mutations do not exceed limits**.
- **Sparse Matrix Efficiency**: Uses **`csr_matrix`** for efficient mutation tracking.

---

### **Dependencies**
Ensure the following Python packages are installed:
```bash
pip install numpy scipy

---

### Notes:
Sparse matrices (csr_matrix) are used to optimize memory usage for large populations.
Stochastic processes (e.g., differentiation, mutation, selection) introduce variability, making results 
probabilistic rather than deterministic. The module is intended for use in large-scale immunological modeling 
and computational immunology research.

"""

import copy
from typing import Self

import numpy as np
import scipy
from scipy.sparse import csr_matrix, lil_matrix

from . import utils
from .parameters import Parameters



class Bcells(Parameters):
    """
    Models B cell population dynamics for a GC simulation.

    Inherits simulation parameters from `Parameters` and extends functionality by maintaining 
    B cell-specific fields and methods.

    Core Attributes (inherited from Parameters):
      - B cell birth and death rates.
      - Initial B cell count (init_Bcells).
      - Biological parameters such as n_residues, n_variants, n_epitope, etc.

    Field Arrays:
      - mut_state: Sparse matrix (csr_matrix) tracking mutation states (0: unmutated, 1: mutated).
      - gc_lineage: Array indicating GC of origin for each B cell.
      - lineage: Array tracking a unique lineage ID per B cell.
      - target_epitope: Array specifying the epitope targeted by each B cell.
      - variant_affinities: Matrix (log10 scale) of B cell binding affinities to antigen variants.
      - activated_time: Array recording the last activation time for each B cell.

    **Example Usage:**
        >>> bcell_population = Bcells(initial_number=1000)
        >>> print(bcell_population.mutation_state_array.shape)  # Expected: (1000, n_residues)
        >>> new_population = bcell_population.get_daughter_bcells(conc_matrix, tcell=100)
        >>> print(len(new_population.lineage))  # Number of newly divided B cells
    """

    def __init__(
        self, 
        updated_params_file: str | None=None, 
        initial_number: str | None=None
    ):
        """
        Initialize a BCells instance.

        Process:
          1. Update parameters from a JSON file if provided.
          2. Set core attributes: birth_rate from Bcell_birth, death_rate (to be set later), 
             and initial count (init_count or default init_Bcells).
          3. Initialize all B cell fields via reset_bcell_fields().
          4. Define list of field names for batch operations.

        Args:
            updated_config (str or None): Path to a JSON config file.
            init_count (int or None): Initial number of B cells; defaults to init_Bcells.

        Example Usage:
            >>> bcell_population = Bcells(updated_params_file="params.json", initial_number=1000)
            >>> print(len(bcell_population.lineage))  # Expected: 1000 (or value from file)
        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)

        self.birth_rate = self.bcell_birth_rate
        self.death_rate = None
        self.initial_number = (
            initial_number if initial_number else self.initial_bcell_number
        )

        # Initialize all B cell arrays
        self.reset_bcell_fields()

        # Arrays we collectively manipulate in e.g. replace_all_arrays
        self.bcell_field_keys = [
            'mutation_state_array',
            'gc_lineage',
            'lineage',
            'target_epitope',
            'memory_reentry_tag', #1 if memory cell that re-enters naive pool, 0 otherwise
            'variant_affinities',
            'activated_time',
        ]

    
    def reset_bcell_fields(self) -> None:
        """
        Initialize or reset all B cell field arrays.

        Sets:
          - mut_state as a sparse matrix of zeros with shape (initial_number, n_residues).
          - gc_lineage, lineage, and target_epitope as zero-filled arrays.
          - variant_affinities as a zero matrix of shape (initial_number, n_variants).
          - activated_time as a zero-filled array.

        Raises:
            ValueError: If `self.initial_number` or `self.n_residues` is uninitialized 
                        or contains invalid values.
            RuntimeError: If an unexpected error occurs while initializing field arrays.

        Example Usage:
            >>> bcell_population.reset_bcell_fields()
            >>> print(bcell_population.mutation_state_array.shape)  # Expected: (initial_number, n_residues)
        """
        try:
            # Initialize mutation state array as a sparse matrix
            self.mutation_state_array = csr_matrix(np.zeros(
                (self.initial_number, self.n_residues), dtype=int
            ))

            # Initialize other B cell field arrays
            self.gc_lineage = np.zeros(self.initial_number, dtype=int)
            self.lineage = np.zeros(self.initial_number, dtype=int)    
            self.target_epitope = np.zeros(self.initial_number, dtype=int)  
            self.memory_reentry_tag = np.zeros(self.initial_number, dtype=int)                            
            self.variant_affinities = np.zeros((self.initial_number, self.n_variants))             
            self.activated_time = np.zeros(self.initial_number)   

        except Exception as e:
            raise RuntimeError(f"Error resetting B cell fields: {e}") 

    def subset_bcell_fields(self, indices: np.ndarray) -> None:
        """
        Replace each B cell field array with its subset defined by the provided indices.

        This function iterates over the keys in `self.Bcell_field_keys` and updates each corresponding 
        attribute by selecting only the elements at positions given in `indices`.

        Args:
            indices (np.ndarray): Array of indices to retain in all field arrays.

        Raises:
            RuntimeError: If an error occurs during the subsetting operation.
        
        Example:
            >>> idx_keep = np.array([0, 1, 3, 4])
            >>> bcell_population.subset_bcell_fields(idx_keep)
        """
        try:
            for key in self.bcell_field_keys:
                array = getattr(self, key)
                setattr(self, key, array[indices])
        except Exception as e:
            raise RuntimeError(f"Error subsetting fields with indices {indices}: {e}")


    def exclude_bcell_fields(self, indices: np.ndarray) -> None:
        """
        Remove B cells at the specified indices from all field arrays.

        This function computes the complementary indices (i.e. the indices not in `indices`)
        and then calls `subset_bcell_fields` with these complementary indices to update all field arrays.

        Args:
            indices (np.ndarray): Array of indices representing B cells to be removed.

        Raises:
            RuntimeError: If filtering fails due to mismatched array dimensions or other issues.
        
        Example:
            >>> idx_remove = np.array([2, 5, 8])
            >>> bcell_population.exclude_fields(idx_remove)
        """
        try:
            keep_indices = utils.get_other_idx(np.arange(self.lineage.size), indices)
            self.subset_bcell_fields(keep_indices)
        except Exception as e:
            raise RuntimeError(f"Error excluding indices {indices}: {e}")


    def merge_bcells(self, other: Self) -> None:
        """
        Merge the B cell data from `other` instance into self.

        This method concatenates each field array (listed in `self.Bcell_field_keys`) from the
        current instance with the corresponding field array from the other instance. Sparse arrays
        (csr_matrix) are merged using vertical stacking, while dense NumPy arrays are concatenated
        along axis 0.

        Args:
            other (Self): Another BCells instance to merge into this instance.

        Raises:
            ValueError: If a field's data type is unsupported for merging.
            RuntimeError: If an unexpected error occurs during the merge.
        """
        try:
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
                    raise ValueError(f"Unsupported field type for {key}: {type(current_field)}")

                setattr(self, key, new_field)
        except Exception as err:
            raise RuntimeError(f"Error merging BCells: {err}")
    

    # ------------------------------------
    # Activation & Mutation
    # ------------------------------------ 
    def set_activation_time(self, current_time: float, shift: bool=True) -> None:
        """
        Set the activation time for all B cells.

        This method fills the `activated_time` array with the provided `current_time`.
        If `shift` is True, a temporal offset of 0.5 * time_step is added to improve numerical alignment.

        Args:
            current_time (float): The current simulation time.
            shift (bool, optional): Whether to add a 0.5 * time_step offset. Defaults to True.

        Raises:
            RuntimeError: If an error occurs while updating the activation times.
        """
        try:
            offset = 0.5 * self.time_step if shift else 0
            self.activated_time.fill(current_time + offset)
        except Exception as e:
            raise RuntimeError(f"Error setting activation time at {current_time}: {e}")
    
    def set_memory_reentry_tag(self) -> None:
        """ Tag all bcells in this population as memory cells that are re-entering naive pool. """
        self.memory_reentry_tag = np.ones(self.lineage.size, dtype=int) 

    def get_dE(
        self, 
        idx_new: int, 
        idx: int, 
        ep: int
    ) -> np.ndarray:
        """
        Compute the affinity change (ΔE) for newly mutated B cells.

        For the new cells generated between indices `idx` and `idx_new`, this method samples
        ΔE values from a log-normal distribution. It assumes that the underlying normal distribution
        has zero mean and a covariance given by:
        
            σ = (mut_PDF[1])² * epitope_covariance_matrix[ep]
        
        The sampled values X are transformed into affinity changes using:
        
            ΔE = -log10(e) * (exp(X) - mut_PDF[2])
        
        Args:
            idx_new (int): Total B cell count after mutation.
            idx (int): B cell count before mutation.
            ep (int): Index of the target epitope.

        Returns:
            np.ndarray: An array of affinity changes with shape 
                        ((idx_new - idx) * n_residues, n_variants).

        Raises:
            RuntimeError: If an error occurs during sampling or transformation.
        """
        try:
            mu = np.zeros(self.n_variants)
            sigma = self.mutation_pdf[1] ** 2 * self.epitope_covariance_matrix[ep]
            num_samples = (idx_new - idx) * self.n_residues

            # Sample from the underlying normal distribution and add the specified offset.
            X = self.mutation_pdf[0] + np.random.multivariate_normal(mu, sigma, num_samples)
            # Transform the samples to log-scale affinity changes.
            dE = -np.log10(np.exp(1)) * (np.exp(X) - self.mutation_pdf[2])
            return dE
        except Exception as e:
            raise RuntimeError(f"Error calculating affinity changes: {e}")

    
    # ------------------------------------
    # Activation
    # ------------------------------------
    def get_activation_signal(self, conc_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-cell activation signals and determine activation status.

        The activation signal for each B cell is derived by combining:
        1. A concentration term, obtained by normalizing the effective antigen concentration 
            (conc_array) by the antigen normalization constant (C0).
        2. An affinity term, computed by clipping the log10-transformed binding affinities 
            to a maximum value (E_sat) and converting them to linear scale via 10^(affinity - E0).

        These two terms are multiplied and raised to the power specified by the capture stringency.
        If an alternative antigen capture model is active (w1 > 0), the signal is further adjusted.

        The overall signal is summed across antigen variants to yield a single value per cell.
        Each cell is then probabilistically activated: for each cell, a random number is drawn 
        uniformly from the interval [0, 1). If the cell's activation signal (clipped to a maximum of 1)
        exceeds the random number, the cell is marked as activated.

        Args:
            conc_array (np.ndarray): Effective antigen concentration for each B cell, with shape
                                    (n_cells, n_ag), where n_cells is the number of B cells and
                                    n_ag is the number of antigen variants.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - activation_signal: Array of activation signals (shape: (n_cells,)).
                - activated: Boolean array indicating activation status for each cell (shape: (n_cells,)).

        Raises:
            ValueError: If conc_array does not have the expected shape.
            RuntimeError: For any unexpected error during the computation.
        """
        try:
            # Validate the shape: expect (n_cells, n_ag)
            if conc_array.ndim != 2 or conc_array.shape[1] != self.n_ag:
                raise ValueError(f"Invalid conc_array shape {conc_array.shape}; expected (n_cells, n_ag).")

            # Compute concentration term by normalizing with Ag_norm (C0)
            conc_term = conc_array / self.C0

            # Compute affinity term:
            # - Clip log10 affinities (for the targeted variants) to a maximum (E_sat)
            # - Convert these values to linear scale using 10^(affinity - E0)
            energies = np.clip(self.variant_affinities[:, 0], None, self.E_sat)
            aff_term = np.power(10, energies - self.E0)

            # Combine concentration and affinity terms with capture stringency exponent
            signal = np.power(conc_term * aff_term, self.stringency)

            # Adjust signal for alternative antigen capture if enabled (w1 > 0)
            if self.w1 > 0:
                signal = ((self.w1 + 1) * signal) / (self.w1 + signal)

            # Sum the signal over all antigen variants to obtain a single value per cell
            activation_signal = signal.sum(axis=1)

            # Determine activation status:
            # For each cell, generate a random number uniformly in [0, 1). If the cell's activation signal
            # (clipped between 0 and 1) is greater than the random number, the cell is activated.
            activated_mask = np.random.uniform(size=activation_signal.size) < np.clip(activation_signal, 0, 1)
        except Exception as e:
            raise RuntimeError(f"Error computing activation signal: {e}")

        return activation_signal, activated_mask
    

    def get_birth_signal(self, 
                         activation_signal: np.ndarray, 
                         activated_mask: np.ndarray, 
                         tcell: float,
                         birth_rate: float) -> np.ndarray:
        """
        Calculate the division (birth) signal for each B cell. 

        This function computes a birth signal that integrates the cell's activation strength
        and the available T-cell help. The process involves:

        1. **Activation Fitness:**  
            Each cell's fitness is given by the product of its activation signal and its 
            activation status (with inactive cells contributing zero).

        2. **T-cell Help Allocation:**  
            Total T-cell help (tcell) is distributed among activated cells in proportion to 
            their fitness. The help for each cell is scaled relative to the average fitness 
            of all activated cells.

        3. **Logistic Scaling:**  
            The final birth signal is computed using a logistic function:
                birth_signal = birth_rate * tcell_help / (1 + tcell_help)
            This scaling prevents the birth signal from becoming unbounded.

        If no cells are activated (or if the sum of fitness values is zero), a zero array is returned.

        Args:
            activation_signal (np.ndarray): Activation signal per cell (shape: (n_cells,)).
            activated_mask (np.ndarray): Boolean array (shape: (n_cells,)); True for activated cells.
            tcell (float): Total available T-cell help.
            birth_rate (float): Baseline division rate for B cells.

        Returns:
            np.ndarray: Birth signals for each B cell, scaled by activation strength 
                        and T-cell help. Shape: `(n_cells,)`.

        Raises:
            ValueError: If `activation_signal` and `activated` arrays have mismatched shapes.
            RuntimeError: If an unexpected error occurs during computation.

        Example Usage:
            >>> activation_signals = np.array([0.2, 0.5, 0.8, 0.1])
            >>> activated_mask = np.array([True, False, True, False])
            >>> birth_signals = bcell_population.get_birth_signal(activation_signals, activated_cells, tcell=100, birth_rate=0.3)
            >>> print(birth_signals.shape)  # Expected: (4,)
        """

        try:
            # Validate input shapes
            if activation_signal.shape != activated_mask.shape:
                raise ValueError("Shapes of `activation_signal` and `activated_mask` do not match.")

            # Calculate fitness for activated cells
            activated_fitness = activation_signal * activated_mask

            # Handle edge cases where no cells are activated
            if activated_fitness.sum() == 0 or activated_mask.sum() == 0:
                return np.zeros_like(activation_signal)

            # Compute average fitness of activated cells
            avg_fitness = activated_fitness[activated_fitness > 0].mean()

            # Calculate T-cell help per activated cell
            tcell_help = (tcell / activated_mask.sum()) / avg_fitness * activated_fitness
            birth_signal = birth_rate * tcell_help / (1 + tcell_help)
            return birth_signal
        except Exception as e:
            raise RuntimeError(f"Error during birth signal computation: {e}")

     # ------------------------------------
    # Seeding & Division
    # ------------------------------------
    def get_seeding_indices(self, conc: np.ndarray, tcell: float) -> np.ndarray:
        """
        Select indices of B cells eligible for germinal center (GC) seeding.

        This method identifies B cells that will enter the GC by performing the following steps:
        1. Build a per-cell antigen concentration array from epitope-level data.
        2. Compute activation signals for each B cell using the antigen concentration and affinity.
        3. Calculate an entry probability (birth signal) based on the activation signal and available T-cell help.
        4. Stochastically select cells for GC entry by comparing the entry probability (scaled by the time step)
            against a random uniform value for each cell.

        Args:
            antigen_conc (np.ndarray): Array of effective antigen concentrations per epitope,
                with shape (n_epitope, n_Ag), where n_epitope is the number of epitopes and
                n_Ag is the number of antigen variants.

        Returns:
            np.ndarray: Indices of B cells selected for GC seeding (shape: (n_selected_cells,)).

        Raises:
            ValueError: If antigen_conc does not have the expected shape (n_epitope, n_Ag).
            RuntimeError: If an error occurs during the selection process.

        Example Usage:
            >>> conc_matrix = np.random.rand(10, 5)  # 10 epitopes, 5 antigen variants
            >>> seeding_indices = bcell_population.get_seeding_idx(conc_matrix)
            >>> print(seeding_indices.shape)  # Expected: (n_selected_cells,)
        """
        try:
            # Initialize the concentration array for each B cell
            conc_array = np.zeros(shape=(self.lineage.size,))
            for ep in range(self.n_ep):
                matching_epitope = self.target_epitope == ep
                conc_array += (
                    conc[ep] * 
                    matching_epitope 
                )

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
        except Exception as e:
            raise RuntimeError(f"Error determining seeding indices: {e}")
        
        # Return indices of selected B cells
        return np.nonzero(incoming_naive)[0]
        

    def get_birth_indices(self, conc: np.ndarray, tcell: float) -> np.ndarray:
        """
        Identify B cells that will undergo division (birth).

        This method determines which B cells will divide based on their antigen 
        exposure and available T-cell help. The selection process involves:

        1. **Antigen Matching**: Each B cell’s targeted epitope is mapped to 
        the provided antigen concentration values.
        2. **Activation Signal Calculation**: Computes activation signals 
        for all B cells based on antigen binding.
        3. **Birth Signal Computation**: Determines the probability of B cell 
        division using T-cell-mediated selection (`get_birth_signal()`).
        4. **Stochastic Selection**: B cells are probabilistically chosen for 
        division based on their birth probabilities.

        Parameters:
            conc (np.ndarray): Effective antigen concentration for each epitope.
                Shape: `(n_ep, n_ag)`, where:
                - `n_ep` is the number of epitopes.
                - `n_ag` is the number of antigen variants.
            tcell (float): The current amount of available T cells providing 
                help for B cell activation.

        Returns:
            np.ndarray: Indices of B cells selected for division.
                        Shape: `(n_selected_cells,)`.

        Raises:
            ValueError: If `conc` does not match the expected shape `(n_ep, n_ag)`.
            RuntimeError: If an error occurs during birth index computation.

        Example Usage:
            >>> conc_matrix = np.random.rand(10, 5)  # 10 epitopes, 5 antigen variants
            >>> birth_indices = bcell_population.get_birth_idx(conc_matrix, tcell=100)
            >>> print(birth_indices.shape)  # Expected: (n_selected_cells,)
        """
        try:
            # Initialize the concentration array for each B cell
            conc_array = np.zeros(shape=(self.lineage.size,))
            for ep in range(self.n_ep):
                #conc[ep] is a float, target_epitope is (nbcells,)
                conc_array += conc[ep] * (self.target_epitope == ep)

            # Calculate activation signals and determine activated cells
            activation_signal, activated = self.get_activation_signal(conc_array)

            if activated.sum(): # at least one B cell is intrinsically activated
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
        except Exception as e:
            raise RuntimeError(f"Error determining birth indices: {e}")

        # Return indices of selected B cells
        return np.nonzero(birth_idx)[0]

    # ------------------------------------
    # Removal / Kill
    # ------------------------------------
    def get_death_indices(self) -> np.ndarray:
        """
        Compute indices of B cells that will undergo apoptosis during the current time step.

        Each B cell dies with probability equal to (death_rate × time_step). This function
        generates a uniform random number for each cell and selects those where the random
        number is below the computed death probability.

        Returns:
            np.ndarray: Indices of B cells to be removed.
        """
        death_idx = np.zeros(self.lineage.shape).astype(bool)
        killed = np.random.uniform(size=self.lineage.size) < self.death_rate * self.dt
        death_idx = np.nonzero(killed)[0]
        return death_idx
    

    def remove_dead_cells(self) -> None:
        """
        Remove B cells slated for apoptosis during the current time step.

        This method performs the following steps:
        1. Compute the indices of cells to be removed using a stochastic process,
            where each cell dies with probability (death_rate × time_step).
        2. Exclude the identified cells from all B cell field arrays.
        3. If no cells remain (i.e., the population is empty), reset all B cell fields.

        Raises:
            RuntimeError: If an error occurs during the removal process.
        """
        try:
            # Step 1: Identify cells for death.
            death_idx = self.get_death_indices()
            # Step 2: Exclude the dead cells from all field arrays.
            self.exclude_bcell_fields(death_idx)
            # Step 3: If the population is empty, reset the fields.
            if self.lineage.size == 0:
                self.reset_bcell_fields()
        except Exception as e:
            raise RuntimeError(f"Error removing dead cells: {e}")

    # ------------------------------------
    # Subsetting & Merging
    # ------------------------------------
    def get_bcells_from_indices(self, indices: np.ndarray) -> Self:
        """
        Return a new BCells instance containing only the cells at the specified indices.

        This method creates a complete, independent copy of the current BCells instance
        (using a deep copy) so that modifications to the subset do not affect the original.
        It then filters all the B cell field arrays to retain only the entries corresponding
        to the provided indices by calling `subset_Bcell_fields`.

        Note that a deep copy is necessary because the field arrays (e.g., mutation_state_array, gc_lineage,
        lineage, target_epitope, variant_affinities, activated_time) are mutable objects. Without a deep copy,
        changes to the subset would reflect back in the original instance.

        Args:
            indices (np.ndarray): Array of indices indicating which B cells to retain.

        Returns:
            Self: A new BCells instance containing only the selected cells.

        Raises:
            RuntimeError: If an error occurs during the subsetting process.

        Example:
            >>> idx = np.array([0, 2, 4, 6])
            >>> subset_cells = bcell_population.select_Bcells(idx)
            >>> print(len(subset_cells.lineage))  # Should match len(idx)
        """
        try:
            new_bcells = copy.deepcopy(self)

            # Avoid error in replace_all_arrays
            if new_bcells.lineage.size == 0:
                return new_bcells
            
            new_bcells.subset_bcell_fields(indices)
            return new_bcells
        
        except Exception as e:
            raise RuntimeError(f"Error selecting cells with indices {indices}: {e}")
    
    
    def get_seeding_bcells(self, antigen_conc: np.ndarray, tcell: float) -> Self:
        """
        Return a new BCells instance containing only cells eligible for GC seeding.

        This method uses the provided antigen concentration data (with shape 
        (n_epitope, n_Ag)) to compute activation signals and determine which B cells 
        are eligible for germinal center (GC) seeding. It then extracts these cells 
        and returns a new, independent BCells instance containing only the selected cells.

        The selection process involves:
        1. Computing the per-cell antigen concentration array via `_build_conc_array`.
        2. Determining activation signals and status using `get_activation_signal`.
        3. Identifying eligible cells via `get_seeding_indices`.

        Args:
            antigen_conc (np.ndarray): Antigen concentration for each epitope 
                (shape: (n_epitope, n_Ag)).
            tcell (float): The available T-cell help for B cell division.

        Returns:
            Self: A new BCells instance with only GC-seeding cells.

        Raises:
            RuntimeError: If an error occurs during the selection process.
        """
        seeding_idx = self.get_seeding_indices(antigen_conc, tcell)
        return self.get_bcells_from_indices(seeding_idx)
    

    def get_daughter_bcells(self, antigen_conc: np.ndarray, tcell: float) -> Self:
        """
        Return a new BCells instance containing only the daughter (dividing) B cells.

        This method identifies B cells that are selected to divide based on their effective 
        antigen exposure and the available T-cell help. It performs the following steps:
        
        1. **Determine Dividing Cells:**  
            Computes the indices of dividing cells by calling `get_birth_indices(antigen_conc, tcell)`.
        
        2. **Extract Subset:**  
            Returns an independent (deep-copied) instance containing only the cells at those indices via 
            `get_Bcells_from_indices`.

        Args:
            antigen_conc (np.ndarray): Effective antigen concentration for each epitope, with shape 
                                    (n_epitope, n_Ag) where n_epitope is the number of epitopes and 
                                    n_Ag is the number of antigen variants.
            tcell (float): The available T-cell help for B cell division.

        Returns:
            Self: A new BCells instance containing only the daughter (dividing) B cells. If no cells are 
                selected, an empty instance is returned.

        Raises:
            RuntimeError: If an error occurs during the selection process.
        """
        try:
            # Identify indices of daughter B cells
            dividing_indices = self.get_birth_indices(antigen_conc, tcell)

            # Create and return a copy containing only daughter B cells
            return self.get_bcells_from_indices(dividing_indices)

        except Exception as e:
            raise RuntimeError(f"Error retrieving daughter B cells: {e}")

    # ------------------------------------
    # Mutation
    # ------------------------------------
    def mutate_bcells_by_indices(
        self, 
        indices: np.ndarray, 
        dE_matrix: np.ndarray
    ) -> Self:
        """
        Return a new BCells instance with mutations applied to the cells at the specified indices.

        For each cell selected by `indices`, a random residue is chosen for mutation. If the cell 
        is unmutated at that residue (state 0), its mutation state is flipped to 1 and its binding 
        affinity is increased by the corresponding value from `deltaE_matrix`. Conversely, if the 
        cell is already mutated (state 1), its state is reverted to 0 and its affinity is decreased.
        
        A deep copy of the selected cells is made to ensure that modifications do not affect the 
        original population.

        Args:
            indices (np.ndarray): Indices of B cells to mutate.
            dE_matrix (np.ndarray): Precomputed affinity changes (ΔE) for mutations, with shape 
                (n_gc, n_cell, n_residues, n_variants).

        Returns:
            Self: A new BCells instance with the selected cells mutated.

        Raises:
            ValueError: If the mutation state array contains non-binary values or if any affinity 
                        exceeds the maximum allowed value.
            RuntimeError: If an error occurs during the mutation process.
        """
        try:
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
            mutated_bcells.variant_affinities[nonmutated_cells] += (
                dE_matrix[
                    gc_lineage_nonmutated, 
                    lineage_nonmutated, 
                    residues_nonmutated
                ]
            )

            mutated_bcells.variant_affinities[already_mutated_cells] -= (
                dE_matrix[gc_lineage_mutated, lineage_mutated, residues_mutated]
            )

            # Ensure that no affinity exceeds the maximum allowable value.
            if np.any(mutated_bcells.variant_affinities > self.max_affinity):
                raise ValueError('Affinity impossibly high.')
        except Exception as e:
            raise RuntimeError(f"Error during mutation process: {e}") 

        return mutated_bcells


    def differentiate_bcells(
        self, 
        export_prob: float, 
        PC_frac: float,
        dE_matrix: np.ndarray,
        mutate: bool=True
    ) -> tuple[Self, Self, Self]:
        """
        Partition B cells into memory, plasma, and non-exported groups.

        The process is as follows:
        1. **Export Selection:**  
            - Randomly select cells for export with probability `output_prob`.
            - From exported cells, select a fraction (`pc_frac`) to differentiate into plasma cells;
            the remaining exported cells become memory cells.
        2. **Non-Export Handling:**  
            - The cells that are not exported (nonexported) are further culled by a death probability
            (p_lethal). The survivors are divided into two groups:
                a. Cells that undergo silent mutations.
                b. Cells that are eligible for affinity-altering mutations.
        3. **Mutation (if enabled):**  
            - If `mutate` is True, apply affinity changes to the cells in group (b) using the precomputed
            deltaE_matrix.
        4. **Merge Non-Exported:**  
            - Combine the silently mutated cells with the affinity-mutated cells to form the final non-exported group.

        Args:
            output_prob (float): Probability that a newly birthed B cell is exported.
            pc_frac (float): Fraction of exported cells that become plasma cells.
            deltaE_matrix (np.ndarray): Precomputed affinity change values with shape 
                (n_gc, n_cell, n_residues, n_variants).
            mutate (bool, optional): Whether to apply affinity-altering mutations to non-exported cells.
                                    Defaults to True.

        Returns:
            tuple[Self, Self, Self]: A tuple containing three BCells instances:
                - Memory cells (exported but not plasma).
                - Plasma cells (exported and differentiating into plasma cells).
                - Non-exported cells (remaining cells, including mutated ones if mutate=True).

        Raises:
            ValueError: If probability calculations or mutation state validations fail.
            RuntimeError: If an unexpected error occurs during differentiation.

        Example Usage:
            >>> precalculated_dEs = np.random.rand(5, 100, 20, 3)  # Example mutation effects
            >>> memory, plasma, nonexported = bcell_population.differentiate_bcells(
            ...     output_prob=0.3, output_pc_fraction=0.5, precalculated_dEs=precalculated_dEs
            ... )
            >>> print(len(memory.lineage), len(plasma.lineage), len(nonexported.lineage))
        """
        try:
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
            silent_prob = min((
                self.mutation_silent_prob * 
                len(nonexport_idx) / 
                (len(survivors_idx) + self.epsilon)
            ), 1)
            silent_mutation_idx = utils.get_sample(survivors_idx, p=silent_prob)
            affinity_change_idx = utils.get_other_idx(survivors_idx, silent_mutation_idx)

            # Step 3: Retrieve B cell populations
            memory_bcells = self.get_bcells_from_indices(memory_idx)
            plasma_bcells = self.get_bcells_from_indices(plasma_idx)
            nonexported_bcells = self.get_bcells_from_indices(silent_mutation_idx)
            
            # Step 4: Mutate affinity-changing B cells if applicable
            if mutate:
                mutated_bcells = self.mutate_bcells_by_indices(
                    affinity_change_idx, 
                    dE_matrix
                )
            else:
                mutated_bcells = self.get_bcells_from_indices(affinity_change_idx)
            
            # Combine mutated B cells with the non-exported group
            nonexported_bcells.add_bcells(mutated_bcells)
        except Exception as e:
            raise RuntimeError(f"Error during B cell differentiation: {e}")

        return memory_bcells, plasma_bcells, nonexported_bcells
    
    # ------------------------------------
    # Analysis
    # ------------------------------------
    def count_high_affinity_cells(self) -> np.ndarray:
        """
        Count B cells with binding affinities exceeding predefined thresholds.

        For each epitope and for each threshold in self.aff_history, this method computes 
        the number of B cells (separately for each antigen variant) whose binding affinities 
        (from self.variant_affinities) exceed the threshold. Only cells with a matching target 
        epitope (from self.target_epitope) are considered.

        Returns:
            np.ndarray: An array of shape (n_variants, n_epitope, n_thresholds), where:
                - n_variants is the number of antigen variants,
                - n_epitope is the number of epitopes,
                - n_thresholds is the number of thresholds in self.aff_history.

        Raises:
            ValueError: If self.aff_history is not defined or is empty.
            RuntimeError: If an error occurs during the computation.

        Example Usage:
            >>> num_high_affinity_bcells = bcell_population.count_high_affinity_cells()
            >>> print(num_high_affinity_bcells.shape)  # Expected: (n_var, n_ep, n_affinities)
        """
        try:
            # Initialize output array: counts[variant, epitope, threshold]
            counts = np.zeros(
                (self.n_variants, self.n_ep, len(self.history_affinity_thresholds))
            )

            # Loop over each threshold and each epitope.
            for threshold_idx, affinity_threshold in enumerate(self.history_affinity_thresholds):
                for ep in range(self.n_ep):
                    # Select cells targeting the current epitope.
                    cell_affinities = self.variant_affinities[self.target_epitope == ep]
                    # Count, for each variant, the number of cells with affinity above the threshold.
                    n_cells = (cell_affinities > affinity_threshold).sum(axis=0)
                    counts[:, ep, threshold_idx] = n_cells
        except Exception as e:
            raise RuntimeError(f"Error counting high-affinity cells: {e}")
        
        return counts
    
    def get_num_in_aff_bins(self) -> np.ndarray:
        """Get the number of bcells with affinities in different affinity bins.
        
        Thresholds are specified in affinities_history. Numbers are also
        specific to each targeted epitope and each variant Ag.

        Returns:
            num_above_aff: Number of bcells with affinity above thresholds.
                np.ndarray (shape=(n_var, n_ep, n_affinities))
        """
        num_in_aff = np.zeros(
            (self.n_variants, self.n_ep, len(self.affinity_bins))
        )
        for aff_idx, affinity_bin in enumerate(self.affinity_bins):
            for ep in range(self.n_ep):
                variant_affinities = self.variant_affinities[
                    self.target_epitope == ep
                ]
                if aff_idx == 0: #first bin
                    affinities_in_range = (variant_affinities <= affinity_bin) 
                else:
                    affinity_prev_bin = self.affinity_bins[aff_idx - 1]
                    affinities_in_range = (variant_affinities > affinity_prev_bin) & (variant_affinities <= affinity_bin) 
                n_cells = (affinities_in_range).sum(axis=0)
                num_in_aff[:, ep, aff_idx] = n_cells
        return num_in_aff

    """NOTE: Previous version of get_num_in_aff_bins had loops over epitopes and variants (see below)
    
    # Preallocate the result array to hold counts for each variant, epitope, and affinity bin.
        num_in_aff = np.zeros((self.n_variants, self.n_epitopes, len(self.affinity_bins)), dtype=int)

        # Loop over each epitope.
        for ep in range(self.n_epitopes):
            # Select the rows (B-cells) where the target epitope equals the current epitope.
            # This yields a sub-array of affinities with shape (n_cells, n_var).
            sub_affinities = self.variant_affinities[self.target_epitope == ep]
            # If no cells target this epitope, skip to the next one.
            if sub_affinities.size == 0:
                continue

            # Use np.digitize to determine the bin index for each affinity value.
            # With right=True, a value x is assigned to bin index i if:
            #   - For i == 0: x <= self.affinity_bins[0]
            #   - For i > 0: self.affinity_bins[i-1] < x <= self.affinity_bins[i]
            # np.digitize returns an array of the same shape as sub_affinities with integer bin indices.
            bin_indices = np.digitize(sub_affinities, self.affinity_bins, right=True)

            # Loop over each antigen variant.
            for var in range(self.n_variants):
                # For the current variant, extract the bin indices for all cells.
                variant_bin_indices = bin_indices[:, var]
                # Count the number of occurrences of each bin index.
                # np.bincount returns an array of length at least (max(bin_indices)+1).
                counts = np.bincount(variant_bin_indices, minlength=len(self.affinity_bins) + 1)
                # We are interested in bins 0 to len(self.affinity_bins)-1.
                num_in_aff[var, ep, :] = counts[:len(self.affinity_bins)]

        return num_in_aff
    """
    
    def get_num_entry(self) -> np.ndarray:
        """ Get the total number of bcells across all affinities that target each epitope
        on each variant. Separate out bcells that are memory cells which re-entered gc (last
        dimension of returned array)

        Returns:
            total_num : Number of bcells that are naive (0) or were memory cells tha re-entered (1) 
                np.ndarray (shape=(n_var, n_ep, 2))
        """
        total_num = np.zeros(
            (self.n_variants, self.n_ep, 2)
        )
        for ep in range(self.n_ep):
            for tag in [0, 1]:
                cells_per_ep = (self.target_epitope == ep) & (self.memory_reentry_tag == tag)
                n_cells = cells_per_ep.sum(axis=0)
                total_num[:, ep, tag] = n_cells
        return total_num


