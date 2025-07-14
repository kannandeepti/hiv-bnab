r""" 
Simulation class
================
The `Simulation` class is the workhorse for all GC dynamics simulations. The
`run()` function runs an entire simulation of a persistent humoral response for
`simulation_time` days and saves a history.pkl file containing the numbers of 
B cells in each population (naive, GC, memory GC, memory EGC, plasma GC, plasma
EGC) and the concentration and affinity of antibodies & antigen each day of the
simulation.

In each time step, `run_timestep()` is called, in which
antibody and antigen concentrations are updated and naive B cells are activated
and selected for GC entry. In each time step, GC B cells undergo one round of 
mutation, selection, and export (see `run_gc`). In the same time step, memory B
cells derived from the GC are selected for EGC entry, and one round of selection
and export in the EGC also occurs (see `run_egc`). 

"""
import copy
import dataclasses
import datetime
import os
import time
from typing import Self, Any
from pathlib import Path

import numpy as np

from . import utils
from .parameters import Parameters
from .bcells import Bcells
from .concentrations import Concentrations



class Simulation(Parameters):
    """Class for running the simulation, inherits from Parameters, so all
    simulation parameters are available as attributes.

    Attributes:
        concentrations(Concentrations): tracks concentration of
            antigen on FDC, and the affinity and concentrations of antibodies.
        precalculated_dEs(np.ndarray of shape (n_gc, n_naive_precursors, n_residues, n_variants)): 
            Precalculated affinity changes upon mutation to each residue for each 
            naive B cell lineage in each GC to each variant.
            This is stored in Simulation so that it doesn't
            need to be copied to the Bcell class, which saves a lot of memory
            and a bit of time.
        attributes_to_replace: List of attributes that need to be replaced
            when reading a previous checkpoint pickle file. This will update
            the simulation to the correct state.
        history: Dict containing the history of the simulation. See reset_history.
    """

    def __init__(self, updated_params_file: str | None=None, parallel_run_idx: int=0):
        """Initialize attributes and set the np.random.seed.
        
        All the default parameters from Parameters are included. If updated_params_file is 
        passed, then the parameters specified in the file are updated from the file.

        All other inherited classes of Parameters need to have updated_params_file
        passed.
        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)
        np.random.seed(self.seed)

        self.concentrations = Concentrations(self.updated_params_file)
        self.set_naive_bcells_per_bin()
        self.precalculated_dEs = np.zeros(
            (self.n_gc, self.naive_bcells_int.sum(), self.n_residues, self.n_variants)
        )

        self.attributes_to_replace = [
            'concentrations',
            'dummy_bcells',
            'gc_bcells',
            'memory_gc_bcells',
            'memory_egc_bcells',
            'plasma_gc_bcells',
            'plasma_egc_bcells',
            'plasmablasts',
        ]
        self.parallel_run_idx = parallel_run_idx
        self.create_file_paths()
        self.reset_history()


    def reset_history(self) -> None:
        """(Re)initialize the ``history`` attribute.

        The method allocates zero-filled NumPy arrays for every statistic that the
        simulation tracks.  All arrays share the leading dimension
        ``n_history_timepoints`` (the number of history checkpoints).

        Top-level keys
        --------------

        * ``'gc_entry'``  
        * ``'total_num'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_gc, n_var, n_ep, 2)``):  
            Count of cells seeding each GC.  Last axis = 0 for naive B cell entry,  
            1 for memory B cell re-entry.

        * ``'gc'``  
        * ``'num_above_aff'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_gc, n_var, n_ep, n_affinities_history)``):  
            Number of GC B cells whose affinity exceeds each threshold in  
            ``self.affinities_history``.  
        * ``'num_in_aff'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_gc, n_var, n_ep, n_affinity_bins)``):  
            Number of GC B cells in each of the ``self.affinity_bins``.  
        * ``'num_by_lineage'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_gc, n_naive_precursors)``):  
            GC B-cell counts grouped by naive precursor lineage.

        * ``'conc'``  
        * ``'ic_fdc_conc'`` (np.ndarray, shape ``(n_history_timepoints, n_ep)``):  
            Concentration of each epitope displayed on FDCs.  
        * ``'ic_fdc_eff_conc'`` (np.ndarray, shape ``(n_history_timepoints, n_ep)``):  
            Effective concentration of each epitope on FDCs after masking. 
        * ``'ab_conc'`` (np.ndarray, shape ``(n_history_timepoints, n_ep)``):  
            Total antibody concentration in plasma targeting each epitope.  
        * ``'ab_ka'`` (np.ndarray, shape ``(n_history_timepoints, n_ep)``):  
            Affinity of antibodies targeting each epitope.

        * ``'plasma_gc'``  
        * ``'num_above_aff'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_var, n_ep, n_affinities_history)``):  
            GC-derived plasma-cell counts above each affinity threshold.  
        * ``'num_in_aff'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_var, n_ep, n_affinity_bins)``):  
            Number of GC-derived plasma cells in each of the ``self.affinity_bins``.   
        * ``'num_by_lineage'`` (np.ndarray, shape  
            ``(n_history_timepoints, n_naive_precursors)``):  
            Plasma-cell counts grouped by precursor lineage.

        * ``'memory_gc'``, ``'plasma_egc'``, ``'memory_egc'``  
        Deep copies of the ``'plasma_gc'`` structure, holding analogous stats for
        GC-derived memory cells and EGC-derived plasma/memory cells.

        * ``'ep_to_lineage'`` (list[int]):  
        Boundaries in the lineage index array that separate naive cells targeting
        different epitopes.

        Returns:
            None
        """

        self.history = {
            'gc_entry' : {
                'total_num': np.zeros((
                    self.n_history_timepoints,
                    self.n_gc,
                    self.n_variants,
                    self.n_ep,
                    2 #0 is naive, 1 is memory cell re-entry
                ))
            },
            'gc': {
                'num_above_aff': np.zeros((
                    self.n_history_timepoints, 
                    self.n_gc,
                    self.n_variants,
                    self.n_ep, 
                    len(self.history_affinity_thresholds)
                )),
                'num_in_aff': np.zeros((
                    self.n_history_timepoints, 
                    self.n_gc,
                    self.n_variants,
                    self.n_ep, 
                    len(self.affinity_bins)
                )),
                'num_by_lineage': np.zeros((
                    self.n_history_timepoints, 
                    self.n_gc, 
                    self.n_naive_precursors
                )),
            },
            'conc': {
                'ic_fdc_conc': np.zeros((
                    self.n_history_timepoints, 
                    self.n_ep 
                )),
                'ic_fdc_eff_conc': np.zeros((
                    self.n_history_timepoints, 
                    self.n_ep 
                )),
                'ab_conc': np.zeros((
                    self.n_history_timepoints, 
                    self.n_ep,
                )),
                'ab_ka': np.zeros((
                    self.n_history_timepoints, 
                    self.n_ep, 
                )),
            },
            'plasma_gc': {
                'num_above_aff': np.zeros((
                    self.n_history_timepoints,
                    self.n_variants,
                    self.n_ep, 
                    len(self.history_affinity_thresholds)
                )),
                'num_in_aff': np.zeros((
                    self.n_history_timepoints, 
                    self.n_variants,
                    self.n_ep, 
                    len(self.affinity_bins)
                )),
                'num_by_lineage': np.zeros((
                    self.n_history_timepoints, 
                    self.n_naive_precursors
                )),
            },

        }
        self.history['memory_gc'] = copy.deepcopy(self.history['plasma_gc'])
        self.history['plasma_egc'] = copy.deepcopy(self.history['plasma_gc'])
        self.history['memory_egc'] = copy.deepcopy(self.history['plasma_gc'])
        self.history['ep_to_lineage'] = [] #store the lineage index boundaries separating naive cells that target different epitopes
    
    def get_parameter_dict(self) -> None:
        """Write parameters to json file.

        Properties from Parameter class are not included.
        """
        parameters = Parameters()
        parameters.update_parameters_from_file(self.updated_params_file)
        return {
            field.name: getattr(parameters, field.name)
            for field in dataclasses.fields(parameters)
        }
    
    def create_file_paths(self) -> None:
        """Build timestamped file-path attributes for simulation outputs.

        The method derives a unique sub-directory name from the current
        wall-clock time (`YYYY_MM_DD_HH_MM_SS`) concatenated with
        ``parallel_run_idx``.  It then stores four paths on the instance:

        * **data_dir (Path)** – Base directory for this simulation run
        (e.g. ``experiment_dir/2025_05_09_13_42_10_0``).
        * **history_path (Path)** – Location of the pickled history object
        (``data_dir / history_file_name``).
        * **sim_path (Path)** – Location of the pickled :class:`Simulation`
        object (``data_dir / simulation_file_name``).
        * **parameter_json_path (Path)** – Location of the exported parameter
        JSON file (``data_dir / param_file_name``).

        Note:
            The directory itself is *not* created here; it is expected to be
            created later when the first file is written.

        Returns:
            None
        """
        date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.data_dir = Path(self.experiment_dir)/f'{date_time}_{self.parallel_run_idx}'

        self.history_path = self.data_dir / self.history_file_name
        self.sim_path = self.data_dir / self.simulation_file_name
        self.parameter_json_path = self.data_dir / self.param_file_name
    
    def set_naive_bcells_per_bin(self) -> None:
        """Set the integer number of naive B cells in each affinity bin using
        stochastic rounding. 
        
        The values in ``self.naive_bcells_arr`` may contain non-integer numbers, so
        we add a uniform random number from :math:`\\mathcal{U}(0, 1)` and apply
        the floor function to obtain the final integer number of naive B cells
        in each affinity bin.

        Returns:
            None
        """
        randu = np.random.uniform(size=self.naive_bcells_arr.shape)
        self.naive_bcells_int = np.floor(self.naive_bcells_arr + randu).astype(int)

    def get_naive_bcells(self, gc_idx: int) -> Bcells:
        """Build and populate the naive-B-cell pool available to one GC.

        The method converts the integer counts stored in ``self.naive_bcells_int``—
        indexed by epitope (``n_ep``) and affinity bin (``len(self.fitness_array)``)—
        into a fully populated :class:`Bcells` object of size
        ``self.naive_bcells_int.sum()``.  For every block of cells that share the
        same epitope and affinity bin, it fills the following *field arrays*:

        * ``gc_lineage`` – GC identifier (all set to `gc_idx`).
        * ``lineage`` – Unique precursor lineage ID (*0 … `n_naive_precursors`-1*) 
        * ``target_epitope`` – Integer epitope index (*0 … `n_ep`-1*).
        * ``variant_affinities`` – Germline binding affinities to each variant:
        variants listed in ``self.naive_high_affinity_variants`` are assigned to the
        affinity value of their bin; all other variants are set to ``self.E0``.
        * ``dE`` – Pre-sampled binding affinity changes upon mutation from
        ``self.precalculated_dEs[gc_idx, …]``.

        For the first GC only (``gc_idx == 0``) the routine also appends the running
        lineage offset to ``self.history['ep_to_lineage']`` so that later analysis
        can map lineages back to their epitope of origin.

        Args:
            gc_idx (int): Zero-based index of the GC

        Returns:
            Bcells: A populated naive-B-cell compartment ready to seed GC ``gc_idx``.
        """
        naive_bcells = Bcells(
            updated_params_file=self.updated_params_file, 
            initial_number=self.naive_bcells_int.sum()
        )

        idx = 0
        for ep in range(self.n_ep):
            for j, fitness in enumerate(self.fitness_array):
                #count number of naive B cells that share an epitope/affinity bin
                idx_new = idx + self.naive_bcells_int[ep, j]
                naive_bcells.gc_lineage[idx:idx_new] = gc_idx
                naive_bcells.lineage[idx: idx_new] = np.arange(idx, idx_new)
                naive_bcells.target_epitope[idx: idx_new] = ep

                # Higher affinities
                naive_bcells.variant_affinities[
                    idx: idx_new, 
                    np.array(self.naive_high_affinity_variants)
                ] = fitness

                # Low affinities (E0)
                naive_bcells.variant_affinities[
                    idx: idx_new, 
                    utils.get_other_idx(
                        np.arange(self.n_variants), 
                        np.array(self.naive_high_affinity_variants)
                    )
                ] = self.E0

                dE = naive_bcells.get_dE(idx_new - idx, ep)
                for var in range(self.n_variants):

                    self.precalculated_dEs[gc_idx, idx:idx_new, :, var] = np.reshape(
                        dE[:, var], (idx_new - idx, self.n_residues), order='F'
                    )

                idx = idx_new
            if gc_idx == 0:
                self.history['ep_to_lineage'].append(idx)
        
        return naive_bcells
    

    def set_death_rates(self) -> None:
        """Set death rates for all Bcell populations.
        
        Returns:
            None
        """
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx].death_rate = self.bcell_death_rate
        plasma_rate = utils.get_death_rate_from_half_life(
            self.plasma_half_life, self.dt
        )
        memory_rate = utils.get_death_rate_from_half_life(
            self.memory_half_life, self.dt
        )
        self.plasma_gc_bcells.death_rate = plasma_rate
        self.plasma_egc_bcells.death_rate = plasma_rate
        self.memory_gc_bcells.death_rate = memory_rate
        self.memory_egc_bcells.death_rate = memory_rate
    

    def create_populations(self) -> None:
        """Initialize empty B cell populations and set B cell death rates.

        Attributes:
            gc_bcells(List[Bcells]): list of `n_gc` B cell populations for each GC
            naive_bcells(List[Bcells]): list of `n_gc` naive B cell populations
                to seed each of the GCs
            naive_cells_entry (List[Bcells]):
                Per-GC counters used to log how many naive or memory cells entered in
                the current timestep (reset after every history update).
            plasma_gc_bcells (Bcells):
                GC-derived plasma cell pool (initially empty).
            plasma_egc_bcells (Bcells):
                EGC-derived plasma cell pool (initially empty).
            memory_gc_bcells (Bcells):
                GC-derived memory cell pool (initially empty).
            memory_egc_bcells (Bcells):
                EGC-derived memory cell pool (initially empty).
            plasmablasts (Bcells):
                Short-lived antibody-secreting cells (created but not yet used).
            dummy_bcells (Bcells):
                Zero-size template object used for deep copies.
        
        Returns:
            None
        """
        self.dummy_bcells = Bcells(updated_params_file=self.updated_params_file)

        self.gc_bcells = [copy.deepcopy(self.dummy_bcells) for _ in range(self.n_gc)]
        self.naive_cells_entry = [copy.deepcopy(self.dummy_bcells) for _ in range(self.n_gc)]
        self.naive_bcells = [self.get_naive_bcells(gc_idx) for gc_idx in range(self.n_gc)]

        self.plasma_gc_bcells = copy.deepcopy(self.dummy_bcells)
        self.plasma_egc_bcells = copy.deepcopy(self.dummy_bcells)
        self.memory_gc_bcells = copy.deepcopy(self.dummy_bcells)
        self.memory_egc_bcells = copy.deepcopy(self.dummy_bcells)
        self.plasmablasts = copy.deepcopy(self.dummy_bcells) 
        # Doing nothing right now.

        self.set_death_rates()


    def run_gc(self, gc_idx: int) -> None:
        """Advance one germinal-center (GC) reaction by a single timestep.

        Workflow
        --------
        1. **Seeding**  
        Select naïve cells from ``self.naive_bcells[gc_idx]``  
        based on the effective antigen concentration
        (``self.ag_eff_conc``) and available T-cell help
        (``self.seeding_tcells_gc``).

        2. **Clonal expansion**  
        Each seeding cell undergoes ``self.naive_bcells_n_divide`` symmetric
        divisions, doubling its copy number each round.

        3. **Book-keeping**  
        Counts for naïve/memory entry during this timestep are accumulated in
        ``self.naive_cells_entry[gc_idx]`` for history logging.

        4. **Birth and differentiation**  
        *  Bcells are selected based on the effective antigen concentration
        (``self.ag_eff_conc``) and available T-cell help
        (``self.n_tfh_gc``). 
        *  The resulting “daughter” population is probabilistically split into
        memory, plasma, and non-exported GC-resident cells using  
        ``self.output_prob`` and ``self.output_pc_fraction``.

        5. **Population updates**  
        *  Memory cells are staged in ``self.temporary_memory_bcells``  
            (merged after each update to all GCs finish).  
        *  Plasma cells are appended to the global
            ``self.plasma_gc_bcells`` pool.  
        *  Non-exported cells are returned to the GC’s main compartment
            ``self.gc_bcells[gc_idx]``.

        Args:
            gc_idx (int): Zero-based index of the germinal center to update.

        Returns:
            None
        """
        seeding_bcells = self.naive_bcells[gc_idx].get_seeding_bcells(self.ag_eff_conc, self.seeding_tcells_gc)
        for _ in range(self.naive_bcells_n_divide):
            seeding_bcells.add_bcells(seeding_bcells)

        # Set activated time
        seeding_bcells.set_activated_time(self.current_time)

        #count number that entered in this time step
        self.naive_cells_entry[gc_idx].add_bcells(seeding_bcells)

        # Seed
        self.gc_bcells[gc_idx].add_bcells(seeding_bcells)
        
        # Birth and differentiation
        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(
            self.ag_eff_conc, self.n_tfh_gc
        )
        differentiated_bcells = daughter_bcells.differentiate_bcells(
            self.output_prob, 
            self.output_pc_fraction, 
            self.precalculated_dEs,
            mutate=True if self.current_time > self.mutate_start_time else False
        )
        memory_bcells, plasma_bcells, nonexported_bcells = differentiated_bcells

        # Set activated time
        memory_bcells.set_activated_time(self.current_time)
        plasma_bcells.set_activated_time(self.current_time)

        self.temporary_memory_bcells.add_bcells(memory_bcells)
        self.plasma_gc_bcells.add_bcells(plasma_bcells)
        self.gc_bcells[gc_idx].add_bcells(nonexported_bcells)


    def run_egc(self) -> None:
        """Advance extra-Germinal Center (EGC) reaction by one timestep.

        Workflow
        --------
        1. **Seeding**  
        Select memory cells from ``self.memory_gc_bcells`` based on the
        current effective antigen concentration (``self.ag_eff_conc``) and the T
        cells available for seeding EGC (``self.seeding_tcells_egc``).
        Add the seeding cells to the EGC memory compartment
        ``self.memory_egc_bcells``.

        2. **Differentiation**  
        *   Generate daughter cells using the same proliferation logic as GCs,
            but with a helper T cells from ``self.n_tfh_egc`` and without mutation
            (`mutate=False`).
        *   Split the daughters into memory and plasma fates with probabilities
            ``self.egc_output_prob`` and ``self.egc_output_pc_fraction``.  

        3. **Population updates**  
        *   Append plasma cells to ``self.plasma_egc_bcells``.  
        *   Append memory cells back into ``self.memory_egc_bcells``  

        Returns:
            None
    """
        seeding_bcells = self.memory_gc_bcells.get_seeding_bcells(self.ag_eff_conc, self.seeding_tcells_egc)

        # Set activated time
        seeding_bcells.set_activated_time(self.current_time)

        # Seed
        self.memory_egc_bcells.add_bcells(seeding_bcells)

        #Birth and differentiation
        daughter_bcells = self.memory_egc_bcells.get_daughter_bcells(
            self.ag_eff_conc, self.n_tfh_egc #Leerang had self.nmax * self.n_gc
        )
        differentiated_bcells = daughter_bcells.differentiate_bcells(
            self.egc_output_prob, 
            self.egc_output_pc_fraction,
            self.precalculated_dEs,
            mutate=False
        )
        memory_bcells, plasma_bcells, _ = differentiated_bcells

        # Set activated time
        memory_bcells.set_activated_time(self.current_time)
        plasma_bcells.set_activated_time(self.current_time)

        self.plasma_egc_bcells.add_bcells(plasma_bcells)
        self.memory_egc_bcells.add_bcells(memory_bcells)

    def split_memory_bcells(self) -> None:
        """Partition GC-derived memory cells into GC re-entry and EGC-seeding pools.

        A fraction of memory cells, drawn according to
        ``self.memory_to_gc_fraction``, is re-routed to ``self.naive_bcells`` where 
        they will compete with naïve cells for reseeding of GCs.
        The remainder stays in ``self.memory_gc_bcells`` and can later seed EGC.

        Returns:
            None

        """

        # Split GC-derived memory cells into GCs and EGC
        memory_to_gc_idx = utils.get_sample(
            np.arange(self.memory_gc_bcells.lineage.size), 
            p=self.memory_to_gc_fraction,
        )

        # Add selected memory cells to naive pool for each GC
        # memory_size_to_split = memory_to_gc_idx.size // self.n_gc      # Not needed, array_split distributes mem to n_gcs
        
        if memory_to_gc_idx.size > 0: 
            per_gc_indices = np.array_split(memory_to_gc_idx, self.n_gc)
            
            for gc_idx, indices_for_this_gc in enumerate(per_gc_indices):
                if indices_for_this_gc.size > 0:
                    memory_to_gc_bcells = self.memory_gc_bcells.get_bcells_from_indices(
                        indices_for_this_gc
                    )
                    #set memory re-entry tag to 1
                    memory_to_gc_bcells.set_memory_reentry_tag()
                    #Add some memory cells to the naive pool to compete for seeding GCs
                    self.naive_bcells[gc_idx].add_bcells(memory_to_gc_bcells)
                    #self.gc_bcells[gc_idx].add_bcells(memory_to_gc_bcells)

        # Remove the memory GC cells from the memory compartment (since now they are in naive)
        self.memory_gc_bcells.exclude_bcell_fields(memory_to_gc_idx)
        if self.memory_gc_bcells.lineage.size == 0:
            self.memory_gc_bcells.reset_bcell_fields()

    def check_overwrite(self, data: Any, file_path: str) -> None:
        """Write file depending on if file exists and if overwriting is allowed.
        
        Args:
            data: the data to write to file.
            file_path: the path to the file.
        """
        write_fn, file_type = {
            '.pkl': (utils.write_pickle, 'pickle'),
            '.json': (utils.write_json, 'parameters')
        }[file_path.suffix]

        if file_path.exists():
            if self.overwrite:
                print(f'Warning: {file_type} file already exists. Overwriting.')
                write_fn(data, file_path)
            else:
                print(f'{file_type} file already exists. Not overwriting.')
        else:
            write_fn(data, file_path)

    
    def update_history(self) -> None:
        """
        Update the history at the current timepoint.
        """

        def _fill_history_nums(
            history: dict,
            history_idx: int, 
            name: str, 
            bcells: Bcells, 
            gc_idx: int | None=None
        ) -> dict:
            """Fill num_above_aff and num_by_lineage in history.
            
            Args:
                history: Dictionary containing history information.
                history_idx: Timestep index for the history arrays.
                name: Name of the key in history to update.
                bcells: Bcell population to get information from.
                gc_idx: Index of the GC, if updating GC information.

            Returns:
                Updated history dictionary.
            """
            
            num_above_aff = bcells.get_num_above_aff()
            num_in_aff = bcells.get_num_in_aff_bins()
            num_by_lineage = np.histogram(
                bcells.lineage, 
                bins=np.arange(self.n_naive_precursors + 1) + 0.5
            )[0]
            
            if gc_idx:
                history[name]['num_above_aff'][history_idx, gc_idx] = num_above_aff
                history[name]['num_in_aff'][history_idx, gc_idx] = num_in_aff
                history[name]['num_by_lineage'][history_idx, gc_idx] = num_by_lineage

            else:
                history[name]['num_above_aff'][history_idx] = num_above_aff
                history[name]['num_in_aff'][history_idx] = num_in_aff
                history[name]['num_by_lineage'][history_idx] = num_by_lineage

            return history

        # Update history every 1 day 
        # Find appropriate time index in history array
        time_diff = np.abs(np.array(self.history_times) - self.current_time)
        assert self.dt > 1e-5
        if time_diff.min() < 1e-5:
            history_idx = np.argmin(time_diff)
        else:
            return
        
        # naive bcells that enter gc
        for gc_idx, bcells in enumerate(self.naive_cells_entry):
            self.history['gc_entry']['total_num'][history_idx, gc_idx] = self.naive_cells_entry[gc_idx].get_num_entry()
            #reset the naive cell entry count
            self.naive_cells_entry[gc_idx].reset_bcell_fields()
            assert(self.naive_cells_entry[gc_idx].lineage.size == 0)

        # GC bcells
        for gc_idx, bcells in enumerate(self.gc_bcells):
            self.history = _fill_history_nums(
                self.history, history_idx, 'gc', self.gc_bcells[gc_idx], gc_idx
            )

        # Plasma and memory bcells, separated by GC/EGC-derived
        for name in ['plasma', 'memory']:
            for gc_egc in ['gc', 'egc']:
                full_name = f'{name}_{gc_egc}'
                bcells: Bcells = getattr(self, f'{full_name}_bcells')
                self.history = _fill_history_nums(
                    self.history, history_idx, full_name, bcells
                )

        # Concentrations
        self.history['conc']['ic_fdc_conc'][history_idx] = self.concentrations.ic_fdc_conc
        self.history['conc']['ic_fdc_eff_conc'][history_idx] = self.ag_eff_conc
        self.history['conc']['ab_conc'][history_idx] = self.concentrations.ab_conc
        self.history['conc']['ab_ka'][history_idx] = self.concentrations.ab_ka


    def run_timestep(self) -> None:
        """Run a single timestep.
        
        Update the effective concentration of each epitope displayed on FDCs that
        is available to B cells. Then run one round of each of the `n_gcs` GC 
        reactions. Then run one round of a single EGC reaction. Then simulate B cell death
        according to death rates of GC B cells, plasma cells, and memory cells.

        Update concentrations & affinities of antibodies from plasma cells.
        Update history. 
        """
        #update the effective concentration of each epitope displayed on FDCs after masking
        self.ag_eff_conc = self.concentrations.get_eff_ic_fdc_conc() #(n_ep,)

        # Before running GCs, create temporary memory bcells and store exported
        # memory cells in temporary_memory_bcells. At the end, append them to the
        # self.memory_gc_bcells. This is because self.memory_gc_bcells gets big, 
        # and we want to avoid making copies of it over every GC.
        self.temporary_memory_bcells = copy.deepcopy(self.dummy_bcells)
        #run each of the `n_gcs` reactions 
        for gc_idx in range(self.n_gc):
            self.run_gc(gc_idx)
        self.memory_gc_bcells.add_bcells(self.temporary_memory_bcells)
        #re-add some of the exported memory cells into naive pool for GC reentry
        self.split_memory_bcells()
        self.run_egc()

        # B cell death according to death rates
        self.set_death_rates()
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx].kill()
        self.plasma_gc_bcells.kill()
        self.plasma_egc_bcells.kill()
        self.memory_gc_bcells.kill()
        self.memory_egc_bcells.kill()

        # Update antibody/antigen concentrations
        self.concentrations.update_concentrations(
            self.current_time, 
            self.plasma_gc_bcells, 
            self.plasma_egc_bcells
        )

        self.update_history()
    
    def print_bcell_populations(self) -> None:
        """ Print out the status of B cell populations over the course of the simulation."""

        print("Number of plasma GC bcells: " + f"{self.plasma_gc_bcells.lineage.size}")
        print("Number of memory GC bcells: " + f"{self.memory_gc_bcells.lineage.size}")
        print("Number of plasma EGC bcells: " + f"{self.plasma_egc_bcells.lineage.size}")
        print("Number of memory EGC bcells: " + f"{self.memory_egc_bcells.lineage.size}")


    def run(self) -> None:
        """Run the simulation.
        
        Set the random seed, initialize B cell populations, and run dynamics for all 
        timesteps, saving results in ``self.history`` every 1 day. 
        
        Then write out the history and simulation pickle files and the parameter 
        json file.
        """
        start_time = time.perf_counter()

        self.create_populations()

        for timestep_idx in range(self.n_timesteps):
            self.timestep_idx = timestep_idx
            self.current_time = self.timestep_idx * self.dt

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            #print out simulation time every 1 day 
            if np.isclose(self.current_time, round(self.current_time)):
                print_string = (
                    f'Sim time: {self.current_time:.2f}, '
                    f'Wall time: {elapsed_time:.1f}'
                )
                print(print_string) 
            self.run_timestep()

        # Write history pickle file and parameter json files
        os.makedirs(self.data_dir)
        self.check_overwrite(utils.compress(self.history), self.history_path)
        self.check_overwrite(self.get_parameter_dict(), self.parameter_json_path)
        
        # Empty to save space
        self.history = {}
        self.naive_bcells = []
        self.precalculated_dEs = []

        #Write out a pickled Simulation object to freeze end state of simulation
        if self.write_simulation:
            self.check_overwrite(self, self.sim_path)

        print('SIMULATION COMPLETE.')