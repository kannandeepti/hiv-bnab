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
    """Class for running the simulation.

    All the parameters from Parameters are included.

    Attributes:
        precalculated_dEs: Precalculated affinity changes for all cells
            to all variants. This is stored in Simulation so that it doesn't
            need to be copied to the Bcell class, which saves a lot of memory
            and a bit of time.
            np.ndarray (shape=(n_gc, n_cell, n_res, n_var)).
        attributes_to_replace: List of attributes that need to be replaced
            when reading a previous checkpoint pickle file. This will update
            the simulation to the correct state.
        history: Dict containing the history of the simulation. See reset_history.
    """

    def __init__(self, updated_params_file: str | None=None, parallel_run_idx: int=0):
        """Initialize attributes.
        
        All the parameters from Parameters are included. If updated_params_file is 
        passed, then the parameters are updated from the file.

        All other inherited classes of Parameters need to have updated_params_file
        passed and set the np.random.seed.
        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)
        np.random.seed(self.seed)

        self.concentrations = Concentrations(self.updated_params_file)
        self.set_naive_bcells_per_bin()
        self.precalculated_dEs = np.zeros(
            (self.n_gc, self.naive_bcells_int.sum(), self.n_res, self.n_var)
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
        """
        history: Dict containing the history of the simulation.
            gc: Dict containing information about GCs.
                num_above_aff: # of GC B cells with affinities greater than
                    affinities in affinities_history.
                    np.ndarray (shape=(n_history_timepoints, n_gc, 
                    n_var, n_ep, n_affinities))
                num_by_lineage: # of GC B cells in each lineage.
                    np.ndarray (shape=(n_history_timepoints, n_gc, n_naive_precursors))
            conc: Dict containing information about concentrations.
                ag_conc: Ag conc array. Similar to Concentrations.ag_conc
                    but with a time dimension.
                    np.ndarray (shape=(n_history_timepoints, n_ep+1, n_ag))
                ab_conc: Ab conc array. Similar to Concentrations.ab_conc
                    but with a time dimension.
                    np.ndarray (shape=(n_history_timepoints, n_ig_types, n_ep))
                ab_ka: Kas to each variant. Similar to Concentrations.ab_ka
                    but with a time dimension.
                    np.ndarray (shape=(n_history_timepoints, n_ig_types, n_ep))
            plasma_gc: Dict containing information about plasma cells from GC.
                num_above_aff: # of EGC B cells with affinities greater than
                    affinities in affinities_history.
                    np.ndarray (shape=(n_history_timepoints, n_var, n_ep, n_affinities))
                num_by_lineage: # of EGC B cells in each lineage.
                    np.ndarray (shape=(n_history_timepoints, n_naive_precursors))
            plasma_egc: Similar to plasma_gc but with plasma cells from EGC.
            memory_gc: Similar to plasma_gc but with memory cells from GC.
            memory_egc: Similar to plasma_gc but with memory cells from EGC.
        """

        self.history = {
            'gc': {
                'num_above_aff': np.zeros((
                    self.n_history_timepoints, 
                    self.n_gc,
                    self.n_var,
                    self.n_ep, 
                    len(self.affinities_history)
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
                    self.n_var,
                    self.n_ep, 
                    len(self.affinities_history)
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
        """Create path attributes. Experiments are labeled using their time.
        
        data_dir: path to the directory for a particular experiment
        prev_sim_path: path to the sim file for the previous vax
        sim_path: path to the sim file for the current vax
        history_path: path to the history pickle file
        """
        date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.data_dir = Path(self.experiment_dir)/f'{date_time}_{self.parallel_run_idx}'

        self.history_path = self.data_dir / self.history_file_name
        self.sim_path = self.data_dir / self.simulation_file_name
        self.parameter_json_path = self.data_dir / self.param_file_name

        if self.vax_idx > 0:
            self.prev_sim_path = os.path.join(
                self.experiment_dir, 
                self.find_previous_experiment(), 
                self.simulation_file_name
            )
    
    def set_naive_bcells_per_bin(self) -> None:
        """Set the number of naive bcells per fitness bin, only integers allowed."""
        randu = np.random.uniform(size=self.naive_bcells_arr.shape)
        self.naive_bcells_int = np.floor(self.naive_bcells_arr + randu).astype(int)


    def get_naive_bcells(self, gc_idx: int) -> Bcells:
        """Create naive bcells.
        
        The numbers of bcells in each fitness class are read from naive_bcells_arr
        with stochastic rounding. An empty naive cell population is created with
        total number from naive_bcells_int. The bcell field arrays for the naive
        bcells are then filled out. These include:
            the gc lineage index
            the lineage index
            the target epitope
            the variant affinities (note: variants specified in 
                naive_high_affinity_variants will have the fitness above E0, while 
                other variants have affinity E0)
            the precalculated dEs from the multivariate log-normal distribution.

        Args:
            gc_idx: The index of the GC.

        Returns:
            naive_bcells: The naive bcell population.
        """
        naive_bcells = Bcells(
            updated_params_file=self.updated_params_file, 
            initial_number=self.naive_bcells_int.sum()
        )

        idx = 0
        for ep in range(self.n_ep):
            for j, fitness in enumerate(self.fitness_array):

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
                        np.arange(self.n_var), 
                        np.array(self.naive_high_affinity_variants)
                    )
                ] = self.E0

                dE = naive_bcells.get_dE(idx_new, idx, ep)
                for var in range(self.n_var):

                    self.precalculated_dEs[gc_idx, idx:idx_new, :, var] = np.reshape(
                        dE[:, var], (idx_new - idx, self.n_res), order='F'
                    )

                idx = idx_new
        
        return naive_bcells
    

    def set_death_rates(self) -> None:
        """Set death rates for all bcell populations."""
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
        """Create bcell populations.

        Attributes:
            gc_bcells: list of n_gc empty Bcells
            naive_bcells: list of n_gc Bcells created by get_naive_bcells
            egc_bcells: empty Bcells
            plasma_gc_bcells: empty Bcells
            memory_gc_bcells: empty Bcells
            plasma_egc_bcells: empty Bcells
            memory_egc_bcells: empty Bcells
            plasmablasts: empty Bcells
        """
        self.dummy_bcells = Bcells(updated_params_file=self.updated_params_file)

        self.gc_bcells = [copy.deepcopy(self.dummy_bcells) for _ in range(self.n_gc)]
        self.naive_bcells = [self.get_naive_bcells(gc_idx) for gc_idx in range(self.n_gc)]

        self.plasma_gc_bcells = copy.deepcopy(self.dummy_bcells)
        self.plasma_egc_bcells = copy.deepcopy(self.dummy_bcells)
        self.memory_gc_bcells = copy.deepcopy(self.dummy_bcells)
        self.memory_egc_bcells = copy.deepcopy(self.dummy_bcells)
        self.plasmablasts = copy.deepcopy(self.dummy_bcells) # Doing nothing right now.

        self.set_death_rates()
    

    def run_gc(self, gc_idx: int) -> None:
        """Run a single GC.

        GCs are seeded from naive bcells, and the naive bcells divide naive_bcells_n_divide
        times.

        Daughter cells are generated based on effective Ag concentration and current tcells,
        differentiated into memory cells, plasma cells, nonexported cells, and added to their 
        respective populations.
        
        Args:
            gc_idx: index of the GC
        """
        seeding_bcells = self.naive_bcells[gc_idx].get_seeding_bcells(self.ag_eff_conc)
        for _ in range(self.naive_bcells_n_divide):
            seeding_bcells.add_bcells(seeding_bcells)

        # Set activated time
        seeding_bcells.set_activated_time(self.current_time)

        # Seed
        self.gc_bcells[gc_idx].add_bcells(seeding_bcells)
        
        # Birth and differentiation
        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(
            self.ag_eff_conc, self.tcell
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
        """Run the EGC.
        
        No continuous seeding is implemented yet, the EGC is seeded only once by the final memory
        cells from the previous vax. This may change in the future.

        Daughter cells are generated, differentiated, and added to their respective populations.
        In the EGC, all cells are exported and have a higher probability of becoming plasma cells.

        tcell is constant at the maximum value in the EGC, and multiplied by n_gc like Leerang does.
        """
        seeding_bcells = self.memory_gc_bcells.get_seeding_bcells(self.ag_eff_conc)

        # Set activated time
        seeding_bcells.set_activated_time(self.current_time)

        # Seed
        self.memory_egc_bcells.add_bcells(seeding_bcells)

        daughter_bcells = self.memory_egc_bcells.get_daughter_bcells(
            self.ag_eff_conc, self.tcell #Leerang had self.nmax * self.n_gc
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

        if np.isclose(self.current_time, round(self.current_time)):
            self.print_bcell_populations()


    def read_checkpoint(self) -> None:
        """Read previous simulation checkpoint, seed GCs and EGC.

        memory_gc_bcells will contain all memory cells.
        
        If memory_to_gc_fraction > 0, then memory cells seed the GC.
        """
        # Read checkpoint and replace attributes
        old_sim: Self = utils.read_pickle(self.prev_sim_path)
        for attribute in self.attributes_to_replace:
            setattr(self, attribute, getattr(old_sim, attribute))
        self.reset_history()

        # Create naive bcells and dEs
        self.naive_bcells = [
           self.get_naive_bcells(gc_idx) for gc_idx in range(self.n_gc)
        ]


        # If True, reset GCs and EGC
        if self.reset_gc_egcs:
            for gc_idx in range(self.n_gc):
                self.gc_bcells[gc_idx] = copy.deepcopy(self.dummy_bcells)
            self.memory_egc_bcells = copy.deepcopy(self.dummy_bcells)

        # Reset activated times
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx].set_activated_time(0, shift=False)
        self.plasma_gc_bcells.set_activated_time(0, shift=False)
        self.plasma_egc_bcells.set_activated_time(0, shift=False)
        self.memory_gc_bcells.set_activated_time(0, shift=False)
        self.memory_egc_bcells.set_activated_time(0, shift=False)

        # Split memory into GCs and EGC
        memory_to_gc_idx = utils.get_sample(
            np.arange(self.memory_gc_bcells.lineage.size), 
            p=self.memory_to_gc_fraction,
        )

        memory_to_egc_idx = utils.get_other_idx(
            np.arange(self.memory_gc_bcells.lineage.size), 
            memory_to_gc_idx
        )

        # Seed GCs
        memory_size_to_split = memory_to_gc_idx.size // self.n_gc
        memory_to_gc_idxs = np.split(memory_to_gc_idx[:memory_size_to_split], self.n_gc)
        for gc_idx in range(self.n_gc):
            memory_to_gc_bcells = self.memory_gc_bcells.get_bcells_from_idx(
                memory_to_gc_idxs[gc_idx]
            )
            self.gc_bcells[gc_idx].add_bcells(memory_to_gc_bcells)

        # Seed EGC
        memory_to_egc_bcells = self.memory_gc_bcells.get_bcells_from_idx(
            memory_to_egc_idx
        )
        self.memory_egc_bcells.add_bcells(memory_to_egc_bcells)


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
            num_by_lineage = np.histogram(
                bcells.lineage, 
                bins=np.arange(self.n_naive_precursors + 1) + 0.5
            )[0]
            
            if gc_idx:
                history[name]['num_above_aff'][history_idx, gc_idx] = num_above_aff
                history[name]['num_by_lineage'][history_idx, gc_idx] = num_by_lineage

            else:
                history[name]['num_above_aff'][history_idx] = num_above_aff
                history[name]['num_by_lineage'][history_idx] = num_by_lineage

            return history

        # Find appropriate time index in history array
        time_diff = np.abs(np.array(self.history_times) - self.current_time)
        assert self.dt > 1e-5
        if time_diff.min() < 1e-5:
            history_idx = np.argmin(time_diff)
        else:
            return

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
        self.history['conc']['ab_conc'][history_idx] = self.concentrations.ab_conc
        self.history['conc']['ab_ka'][history_idx] = self.concentrations.ab_ka


    def run_timestep(self) -> None:
        """Run a single timestep.
        
        Calculate the number of tcells and effective Ag concentration. Run the GCs and EGC.
        Run death phase for all bcell populations. Update the concentrations using plasma cells.
        Update the history dictionary.
        """
        self.tcell: float = self.n_tcells_arr[self.timestep_idx]
        self.ag_eff_conc = self.concentrations.get_eff_ic_fdc_conc() #(n_ep,)

        # Before running GCs, create temporary memory bcells and store exported
        # memory cells in temporary_memory_bcells. At the end, append them to the
        # memory_bcells. This is because memory_bcells gets big, and we want to avoid
        # making copies of it over every GC.
        self.temporary_memory_bcells = copy.deepcopy(self.dummy_bcells)
        for gc_idx in range(self.n_gc):
            self.run_gc(gc_idx)
        self.memory_gc_bcells.add_bcells(self.temporary_memory_bcells)

        if np.isclose(self.current_time, round(self.current_time)):
            self.print_bcell_populations()
            
        if self.current_time < self.egc_stop_time or self.persistent_infection:
            self.run_egc()

        # Kill bcells
        self.set_death_rates()
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx].kill()
        self.plasma_gc_bcells.kill()
        self.plasma_egc_bcells.kill()
        self.memory_gc_bcells.kill()
        self.memory_egc_bcells.kill()

        # Update concentrations
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
        
        Set the random seed, create populations either by reading checkpoint or initializing,
        and run dynamics for all timesteps. Then write out the pickle files and 
        parameter json file.
        """
        start_time = time.perf_counter()

        if self.vax_idx > 0:
            self.read_checkpoint()
        else:
            self.create_populations()

        # Keeping this here so we can verify that dEs are the same across doses.
        print(f'{self.precalculated_dEs[0, 0, 0, 0] = }')

        for timestep_idx in range(self.n_timesteps):
            self.timestep_idx = timestep_idx
            self.current_time = self.timestep_idx * self.dt

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            if np.isclose(self.current_time, round(self.current_time)):
                print_string = (
                    f'Sim time: {self.current_time:.2f}, '
                    f'Wall time: {elapsed_time:.1f}'
                )
                print(print_string) 
            self.run_timestep()

        # memory_gc_bcells will contain all memory cells to read later.
        # self.memory_gc_bcells.add_bcells(self.memory_egc_bcells)

        # Write files
        os.makedirs(self.data_dir)
        self.check_overwrite(utils.compress(self.history), self.history_path)
        self.check_overwrite(self.get_parameter_dict(), self.parameter_json_path)
        
        # Empty to save space
        self.history = {}
        self.naive_bcells = []
        self.precalculated_dEs = []
        self.check_overwrite(self, self.sim_path)

        print('SIMULATION COMPLETE.')