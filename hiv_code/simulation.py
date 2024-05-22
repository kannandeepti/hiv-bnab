import copy
import dataclasses
import datetime
import os
import time
from typing import Self, Any

import numpy as np

from . import utils
from .parameters import Parameters
from .bcells import Bcells
from .concentrations import Concentrations



class Simulation(Parameters):


    def __init__(self, updated_params_file: str | None=None, parallel_run_idx: int=0):
        """Initialize attributes.
        
        All the parameters from Parameters are included. If updated_params_file is 
        passed, then the parameters are updated from the file.

        All other inherited classes of Parameters need to have updated_params_file
        passed.
        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)

        self.parallel_run_idx = parallel_run_idx
        self.create_file_paths()
        self.concentrations = Concentrations(self.updated_params_file)
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
            egc: Dict containing information about EGCs.
                num_above_aff: # of EGC B cells with affinities greater than
                    affinities in affinities_history.
                    np.ndarray (shape=(n_history_timepoints, n_var, n_ep, n_affinities))
                num_by_lineage: # of EGC B cells in each lineage.
                    np.ndarray (shape=(n_history_timepoints, n_naive_precursors))
            plasma: Similar to egc but with plasma cells.
            memory: Similar to egc but with memory cells.
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
            'egc': {
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
            'conc': {
                'ag_conc': np.zeros((
                    self.n_history_timepoints, 
                    self.n_ep + 1, 
                    self.n_ag, 
                )),
                'ab_conc': np.zeros((
                    self.n_history_timepoints, 
                    self.n_ig_types,
                    self.n_ep,
                )),
                'ab_ka': np.zeros((
                    self.n_history_timepoints, 
                    self.n_var, 
                    self.n_ig_types, 
                    self.n_ep, 
                )),
            },
        }
        self.history['plasma'] = copy.deepcopy(self.history['egc'])
        self.history['memory'] = copy.deepcopy(self.history['egc'])

    
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
        self.data_dir = os.path.join(
            self.experiment_dir, f'{date_time}_{self.parallel_run_idx}'
        )

        self.history_path = os.path.join(self.data_dir, self.history_file_name)
        self.sim_path = os.path.join(self.data_dir, self.simulation_file_name)
        self.parameter_json_path = os.path.join(self.data_dir, self.param_file_name)

        if self.vax_idx > 0:
            self.prev_sim_path = os.path.join(
                self.experiment_dir, 
                self.find_previous_experiment(), 
                self.simulation_file_name
            )


    def get_naive_bcells(self) -> Bcells:
        """Create naive bcells.
        
        The numbers of bcells in each fitness class are read from naive_bcells_arr
        with stochastic rounding. An empty naive cell population is created with
        total number from naive_bcells_int. The bcell field arrays for the naive
        bcells are then filled out. These include:
            the lineage index
            the target epitope
            the variant affinities (note: variants specified in 
                naive_high_affinity_variants will have the fitness above E0, while 
                other variants have affinity E0)
            the precalculated dEs from the multivariate log-normal distribution.
        """
        randu = np.random.uniform(size=self.naive_bcells_arr.shape)
        naive_bcells_int = np.floor(self.naive_bcells_arr + randu).astype(int)
        naive_bcells = Bcells(
            updated_params_file=self.updated_params_file, 
            initial_number=naive_bcells_int.sum()
        )

        idx = 0
        for ep in range(self.n_ep):
            for j, fitness in enumerate(self.fitness_array):
                idx_new = idx + naive_bcells_int[ep, j]
                naive_bcells.lineage[idx: idx_new] = np.arange(idx, idx_new) + 1
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

                    naive_bcells.precalculated_dEs[idx:idx_new, :, var] = np.reshape(
                        dE[:, var], (idx_new - idx, self.n_res), order='F'
                    )

                idx = idx_new
        
        return naive_bcells
    

    def set_death_rates(self) -> None:
        """Set death rates for all bcell populations."""
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx].death_rate = self.bcell_death_rate
        self.egc_bcells.death_rate = self.bcell_death_rate
        self.plasma_bcells.death_rate = utils.get_death_rate_from_half_life(
            self.plasma_half_life, self.dt
        )
        self.memory_bcells.death_rate = utils.get_death_rate_from_half_life(
            self.memory_half_life, self.dt
        )
    

    def create_populations(self) -> None:
        """Create bcell populations.

        Attributes:
            gc_bcells: list of n_gc empty Bcells
            naive_bcells: list of n_gc Bcells created by get_naive_bcells
            egc_bcells: empty Bcells
            plasma_bcells: empty Bcells
            memory_bcells: empty Bcells
            plasmablasts: empty Bcells
        """
        self.dummy_bcells = Bcells(updated_params_file=self.updated_params_file)

        self.gc_bcells = [copy.deepcopy(self.dummy_bcells) for _ in range(self.n_gc)]
        self.naive_bcells = [self.get_naive_bcells() for _ in range(self.n_gc)]

        self.egc_bcells = copy.deepcopy(self.dummy_bcells)
        self.plasma_bcells = copy.deepcopy(self.dummy_bcells)
        self.memory_bcells = copy.deepcopy(self.dummy_bcells)
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
        seeding_bcells.activated_time = np.ones(
            shape=seeding_bcells.activated_time.shape
        ) * (self.current_time + 0.5 * self.dt)

        self.gc_bcells[gc_idx].add_bcells(seeding_bcells)

        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(
            self.ag_eff_conc, self.tcell
        )
        differentiated_bcells = daughter_bcells.differentiate_bcells(
            self.output_prob, self.output_pc_fraction, utils.DerivedCells.GC.value
        )
        memory_bcells, plasma_bcells, nonexported_bcells = differentiated_bcells

        self.temporary_memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.gc_bcells[gc_idx].add_bcells(nonexported_bcells)


    def run_egc(self) -> None:
        """Run the EGC.
        
        No continuous seeding is implemented yet, the EGC is seeded only once by the final memory
        cells from the previous vax. This may change in the future.

        Daughter cells are generated, differentiated, and added to their respective populations.
        In the EGC, all cells are exported and have a higher probability of becoming plasma cells.
        """
        # seeding_bcells = self.memory_bcells.get_seeding_bcells(conc)
        # self.egc_bcells.add_bcells(seeding_bcells)

        daughter_bcells = self.egc_bcells.get_daughter_bcells(
            self.ag_eff_conc, self.tcell
        )
        differentiated_bcells = daughter_bcells.differentiate_bcells(
            self.egc_output_prob, 
            self.egc_output_pc_fraction,
            utils.DerivedCells.EGC.value,
            mutate=False
        )
        memory_bcells, plasma_bcells, nonexported_bcells = differentiated_bcells

        self.memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.egc_bcells.add_bcells(nonexported_bcells)


    def read_checkpoint(self) -> None:
        """Read previous simulation checkpoint, seed GCs and EGC.
        
        If memory_to_gc_fraction > 0, then memory cells seed the GC.
        """
        self: Self = utils.read_pickle(self.prev_sim_path)
        self.reset_history()

        # Split memory into GCs and EGC
        memory_to_gc_idx = utils.get_sample(
            np.arange(self.memory_bcells.lineage.size), 
            p=self.memory_to_gc_fraction
        )

        memory_to_egc_idx = utils.get_other_idx(
            np.arange(self.memory_bcells.lineage.size), 
            memory_to_gc_idx
        )

        # Seed GCs
        memory_size_to_split = memory_to_gc_idx.size // self.n_gc
        memory_to_gc_idxs = np.split(memory_to_gc_idx[:memory_size_to_split], self.n_gc)
        for gc_idx in range(self.n_gc):
            memory_to_gc_bcells = self.memory_bcells.get_bcells_from_idx(
                memory_to_gc_idxs[gc_idx]
            )
            self.gc_bcells[gc_idx].add_bcells(memory_to_gc_bcells)

        # Seed EGC
        memory_to_egc_bcells = self.memory_bcells.get_bcells_from_idx(memory_to_egc_idx)
        self.egc_bcells.add_bcells(memory_to_egc_bcells)


    def check_overwrite(self, data: Any, file_path: str) -> None:
        """Write file depending on if file exists and if overwriting is allowed.
        
        Args:
            data: the data to write to file.
            file_path: the path to the file.
        """
        write_fn, file_type = {
            'pkl': (utils.write_pickle, 'pickle'),
            'json': (utils.write_json, 'parameters')
        }[file_path.split('.')[-1]]

        if os.path.exists(file_path):
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

        history: Dict containing the history of the simulation.
            gc: Dict containing information about GCs.
                num_above_aff: # of GC B cells with affinities greater than
                    affinities in affinities_history.
                    np.ndarray (shape=(n_history_timepoints, n_gc, 
                    n_var, n_ep, n_affinities))
                num_by_lineage: # of GC B cells in each lineage.
                    np.ndarray (shape=(n_history_timepoints, n_gc, n_naive_precursors))
            egc: Dict containing information about EGCs.
                num_above_aff: # of EGC B cells with affinities greater than
                    affinities in affinities_history.
                    np.ndarray (shape=(n_history_timepoints, n_var, n_ep, n_affinities))
                num_by_lineage: # of EGC B cells in each lineage.
                    np.ndarray (shape=(n_history_timepoints, n_naive_precursors))
            plasma: Similar to egc but with plasma cells.
            memory: Similar to egc but with memory cells.
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
        """
        # Find appropriate time index in history array
        time_diff = np.abs(np.array(self.history_times) - self.current_time)
        if time_diff.min() < 1e-5:
            history_idx = np.argmin(time_diff)
        else:
            return

        for gc_idx, bcells in enumerate(self.gc_bcells):
            self.history['gc']['num_above_aff'][
                history_idx, gc_idx] = bcells.get_num_above_aff()
            self.history['gc']['num_by_lineage'][
                history_idx, gc_idx] = np.histogram(
                    bcells.lineage, bins=np.arange(self.n_naive_precursors + 1) + 0.5
                )[0]
            
        for bcell_name in ['egc', 'plasma', 'memory']:
            bcells: Bcells = getattr(self, f'{bcell_name}_bcells')
            self.history[bcell_name]['num_above_aff'][
                history_idx] = bcells.get_num_above_aff()
            self.history[bcell_name]['num_by_lineage'][
                history_idx] = np.histogram(
                    bcells.lineage, bins=np.arange(self.n_naive_precursors + 1) + 0.5
                )[0]
        
        self.history['conc']['ag_conc'][history_idx] = self.concentrations.ag_conc
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
        self.memory_bcells.add_bcells(self.temporary_memory_bcells)

        if self.current_time < self.egc_stop_time:
            self.run_egc()

        # Kill bcells
        # set_birth_rates not necessary here if all bcells have the same birth rate. 
        # also if gc/egc derived cells have different death rates that needs to be adjusted here
        self.set_death_rates()
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx].kill()
        self.egc_bcells.kill()
        self.plasma_bcells.kill()
        self.memory_bcells.kill()

        # Update concentrations
        plasma_bcells_gc = self.plasma_bcells.get_filtered_by_tag(
            utils.DerivedCells.GC.value
        )
        plasma_bcells_egc = self.plasma_bcells.get_filtered_by_tag(
            utils.DerivedCells.EGC.value
        )
        self.concentrations.update_concentrations(
            self.current_time, self.plasmablasts, plasma_bcells_gc, plasma_bcells_egc
        )

        self.update_history()


    def run(self) -> None:
        """Run the simulation.
        
        Set the random seed, create populations either by reading checkpoint or initializing,
        and run dynamics for all timesteps. Then write out the pickle files and 
        parameter json file.
        """
        start_time = time.perf_counter()
        np.random.seed(self.seed)

        if self.vax_idx > 0:
            self.read_checkpoint()
            import pdb; pdb.set_trace()  # XXX todo, fix reading checkpoint issue
        else:
            self.create_populations()

        import pdb; pdb.set_trace()

        for timestep_idx in range(self.n_timesteps):
            self.timestep_idx = timestep_idx
            self.current_time = self.timestep_idx * self.dt

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            print(f'{self.current_time = }', f'{elapsed_time:.1f}') # XXX
            
            self.run_timestep()

        # Write files
        os.makedirs(self.data_dir)
        self.check_overwrite(self.history, self.history_path)
        self.check_overwrite(self.get_parameter_dict(), self.parameter_json_path)
        self.history = {}  # Empty history to save space
        self.check_overwrite(self, self.sim_path)