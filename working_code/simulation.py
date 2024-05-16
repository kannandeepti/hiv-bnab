import dataclasses
import os
import time
from typing import Self, Any
import numpy as np
import utils
from parameters import Parameters
from bcells import Bcells
from concentrations import Concentrations



class Simulation(Parameters):


    def __init__(self):
        """Initialize attributes.
        
        All the parameters from Parameters are included. file_paths and concentrations
        are specific to Simulation and are created as well.
        """
        super().__init__()
        self.create_file_paths()
        self.concentrations = Concentrations()

    
    def get_parameter_dict(self) -> None:
        """Write parameters to json file.

        Properties from Parameter class are not included.
        
        XXX still need to adjust this to write non-default parameters.
        """
        parameters = Parameters()
        return {
            field.name: getattr(parameters, field.name)
            for field in dataclasses.fields(parameters)
        }

    
    def create_file_paths(self) -> None:
        """Create path attributes.
        
        data_dir: path to the directory for a particular experiment
        prev_file_path: path to the pickle file for the previous vax
        file_path: path to the pickle file for the current vax
        """
        self.data_dir = '' # XXX
        self.prev_pickle_path = '' # XXX
        self.pickle_path = '' #XXX
        self.parameter_json_path = os.path.join(self.data_dir, 'parameters.json')


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
        naive_bcells = Bcells(initial_number=naive_bcells_int.sum())

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
        """

        self.gc_bcells = [Bcells() for _ in range(self.n_gc)]
        self.naive_bcells = [Bcells() for _ in range(self.n_gc)]
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx] = Bcells()
            self.naive_bcells[gc_idx] = self.get_naive_bcells()

        self.plasmablasts = Bcells()
        self.egc_bcells = Bcells()
        self.plasma_bcells = Bcells()
        self.memory_bcells = Bcells()

        self.set_death_rates()
    

    def run_gc(self, gc_idx: int) -> None:
        """Run a single GC.

        GCs are seeded from naive bcells, and the naive bcells divide naive_bcells_n_divide
        times. If memory_to_gc_fraction is set, then memory cells are also seeding the GC.

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

        if self.memory_to_gc_fraction > 0.:  # XXX adjust this so that we can control how many memory cells are added based on memory_to_gc_fraction
            seeding_memory_bcells = self.memory_bcells.get_seeding_bcells(
                self.ag_eff_conc
            )
            self.gc_bcells[gc_idx].add_bcells(seeding_memory_bcells)

        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(
            self.ag_eff_conc, self.tcell
        )
        differentiated_bcells = daughter_bcells.differentiate_bcells(
            self.output_prob, self.output_pc_fraction, utils.DerivedCells.GC.value
        )
        memory_bcells, plasma_bcells, nonexported_bcells = differentiated_bcells

        self.memory_bcells.add_bcells(memory_bcells)
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


    def run_timestep(self, timestep_idx: int) -> None:
        """Run a single timestep.
        
        Calculate the number of tcells and effective Ag concentration. Run the GCs and EGC.
        Run death phase for all bcell populations. Update the concentrations using plasma cells.
        """
        self.tcell: float = self.n_tcells_arr[timestep_idx]
        masked_ag_conc = self.concentrations.get_masked_ag_conc()
        self.ag_eff_conc = (
            np.array([self.ag_eff, 1]) @ 
            masked_ag_conc.transpose((1,0,2))
        )

        for gc_idx in range(self.n_gc):
            self.run_gc(gc_idx)

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

        plasma_bcells_gc = self.plasma_bcells.get_filtered_by_tag(
            utils.DerivedCells.GC
        )
        plasma_bcells_egc = self.plasma_bcells.get_filtered_by_tag(
            utils.DerivedCells.EGC
        )
        self.concentrations.update_concentrations(
            self.current_time, self.plasmablasts, plasma_bcells_gc, plasma_bcells_egc
        )


    def read_checkpoint(self) -> None:
        """Read previous simulation checkpoint, reset GC bcells, and set EGC bcells to memory cells."""
        self: Self = utils.read_pickle(self.prev_pickle_path)
        for gc_idx in range(self.n_gc):
            self.gc_bcells[gc_idx] = Bcells()  # XXX To be like Leerang's code, I think we should reset the GC bcells and add the non-egc bcells here
        self.egc_bcells = self.memory_bcells  # XXX make changes so we can control what fraction of memory cells become egc bcells.


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


    def run(self) -> None:
        """Run the simulation.
        
        Set the random seed, create populations either by reading checkpoint or initializing,
        and run dynamics for all timesteps. Then write out the simulation pickle file and 
        parameter json file.
        """
        start_time = time.perf_counter()
        np.random.seed(self.seed)

        if self.vax_idx > 0:
            self.read_checkpoint()
        else:
            self.create_populations()

        for timestep_idx in range(self.n_timesteps):
            self.current_time = timestep_idx * self.dt

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            print(f'{self.current_time = }', f'{elapsed_time:.1f}') # XXX
            
            self.run_timestep(timestep_idx)

        # Write files
        self.check_overwrite(self, self.pickle_path)
        self.check_overwrite(self.get_parameter_dict(), self.parameter_json_path)





    

