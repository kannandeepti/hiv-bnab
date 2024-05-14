import dataclasses
import os
from typing import Self
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

    
    def write_param_json(self) -> None:
        """Write parameters to json file.
        
        XXX still need to adjust this to write non-default parameters.
        """
        parameters = Parameters()
        parameter_dict = {
            field.name: getattr(parameters, field.name)
            for field in dataclasses.fields(parameters)
        }
        utils.write_json(parameter_dict, os.path.join(self.data_dir, 'parameters.json'))

    
    def create_file_paths(self) -> None:
        """Create path attributes.
        
        data_dir: path to the directory for a particular experiment
        prev_file_path: path to the pickle file for the previous vax
        file_path: path to the pickle file for the current vax
        """
        self.data_dir = '' # XXX
        self.prev_file_path = '' # XXX
        self.file_path = '' #XXX


    def get_naive_bcells(self) -> Bcells:
        """Create naive bcells."""
        naive_bcells = Bcells() #XXX
        naive_bcells_arr = naive_bcells.get_naive_bcells_arr()
        randu = np.random.uniform(size=naive_bcells_arr.shape)
        naive_bcells_int = np.floor(naive_bcells_arr + randu).astype(int)

        # Stochastically round up or down the frequency to get integer numbers
        # Indexing takes care of the epitope type
        idx = 0
        for ep in range(self.n_ep):
            for j, fitness in enumerate(self.fitness_array):
                idx_new = idx + naive_bcells_int[ep, j]
                naive_bcells.lineage[idx: idx_new] = np.arange(idx, idx_new) + 1
                naive_bcells.target_epitope[idx: idx_new] = ep
                # XXX assuming all variants except WT are E0
                naive_bcells.variant_affinities[idx: idx_new, 0] = fitness          # Vax Strain aff
                naive_bcells.variant_affinities[idx: idx_new, 1:self.n_var] = self.E0    # Variant1 aff

                dE = naive_bcells.get_dE(idx_new, idx, ep)
                for var in range(self.n_var):

                    naive_bcells.precalculated_dEs[idx:idx_new, :, var] = np.reshape(
                        dE[:, var], (idx_new - idx, self.n_res), order='F'
                    )

                idx = idx_new
        
        return naive_bcells
    

    def set_death_rates(self) -> None:
        """Set death rates for all bcell populations."""
        for gc_idx in range(self.num_gc):
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
            gc_bcells: list of num_gc empty Bcells
            naive_bcells: list of num_gc Bcells created by get_naive_bcells
            egc_bcells: empty Bcells
            plasma_bcells: empty Bcells
            memory_bcells: empty Bcells
        """

        self.gc_bcells = [Bcells() for _ in range(self.num_gc)]
        self.naive_bcells = [Bcells() for _ in range(self.num_gc)]
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx] = Bcells()
            self.naive_bcells[gc_idx] = self.get_naive_bcells()

        self.egc_bcells = Bcells()
        self.plasma_bcells = Bcells()
        self.memory_bcells = Bcells()

        self.set_death_rates()
    

    def run_gc(self, gc_idx: int) -> None:
        """Run a single GC.

        GCs are seeded from naive bcells, and the naive bcells divide naive_bcells_num_divide
        times. If memory_to_gc_fraction is set, then memory cells are also seeding the GC.

        Daughter cells are generated based on effective Ag concentration and current tcells,
        differentiated into memory cells, plasma cells, nonexported cells, and added to their 
        respective populations.
        
        Args:
            gc_idx: index of the GC
        """
        seeding_bcells = self.naive_bcells[gc_idx].get_seeding_bcells(self.ag_eff_conc)
        for _ in range(self.naive_bcells_num_divide):
            seeding_bcells.add_bcells(seeding_bcells)
        self.gc_bcells[gc_idx].add_bcells(seeding_bcells)

        if self.memory_to_gc_fraction > 0.:  # XXX adjust this so that we can control how many memory cells are added based on memory_to_gc_fraction
            seeding_memory_bcells = self.memory_bcells.get_seeding_bcells(self.ag_eff_conc)
            self.gc_bcells[gc_idx].add_bcells(seeding_memory_bcells)

        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(self.ag_eff_conc, self.tcell)
        memory_bcells, plasma_bcells, nonexported_bcells = daughter_bcells.differentiate_bcells(
            self.output_prob, self.output_pc_fraction, utils.DerivedCells.GC.value
        )

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

        daughter_bcells = self.egc_bcells.get_daughter_bcells(self.ag_eff_conc, self.tcell)
        memory_bcells, plasma_bcells, nonexported_bcells = daughter_bcells.differentiate_bcells(
            self.egc_output_prob, 
            self.egc_output_pc_fraction,
            utils.DerivedCells.EGC.value,
            mutate=False
        )

        self.memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.egc_bcells.add_bcells(nonexported_bcells)


    def run_timestep(self, timestep_idx: int) -> None:
        """Run a single timestep.
        
        Calculate the number of tcells and effective Ag concentration. Run the GCs and EGC.
        Run death phase for all bcell populations. Update the concentrations using plasma cells.
        """
        self.tcell: float = self.num_tcells_arr[timestep_idx]
        masked_ag_conc = self.concentrations.get_masked_ag_conc()
        self.ag_eff_conc = np.array([self.ag_eff, 1]) @ masked_ag_conc

        for gc_idx in range(self.num_gc):
            self.run_gc(gc_idx)

        if self.current_time < self.egc_stop_time:
            self.run_egc()

        # Kill bcells
        # set_birth_rates not necessary here if all bcells have the same birth rate. 
        # also if gc/egc derived cells have different death rates that needs to be adjusted here
        self.set_death_rates()
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx].kill()
        self.egc_bcells.kill()
        self.plasma_bcells.kill()
        self.memory_bcells.kill()

        plasma_bcells_gc = self.plasma_bcells.get_filtered_by_tag(utils.DerivedCells.GC)
        plasma_bcells_egc = self.plasma_bcells.get_filtered_by_tag(utils.DerivedCells.EGC)
        self.concentrations.update_concentrations(self.current_time, plasma_bcells_gc, plasma_bcells_egc)


    def read_checkpoint(self) -> None:
        """Read previous simulation checkpoint, reset GC bcells, and set EGC bcells to memory cells."""
        self: Self = utils.read_pickle(self.prev_file_path)
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx] = Bcells()  # XXX To be like Leerang's code, I think we should reset the GC bcells and add the non-egc bcells here
        self.egc_bcells = self.memory_bcells  # XXX make changes so we can control what fraction of memory cells become egc bcells.


    def run(self) -> None:
        """Run the simulation.
        
        Set the random seed, create populations either by reading checkpoint or initializing,
        and run dynamics for all timesteps. Then write out the simulation pickle file and 
        parameter json file.
        """

        np.random.seed(self.seed)

        if self.vax_idx > 0:
            self.read_checkpoint()
        else:
            self.create_populations()

        for timestep_idx in range(self.num_timesteps):
            self.current_time = timestep_idx * self.dt
            self.run_timestep(timestep_idx)

        utils.write_pickle(self, self.file_path)
        self.write_param_json()





    

