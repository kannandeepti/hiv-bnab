from typing import Self
import numpy as np
import utils
from parameters import Parameters
from bcells import Bcells



class Simulation(Parameters):


    def __init__(self):
        super().__init__()
        self.create_file_path()

    
    def create_file_path(self) -> None:
        self.file_path = '' #XXX


    def get_naive_bcells(self) -> Bcells:
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
                naive_bcells.target_epitope[idx: idx_new] = ep + 1
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
        seeding_bcells = self.naive_bcells[gc_idx].get_seeding_bcells(conc)
        self.gc_bcells[gc_idx].add_bcells(seeding_bcells)

        if self.memory_to_gc_fraction > 0.:
            seeding_memory_bcells = self.memory_bcells.get_seeding_bcells(conc)
            self.gc_bcells[gc_idx].add_bcells(seeding_memory_bcells)

        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(conc, self.tcell)
        memory_bcells, plasma_bcells, nonexported_bcells = daughter_bcells.divide_bcells(
            utils.DerivedCells.GC.value
        )

        self.memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.gc_bcells[gc_idx].add_bcells(nonexported_bcells)


    def run_egc(self) -> None:
        """EGC is seeded only once by final memory cells from previous shot."""
        # seeding_bcells = self.memory_bcells.get_seeding_bcells(conc)
        # self.egc_bcells.add_bcells(seeding_bcells)

        daughter_bcells = self.egc_bcells.get_daughter_bcells(conc, self.tcell)
        memory_bcells, plasma_bcells, nonexported_bcells = daughter_bcells.divide_bcells(
            utils.DerivedCells.EGC.value, mutate=False
        )

        self.memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.egc_bcells.add_bcells(nonexported_bcells)


    def run_timestep(self, timestep_idx: int) -> None:
        self.tcell = self.num_tcells_arr[timestep_idx]
        for gc_idx in range(self.num_gc):
            self.run_gc(gc_idx)

        if self.current_time < self.egc_stop_time:
            self.run_egc()

        # Kill bcells
        self.set_death_rates()  # set_birth_rates not necessary here if all bcells have the same birth rate. also if gc/egc derived cells have different death rates that needs to be adjusted here
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx].kill()
        self.egc_bcells.kill()
        self.plasma_bcells.kill()
        self.memory_bcells.kill()

        self.update_concentrations(conc)


    def read_checkpoint(self) -> None:
        self: Self = utils.read_pickle(self.file_path)
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx] = Bcells()
        self.egc_bcells = self.memory_bcells


    def run(self) -> None:

        np.random.seed(self.seed)

        if self.vax_idx > 0:
            self.read_checkpoint()
        else:
            self.create_populations()

        for timestep_idx in range(self.num_timesteps):
            self.current_time = timestep_idx * self.dt
            self.run_timestep(timestep_idx)





    

