import numpy as np
import utils
from parameters import Parameters  # imagining a class that contains all parameters, child classes will have access to all parameters
from bcells import Bcells


class Simulation(Parameters):


    def __init__(self):
        super().__init__()


    def get_num_tcells(self) -> np.ndarray:  # todo make part of parameters
        tspan = np.arange(0, self.tmax + self.dt, self.dt)

        if self.tmax <= 14:
            num_tcells = self.num_tmax * tspan / 14
        else:
            d14_idx = round(14 / self.dt + 1)
            num_tcells[:d14_idx] = self.num_tmax * tspan[:d14_idx] / 14
            for i in range(d14_idx, len(tspan)):
                num_tcells[i] = num_tcells[i - 1] * np.exp(-self.d_Tfh * self.dt)

        return num_tcells
    

    def create_bcells(self) -> Bcells: #XXX
        return Bcells()


    def get_naive_bcells(self) -> Bcells:
        naive_bcells = Bcells() #XXX
        naive_bcells_arr = naive_bcells.get_naive_bcells_arr()
        sigma = naive_bcells.get_sigma()
        randu = np.random.uniform(size=naive_bcells_arr.shape)
        naive_bcells_int = np.floor(naive_bcells_arr + randu).astype(int)

        # Bin at which the value of frequency is 1 for dominant/subdominant cells
        # 6 to 8 with interval of 0.2
        fitness_array = np.linspace(self.E0, self.E0 + 2, self.num_class_bins)  # todo make part of parameter class

        # Stochastically round up or down the frequency to get integer numbers
        # Indexing takes care of the epitope type
        idx = 0
        for ep in range(self.n_ep):
            for j, fitness in enumerate(fitness_array):
                idx_new = idx + naive_bcells_int[ep, j]
                naive_bcells.lineage[idx: idx_new] = np.arange(idx, idx_new) + 1
                naive_bcells.target_epitope[idx: idx_new] = ep + 1
                # XXX assuming all variants except WT are E0
                naive_bcells.variant_affinities[idx: idx_new, 0] = fitness          # Vax Strain aff
                naive_bcells.variant_affinities[idx: idx_new, 1:self.nvar] = self.E0    # Variant1 aff

                dE = naive_bcells.get_dE(idx_new, idx, sigma, ep)
                for var in range(self.n_var):

                    reshaped = np.reshape(
                        dE[:, var], (idx_new - idx, self.n_res), order='F')

                    naive_bcells.precalculated_dEs[idx:idx_new, :, var] = reshaped

                idx = idx_new
        
        return naive_bcells
    

    def set_death_rates(self) -> None:
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx].death_rate = self.bcell_death_rate
        self.egc_bcells.death_rate = self.bcell_death_rate
        self.plasma_bcells.death_rate = utils.get_death_rate(self.plasma_half_life)
        self.memory_bcells.death_rate = utils.get_death_rate(self.memory_half_life)
    

    def create_populations(self) -> None:

        self.gc_bcells = [Bcells() for _ in range(self.num_gc)]
        self.naive_bcells = [Bcells() for _ in range(self.num_gc)]
        for gc_idx in range(self.num_gc):
            self.gc_bcells[gc_idx] = self.create_bcells()
            self.naive_bcells[gc_idx] = self.get_naive_bcells()

        self.egc_bcells = self.create_bcells()
        self.plasma_bcells = self.create_bcells()
        self.memory_bcells = self.create_bcells()

        self.set_death_rates()
    

    def run_gc(self, gc_idx: int) -> None:
        seeding_bcells = self.naive_bcells[gc_idx].get_seeding_bcells(conc)
        self.gc_bcells[gc_idx].add_bcells(seeding_bcells)

        if self.seed_with_memory:
            seeding_memory_bcells = self.memory_bcells.get_seeding_bcells(conc)
            self.gc_bcells[gc_idx].add_bcells(seeding_memory_bcells)

        daughter_bcells = self.gc_bcells[gc_idx].get_daughter_bcells(conc, tcell)
        memory_bcells, plasma_bcells, nonexported_bcells = daughter_bcells.divide_bcells()

        self.memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.gc_bcells[gc_idx].add_bcells(nonexported_bcells)

        self.gc_bcells[gc_idx].kill()


    def run_egc(self) -> None:
        seeding_bcells = self.memory_bcells.get_seeding_bcells(conc)
        self.egc_bcells.add_bcells(seeding_bcells)

        daughter_bcells = self.egc_bcells.get_daughter_bcells(conc, tcell)
        memory_bcells, plasma_bcells, nonexported_bcells = daughter_bcells.divide_bcells(mutate=False)

        self.memory_bcells.add_bcells(memory_bcells)
        self.plasma_bcells.add_bcells(plasma_bcells)
        self.egc_bcells.add_bcells(nonexported_bcells)

        self.egc_bcells.kill()


    def run_timestep(self) -> None:
        for gc_idx in range(self.num_gc):
            self.run_gc(gc_idx)

        if self.current_time < self.egc_stop_time:
            self.run_egc()

        # Kill memory and plasma Bcells
        self.plasma_bcells.kill()
        self.memory_bcells.kill()

        self.update_concentrations(conc)


    def run(self) -> None:
        self.create_populations()

        for timestep_idx in range(self.num_timesteps):
            self.run_timestep()
            self.current_time = timestep_idx * self.dt





    

