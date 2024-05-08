import numpy as np
import utils
from parameters import Parameters  # imagining a class that contains all parameters, child classes will have access to all parameters
from bcells import Bcells


class Simulation(Parameters):


    def __init__(self):
        super().__init__()


    def getnumTcells(self) -> np.ndarray:
        tspan = np.arange(0, self.tmax + self.dt, self.dt)

        if self.tmax <= 14:
            num_tcells = self.num_tmax * tspan / 14
        else:
            d14_idx = round(14 / self.dt + 1)
            num_tcells[:d14_idx] = self.num_tmax * tspan[:d14_idx] / 14
            for i in range(d14_idx, len(tspan)):
                num_tcells[i] = num_tcells[i - 1] * np.exp(-self.d_Tfh * self.dt)

        return num_tcells


    def get_naive_bcells_for_one_gc(self) -> Bcells:
        naive_bcells = Bcells() #XXX
        naive_bcells_arr = XXX.get_naive_bcells_arr()
        sigma = XXX.get_sigma()
        randu = np.random.uniform(size=naive_bcells_arr.shape)
        naive_bcells_int = np.floor(naive_bcells_arr + randu).astype(int)

        # Bin at which the value of frequency is 1 for dominant/subdominant cells
        # 6 to 8 with interval of 0.2
        fitness_array = np.linspace(self.f0, self.f0 + 2, self.num_class_bins)

        # Stochastically round up or down the frequency to get integer numbers
        # Indexing takes care of the epitope type
        idx = 0
        for ep in range(self.n_ep):
            for j, fitness in enumerate(fitness_array):
                idx_new = idx + naive_bcells_int[ep, j]
                naive_bcells.lineage[idx: idx_new] = np.arange(idx, idx_new) + 1
                naive_bcells.target_epitope[idx: idx_new] = ep + 1
                # XXX assuming all variants except WT are f0
                naive_bcells.variant_affinities[idx: idx_new, 0] = fitness          # Vax Strain aff
                naive_bcells.variant_affinities[idx: idx_new, 1:self.nvar] = self.f0    # Variant1 aff

                dE = XXX.get_dE(idx_new, idx, sigma, ep)
                for var in range(self.n_var):

                    reshaped = np.reshape(
                        dE[:, var], (idx_new - idx, self.n_res), order='F')

                    naive_bcells.precalculated_affinity_changes[
                        var][idx:idx_new, :] = reshaped

                idx = idx_new
        
        return naive_bcells
    

    def run_gc(self, gc_idx) -> None:
        seeding_bcells_idx = self.naive_bcells[gc_idx].naive_flux_for_one_gc(conc)  # rename naive_flux_for_one_gc to enter_gc_egc
        self.add_bcells(
            self.gc_bcells[gc_idx], self.naive_bcells[gc_idx], seeding_bcells_idx
        )

        if self.seed_with_memory:
            seeding_mem_bcells_idx = self.memory_bcells.naive_flux_for_one_gc(conc)
            self.add_bcells(
                self.gc_bcells[gc_idx], self.memory_bcells, seeding_mem_bcells_idx
            )

        birth_idx = self.gc_bcells[gc_idx].birth(conc, tcell, self.beta_max)
        self.add_bcells(self.gc_bcells[gc_idx], self.gc_bcells[gc_idx], birth_idx) # no mutation yet XXX

        # no export of cells yet

        death_idx = self.gc_bcells[gc_idx].death(self.mu)
        self.gc_bcells[gc_idx].kill(death_idx)


    def run_egc(self) -> None:
        seeding_mem_bcells_idx = self.memory_bcells.enter_gc_egc(conc)
        self.add_bcells(
            self.egc_bcells, self.memory_bcells, seeding_mem_bcells_idx
        )

        birth_idx = self.egc_bcells.birth(conc, tcell_XXX, self.XXX)
        self.add_bcells(self.egc_bcells, self.egc_bcells, birth_idx)  # no need for mutation

        # no export of cells yet

        death_idx = self.egc_bcells.death(self.mu_XXX)  # probably want to combine .death and .kill into one function
        self.egc_bcells.kill(death_idx)


    def run_timestep(self) -> None:
        for gc_idx, _ in enumerate(self.gc_bcells):
            self.run_gc(gc_idx)

        if self.current_time < self.egc_stop_time:
            self.run_egc()

        pc_death_idx = self.plasma_bcells.death(self.mu_pc)
        mem_death_idx = self.memory_bcells.death(self.mu_mem)
        self.plasma_bcells.kill(pc_death_idx)
        self.memory_bcells.kill(mem_death_idx)

        self.update_concentrations(conc)


    def run(self) -> None:
        self.create_bcells()

        for _ in range(self.num_timesteps):
            self.run_timestep()
            self.current_time += self.dt





    

