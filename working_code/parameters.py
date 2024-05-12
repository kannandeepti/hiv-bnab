import numpy as np


class Parameters:

    def __init__(self):

        self.seed = 0
        self.vax_timing = [28, 28, 28]
        self.vax_idx = 0
        self.dt = 0.01

        self.num_gc = 200
        self.initial_bcell_number = 0
        self.n_res = 80
        self.n_var = 2
        self.n_ep = 2
        self.mutation_pdf = [3.1, 1.2, 3.08]
        self.mutation_death_prob = 0.3
        self.mutation_silent_prob = 0.5  # think Leerang accidently flipped these 2 values
        self.output_prob = 0.05
        self.output_pc_fraction = 0.1
        
        self.gc_entry_birth_rate = 1
        self.bcell_birth_rate = 2.5
        self.bcell_death_rate = 0.5
        self.plasma_half_life = 4
        self.memory_half_life = np.inf
        self.memory_to_gc_fraction = 0.

        self.egc_stop_time = 6.

        self.E0 = 6
        self.E1h = 7
        self.dE12 = 0.4
        self.class_size = 0.2
        self.num_naive_precursors = 2000
        self.num_class_bins = 11
        self.max_affinity = 16.

        self.C0 = 0.008

        self.p2 = 0.15
        self.p3 = 0.05
        self.w1 = 0
        self.w2 = 0.5

        self.nmax = 10.
        self.tmax = 360
        self.num_tmax = 1200
        self.d_Tfh = 0.01

        self.fitness_array = np.linspace(self.E0, self.E0 + 2, self.num_class_bins)  # 6 to 8 with interval of 0.2  # Bin at which the value of frequency is 1 for dominant/subdominant cells
        self.sigma = self.get_sigma()
        self.num_timesteps = int(self.vax_timing[self.vax_idx] / self.dt)
        self.num_tcells_arr = self.get_num_tcells_arr()


    def get_num_tcells_arr(self) -> np.ndarray:  # todo make part of parameters
        tspan = np.arange(0, self.tmax + self.dt, self.dt)

        if self.tmax <= 14:
            num_tcells_arr = self.num_tmax * tspan / 14
        else:
            d14_idx = round(14 / self.dt + 1)
            num_tcells_arr[:d14_idx] = self.num_tmax * tspan[:d14_idx] / 14
            for i in range(d14_idx, len(tspan)):
                num_tcells_arr[i] = num_tcells_arr[i - 1] * np.exp(-self.d_Tfh * self.dt)

        return num_tcells_arr


    def get_sigma(self) -> list[np.ndarray]:
        """Get sigma: covariance matrix per epitope."""# XXX
        sigma = [[] for _ in self.n_ep]
        for ep in range(self.n_ep):
            if ep == 0:  # dom
                sigma[ep] = np.array([[1, 0.4, 0.4], [0.4, 1, 0], [0.4, 0, 1]]) #XXX

            else:
                sigma[ep] = np.diag(np.ones(self.n_var))
                for var in range(self.n_var - 1):  # non-WT var
                    # each row corresponds to a different epitope
                    rho = self.rho[ep - 1][var]
                    sigma[ep][0][var + 1] = rho
                    sigma[ep][var + 1][0] = rho
        return sigma