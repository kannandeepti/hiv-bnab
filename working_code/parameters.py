import dataclasses
import numpy as np


@dataclasses.dataclass
class Parameters:
    """
    Base class that contains simulation parameters.

    Inherited classes will have access to all these parameters.
    """

    seed = 0
    vax_timing = [28, 28, 28]
    vax_idx = 0
    dt = 0.01

    num_gc = 200
    initial_bcell_number = 0
    n_res = 80
    n_var = 3
    n_ep = 3
    n_ag = 1
    mutation_pdf = [3.1, 1.2, 3.08]
    mutation_death_prob = 0.3
    mutation_silent_prob = 0.5  # think Leerang accidently flipped these 2 values
    output_prob = 0.05
    output_pc_fraction = 0.1
    
    gc_entry_birth_rate = 1
    bcell_birth_rate = 2.5
    bcell_death_rate = 0.5
    plasma_half_life = 4
    memory_half_life = np.inf
    memory_to_gc_fraction = 0.
    naive_bcells_num_divide = 2

    egc_stop_time = 6.

    E0 = 6
    E1h = 7
    dE12 = 0.4
    dE13 = 0.4
    class_size = 0.2
    num_naive_precursors = 2000
    num_class_bins = 11
    max_affinity = 16.

    C0 = 0.008

    p2 = 0.15
    p3 = 0.05
    w1 = 0
    w2 = 0.5

    ag_eff = 0.01
    masking = 0
    q12 = 0.
    q13 = 0.
    q23 = 0.
    initial_ka = 1e-3
    ag0 = 10.
    igm0 = 0.01
    delivery_type = 'bolus'
    deposit_rate = 24.
    d_igm = np.log(2) / 28
    d_igg = np.log(2) / 28
    d_ag = 3.
    d_IC = 0.15
    k = 0
    T = 0
    conc_threshold = 1e-10
    delay = 2.
    nm_to_m_conversion = -9
    max_ka = 1e11
    production = 1

    nmax = 10.
    tmax = 360
    num_tmax = 1200
    d_Tfh = 0.01

    r_igm = igm0 * production / num_gc
    r_igg = igm0 * production / num_gc
    fitness_array = list(np.linspace(E0, E0 + 2, num_class_bins))  # 6 to 8 with interval of 0.2  # Bin at which the value of frequency is 1 for dominant/subdominant cells
    num_timesteps = int(vax_timing[vax_idx] / dt)

    # XXX
    if delivery_type == 'bolus':
        F0 = 0
        dose_time = 0
        dose = ag0
    else:
        raise ValueError('Slow delivery not used.')


    @property
    def num_tcells_arr(self) -> np.ndarray:
        tspan = np.arange(0, self.tmax + self.dt, self.dt)

        if self.tmax <= 14:
            num_tcells_arr = self.num_tmax * tspan / 14
        else:
            d14_idx = round(14 / self.dt + 1)
            num_tcells_arr[:d14_idx] = self.num_tmax * tspan[:d14_idx] / 14
            for i in range(d14_idx, len(tspan)):
                num_tcells_arr[i] = num_tcells_arr[i - 1] * np.exp(-self.d_Tfh * self.dt)

        return num_tcells_arr

    @property
    def sigma(self) -> list[np.ndarray]:
        """Get sigma: covariance matrix per epitope."""# XXX
        sigma = [[] for _ in self.n_ep]
        for ep in range(self.n_ep):
            if ep == 0:  # dom
                sigma[ep] = np.array([[1, 0.4, 0.4], [0.4, 1, 0], [0.4, 0, 1]]) #XXX

            else:
                sigma[ep] = np.diag(np.ones(self.n_var))
                for var in range(self.n_var - 1):  # non-WT var
                    # each row corresponds to a different epitope
                    rho = self.rho[ep - 1][var] #XXX
                    sigma[ep][0][var + 1] = rho
                    sigma[ep][var + 1][0] = rho
        return sigma
            