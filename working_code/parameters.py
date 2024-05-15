from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    """
    Base dataclass that contains simulation parameters.

    Inherited classes will have access to all these parameters.
    """

    overwrite: bool = True
    """Whether to overwrite existing files."""

    seed: int = 0
    """
    The random seed for reproducibility.
    """
    
    dt: float = 0.01
    """
    The time step for simulations, in days.
    """
    
    epsilon: float = 1e-11
    """
    A very small value used to avoid divide-by-0 errors.
    """
    
    n_gc: int = 200
    """
    The number of germinal centers.
    """
    
    initial_bcell_number: int = 1
    """
    The initial number of B cells.
    """
    
    n_res: int = 80
    """
    The number of residues.
    """
    
    n_var: int = 3
    """
    The number of variants.
    """

    n_ag: int = 1
    assert n_ag < n_var
    """
    The number of antigens.
    """
    
    n_ep: int = 3
    """
    The number of epitopes.
    """
    
    mutation_death_prob: float = 0.3
    """
    The probability of death due to mutation.
    """
    
    mutation_silent_prob: float = 0.5
    """
    The probability of silent mutation.
    """
    
    output_prob: float = 0.05
    """
    The probability of exporting a birthed cell.
    """
    
    output_pc_fraction: float = 0.1
    """
    The fraction of exported cells that are plasma cells.
    """
    
    egc_output_prob: float = 1.
    """
    The probability of export for EGCs.
    """
    
    egc_output_pc_fraction: float = 0.6
    """
    The fraction of exported cells that are plasma cells for EGCs.
    """
    
    gc_entry_birth_rate: float = 1
    """
    The birth rate of GC entry.
    """
    
    bcell_birth_rate: float = 2.5
    """
    The birth rate of B cells.
    """
    
    bcell_death_rate: float = 0.5
    """
    The death rate of B cells.
    """
    
    plasma_half_life: float = 4
    """
    The half-life of plasma cells.
    """
    
    memory_half_life: float = float('inf')
    """
    The half-life of memory cells.
    """
    
    memory_to_gc_fraction: float = 0.
    """
    The fraction of memory cells to germinal centers.
    """
    
    naive_bcells_n_divide: int = 2
    """
    The number of times a naive cell divides upon seeding a GC.
    """
    
    mutation_pdf: tuple[float] = (3.1, 1.2, 3.08)
    """
    The PDF parameters for mutation dE.
    """
    
    E0: float = 6
    """
    Initial energy level.
    """
    
    E1h: float = 7
    """
    Energy level. XXX
    """
    
    dE12: float = 0.4
    """
    Energy level change.
    """
    
    dE13: float = 0.4
    """
    Energy level change.
    """
    
    n_class_bins: int = 11
    """
    The number of class bins.
    """

    fitness_array: tuple[float] = tuple(
        np.linspace(E0, E0 + 2, n_class_bins)
    )
    """6 to 8 with interval of 0.2"""

    class_size = fitness_array[1] - fitness_array[0]
    """
    The size of a bin in fitness_array
    """
    
    n_naive_precursors: int = 2000
    """
    The number of naive precursors.
    """
    
    max_affinity: float = 16.
    """
    The maximum affinity.
    """
    
    C0: float = 0.008
    """
    Value to normalize concentrations.
    """
    
    p2: float = 0.15
    """
    Parameter value. XXX
    """
    
    p3: float = 0.05
    """
    Parameter value. XXX
    """
    
    w1: float = 0
    """
    Parameter for Ag capture saturation, "alternative Ag capture model".
    """
    
    w2: float = 0.5
    """
    Stringency of Ag capture.
    """
    
    ag_eff: float = 0.01
    """
    XXX
    """
    
    masking: int = 0
    assert masking in [0, 1]
    """
    Whether to use epitope masking.
    """
    
    initial_ka: float = 1e-3
    """
    Initial ka value.
    """
    
    igm0: float = 0.01
    """
    Initial IgM concentration (nm-1).
    """
    
    deposit_rate: float = 24.
    """
    The deposit rate (units XXX).
    """
    
    d_igm: float = np.log(2) / 28
    """
    Decay rate of IgM. XXX
    """
    
    d_igg: float = np.log(2) / 28
    """
    Decay rate of IgG. XXX
    """
    
    d_ag: float = 3.
    """
    Decay rate of antigen. XXX
    """
    
    d_IC: float = 0.15
    """
    Decay rate of immune complexes. XXX
    """
    
    conc_threshold: float = 1e-10
    """
    Concentration threshold, below which it is set to 0.
    """
    
    delay: float = 2.
    """
    Time delay for producing Abs.
    """
    
    nm_to_m_conversion: int = -9
    """
    Conversion factor from nM to M.
    """
    
    max_ka: float = 1e11
    """
    Maximum ka value.
    """
    
    production: int = 1
    """
    Production XXX.
    """
    
    nmax: float = 10.
    """
    Tcell amount for seeding GCs.
    """
    
    tmax: int = 360
    """
    . XXX
    """
    
    n_tmax: int = 1200
    """
    . XXX
    """
    
    d_Tfh: float = 0.01
    """
    Rate of change of Tfh.
    """
    
    r_igm: float = igm0 * production / n_gc
    """
    Rate of IgM production.
    """
    
    r_igg: float = igm0 * production / n_gc
    """
    Rate of IgG production.
    """

    ########################################
    # Specific to bolus doses.
    ########################################

    vax_timing: tuple[int] = (28, 28, 28)
    """
    The timing of vaccinations, in days.
    """
    
    vax_idx: int = 0
    """
    The index of the current vaccination timing.
    """
    
    k: int = 0
    """
    Value of k. XXX
    """
    
    T: int = 0
    """
    Value of T. XXX
    """

    ag0: float = 10.
    """
    Initial antigen concentration (nM-1).
    """

    F0: int =  0
    """XXX"""

    dose_time: float = 0.
    """Time to give the vaccine dose."""

    dose: float = ag0
    """Concentration of Ag in the dose."""

    egc_stop_time: float = 6.
    """
    The time to stop EGCs.
    """

    n_timesteps = int(vax_timing[vax_idx] / dt)
    """Number of timesteps for this simulation."""
    
    ################################################################
    # Properties are not included when calling dataclasses.fields.
    ################################################################
    
    @property
    def overlap_matrix(self) -> None:
        """Defines the epitope overlap matrix."""
        overlap_matrix = np.eye(self.n_ep)
        # XXX adjust if there is some overlap
        return overlap_matrix
    

    @property
    def rho(self) -> list[list[float]]:
        """Using a property in case it needs to be more complicated later."""
        return [
            [0.95, 0.4], 
            [0.4, 0.95]
        ]


    @property
    def n_tcells_arr(self) -> np.ndarray:
        tspan = np.arange(0, self.tmax + self.dt, self.dt)

        if self.tmax <= 14:
            n_tcells_arr = self.n_tmax * tspan / 14
        else:
            d14_idx = round(14 / self.dt + 1)
            n_tcells_arr[:d14_idx] = self.n_tmax * tspan[:d14_idx] / 14
            for i in range(d14_idx, len(tspan)):
                n_tcells_arr[i] = n_tcells_arr[i - 1] * np.exp(-self.d_Tfh * self.dt)

        return n_tcells_arr

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
            