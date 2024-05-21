import dataclasses
import os

import numpy as np

from . import utils


@dataclasses.dataclass
class Parameters:
    """
    Base dataclass that contains simulation parameters.

    Inherited classes will have access to all these parameters.
    """

    updated_params_file: str | None = None
    """
    File to updated parameters. If None, then use defaults.
    """

    experiment_dir: str = 'experiments'
    """
    Directory for containing the experiment data.
    """

    param_file_name: str = 'parameters.json'
    """
    File name for writing the parameters of an experiment.
    """

    simulation_file_name: str = 'simulation.pkl'
    """
    File name for writing the simulation data.
    """

    overwrite: bool = True
    """
    Whether to overwrite existing files.
    """

    seed: int = 0
    """
    The random seed for reproducibility.
    """
    
    dt: float = 0.05 # XXX
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
    
    initial_bcell_number: int = 0
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

    n_ag: int = 2
    assert n_ag <= n_var
    """
    The number of antigens in circulation.
    """
    
    f_ag: tuple[float] = (0.5, 0.5)
    """ fraction of antigen i of total antigen. """
    
    n_ep: int = 3
    """
    The number of total unique epitopes across a population of antigens in circulation.
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

    naive_high_affinity_variants: tuple = tuple([0])
    """
    Indices of variants that are initially higher affinity. Other variants
    have affinities set to E0.
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

    E1hs: tuple = (7, 6.6, 6.6)
    assert len(E1hs) == n_ep
    """
    Energy levels for geometric distribution for each epitope.
    """

    n_class_bins: int = 11
    """
    The number of class bins.
    """

    fitness_array: tuple[float] = tuple(
        np.linspace(E0, E0 + 2, n_class_bins)
    )
    """
    6 to 8 with interval of 0.2.
    """

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

    affinities_history: tuple[float] = (0., 6., 7., 8., 9.)
    """
    Affinity thresholds for counting cells for the simulation history.
    """
    
    C0: float = 0.008
    """
    Value to normalize concentrations.
    """

    naive_target_fractions: tuple = (0.8, 0.15, 0.05)
    assert np.isclose(sum(naive_target_fractions), 1)
    assert len(naive_target_fractions) == n_ep
    """
    Fractions of naive cells targeting each epitope.
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

    ig_types: tuple[str] = ('IgM', 'IgG-GCPC', 'IgG-EGCPC')
    """
    Types of Ig.
    """
    
    n_ig_types: int = len(ig_types)
    """
    Number of types of Ig.
    """
    
    initial_ka: float = 1e-3
    """
    Initial ka value.
    """
    # No IgM in this calculation
    #igm0: float = 0.01
    """
    Initial IgM concentration (nm-1).
    """

    bnab_conc: float = 0.01
    """
    Concentration of administered bnAbs (fixed) (nm-1).
    """
    
    deposit_rate: float = 24.
    """
    The deposit rate (units XXX).
    """

    d_bnab: float = 0.
    """
    Decay rate of administered bnAbs. (Assume they don't decay)
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
    Maximum ka value allowed (nM-1).
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
    Maximum time allowed (days).
    """
    
    n_tmax: int = 1200
    """
    . XXX
    """

    tspan_dt: float = 0.25
    """
    Timestep to save results in history (days).
    """

    history_times: tuple = tuple(np.arange(0,  tmax + tspan_dt, tspan_dt))
    """
    Timepoints to save results in history (days).
    """

    n_history_timepoints: int = len(history_times)
    """
    Number of history timepoints.
    """
    
    d_Tfh: float = 0.01
    """
    Rate of change of Tfh. XXX
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
    """
    Time to give the vaccine dose.
    """

    dose: float = ag0
    """
    Concentration of Ag in the dose.
    """

    egc_stop_time: float = 6.
    """
    The time to stop EGCs.
    """

    n_timesteps = int(vax_timing[vax_idx] / dt)
    """
    Number of timesteps for this simulation.
    """
    
    ################################################################
    # Properties are not included when calling dataclasses.fields.
    ################################################################
    
    @property
    def ag_ep_matrix(self) -> np.ndarray:
        """ Returns a n_ep by n_ag matrix with 1's and 0's specifying
        which antigens have which epitopes. """
        # TODO: read from file -> could also discretize from a matrix that specifiesx
        # rho values (conservation) of epitopes across strains n_ep_per_ag x n_ag.
        # i.e. rho[0, 1] = how similar ep0 on ag2 is to ep0 on ag 1.
        ag_ep_matrix = np.array([
            [1, 1], #both antigens share ep 0
            [1, 0], #antigen 1 has eps 1, 2
            [1, 0],
            [0, 1], #antigen 2 has eps 3, 4
            [0, 1]
        ])
        assert(ag_ep_matrix.shape == (self.n_ep, self.n_ag))
        return ag_ep_matrix

    @property
    def overlap_matrix(self) -> np.ndarray:
        """Defines the epitope overlap matrix. XXX move to attributes"""
        overlap_matrix = np.eye(self.n_ep)
        # XXX adjust if there is some overlap
        return overlap_matrix


    @property
    def sigma(self) -> np.ndarray:
        """Get sigma: covariance matrix per epitope. XXX move to attributes"""
        sigma = np.zeros((self.n_ep, self.n_var, self.n_var))
        sigma[0] = np.array([
            [1, 0.4, 0.4],
            [0.4, 1, 0],
            [0.4, 0, 1]
        ])
        sigma[1] = np.array([
            [1, 0.95, 0.4],
            [0.95, 1., 0],
            [0.4, 0., 1]
        ])
        sigma[2] = np.array([
            [1, 0.4, 0.95],
            [0.4, 1, 0],
            [0.95, 0, 1]
        ])
        return sigma


    @property
    def n_tcells_arr(self) -> np.ndarray:
        """Calculate tcell amounts over time spaced dt apart.
        
        Returns:
            n_tcells_arr: Number of tcells over time.
                np.ndarray (shape=tmax/dt)
        """
        tspan = np.arange(0, self.tmax + self.dt, self.dt)
        n_tcells_arr = np.zeros(shape=tspan.shape)

        if self.tmax <= 14:
            n_tcells_arr = self.n_tmax * tspan / 14
        else:
            d14_idx = round(14 / self.dt + 1)
            n_tcells_arr[:d14_idx] = self.n_tmax * tspan[:d14_idx] / 14
            for i in range(d14_idx, len(tspan)):
                n_tcells_arr[i] = n_tcells_arr[i - 1] * np.exp(-self.d_Tfh * self.dt)

        return n_tcells_arr
    

    @property
    def naive_bcells_arr(self) -> np.ndarray:
        """Get number of naive B cells in each fitness class.

        Fitness class refers to the bins for affinities that Bcells start with.
        For example, the first bin is between affinities of 6 and 6.2, the second bin
        is between affinities of 6.2 and 6.4, etc. until 7.8 to 8. Between 6 and 8
        with widths of 0.2, there are 11 such bins.

        We wish to calculate how many naive bcells with affinities in each bin and
        targeting each epitope will be created. Those numbers are stored in
        naive_bcells_arr. Higher affinities will have fewer numbers according
        to the geometric distribution.
        
        Returns:
            naive_bcells_arr: np.ndarray (shape=(n_ep, n_class_bins))
        """
        max_classes = np.around((np.array(self.E1hs) - self.E0 )/ self.class_size) + 1
        geo_dist_slopes = np.zeros(self.n_ep)  # slopes of geometric distribution
        naive_bcells_arr = np.zeros(shape=(self.n_ep, self.n_class_bins))

        for ep in range(self.n_ep):
            if max_classes[ep] > 1:
                func = (
                    lambda x: self.n_naive_precursors - 
                    (x ** max_classes[ep] - 1) / (x - 1)
                )
                geo_dist_slopes[ep] = utils.fsolve_mult(func, guess=1.1)
            else:
                geo_dist_slopes[ep] = self.n_naive_precursors

        for ep in range(self.n_ep):

            if max_classes[ep] > 1:
                naive_bcells_arr[ep, :self.n_class_bins] = (
                    self.naive_target_fractions[ep] * 
                    geo_dist_slopes[ep] ** 
                    (max_classes[ep] - (np.arange(self.n_class_bins) + 1))
                )

            elif max_classes[ep] == 1:
                naive_bcells_arr[ep, 0] = (
                    self.naive_target_fractions[ep] * 
                    self.n_naive_precursors
                )

        return naive_bcells_arr


    ################################################################
    # Methods.
    ################################################################
    

    def update_parameters_from_file(self, file_path: str | None=None) -> None:
        """Update parameter from default by reading file, if file is not None.
        
        Args:
            file_path: path to the json file containing new parameters.
        """
        self.updated_params_file = file_path
        if self.updated_params_file:
            updated_params = utils.read_json(file_path)
            for key, value in updated_params.items():
                setattr(self, key, value)


    def find_previous_experiment(self) -> str:
        """Find experiment directory for previous experiment.
        
        Search the experiment directory, and read the parameter json file
        from each experiment subdirectory. If the parameter json file matches
        the current experiment's parameters in every parameter except vax_idx,
        which the previous experiment should be 1 less, then it is the previous
        experiment. If no previous experiment is found, raise an error.

        Returns:
            experiment: the name of the experiment directory corresponding to
                the previous experiment.
        """
        param_dict = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

        for experiment in os.listdir(self.experiment_dir):

            previous_exp = True

            data_dir = os.path.join(self.experiment_dir, experiment)
            exp_params = utils.read_json(data_dir, self.param_file_name)
            
            for param, value in param_dict.items():

                if param != 'vax_idx':
                    if value != exp_params[param]:
                        previous_exp = False
                else:
                    if value != exp_params[param] + 1:
                        previous_exp = False

            if previous_exp:
                return experiment
                    
        raise ValueError('Did not find previous experiment')





            