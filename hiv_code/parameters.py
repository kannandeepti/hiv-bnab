import dataclasses
import os
import numpy as np

from . import utils


@dataclasses.dataclass
class Parameters():
    """
    Base dataclass that contains simulation parameters.

    Inherited classes will have access to all these parameters. Some arrays are
    stored as tuples because the np.ndarray type cannot be serialized into a json
    file.
    """

    updated_params_file: str | None = None
    """
    File to updated parameters. If None, then use defaults.
    """

    experiment_dir: str = 'experiments'
    """
    Directory for containing the experiment data.
    """

    history_file_name: str = 'history.pkl'
    """
    File name for writing the history pickle file.
    """

    param_file_name: str = 'parameters.json'
    """
    File name for writing the parameters of an experiment.
    """

    simulation_file_name: str = 'simulation.pkl'
    """
    File name for writing the simulation data.
    """

    write_simulation : bool = False
    """
    Whether to write the simulation.pkl file
    """

    overwrite: bool = True
    """
    Whether to overwrite existing files.
    """
  
    reset_gc_egcs: bool = True
    """
    Whether to reset GCs and EGCs upon a new vaccination.
    If True, this matches Leerang's simulations.
    """
    seed: int = 0
    """
    The random seed for reproducibility.
    """
    
    dt: float = 0.05
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
    
    n_var: int = 1
    """
    The number of variants. Just 1 for HIV in the epitope representation
    """

    n_ag: int = 2
    """
    The number of antigens in circulation.
    """

    n_conserved_epitopes: int = 1
    """
    The number of epitopes that are conserved across all n_ag antigens.
    """

    n_variable_epitopes: int = 1
    """
    The number of epitopes on each antigen that are variable.
    """

    epitope_overlap: float = 0.
    """
    Overlap between conserved epitope and variable epitope on each antigen.
    TODO: generalize to input the full matrix here
    """

    f_ag: tuple[float] = (0.5, 0.5)
    """ fraction of antigen i of total antigen. """

    fdc_capacity: float = 1.0
    """ TODO: how to choose this value? how does it compare to C0?"""
    
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
    The probability of exporting a birthed GC B cell.
    """
    
    output_pc_fraction: float = 0.1
    """
    The fraction of exported GC B cells that differentiate into plasma cells.
    """
    
    egc_output_prob: float = 1.
    """
    The probability of exporting a birthed EGC B cell.
    """
    
    egc_output_pc_fraction: float = 0.6
    """
    The fraction of exported EGC B cells that differentiate into plasma cells.
    """
    
    gc_entry_birth_rate: float = 1
    """
    The birth rate of GC/EGC entry (day-1).
    """
    
    bcell_birth_rate: float = 2.77
    """
    The birth rate of GC/EGC B cells (day-1).
    """
    
    bcell_death_rate: float = 0.58
    """
    The death rate of GC B cells (day-1).
    """
    
    plasma_half_life: float = 4
    """
    The half-life of plasma cells (days).
    """
    
    memory_half_life: float = 11
    """
    The half-life of memory cells (days).
    """
    
    memory_to_gc_fraction: float = 0.
    """
    The fraction of memory cells to germinal centers.
    """

    naive_high_affinity_variants: tuple = tuple([0])
    """
    Indices of variants that are assigned germline affinities according to geometric distribution. 
    Other variants have germline affinities set to E0.
    """
    
    naive_bcells_n_divide: int = 2
    """
    The number of times a naive cell divides upon seeding a GC.
    """
    
    mutation_pdf: tuple[float] = (3.1, 1.2, 3.08)
    """
    The PDF parameters for mutation dE.
    """

    Esat: float = 10.
    """
    Affinity saturation level. If energy is above this value, it is clipped.
    """

    E0: float = 6
    """
    Initial affinity level.
    """

    E1hs: tuple = (7, 6.6, 6.6)
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
    
    n_naive_precursors: int = 2000
    """
    The number of naive precursors.
    """
    
    max_affinity: float = 16.
    """
    The maximum affinity.
    """

    affinities_history: tuple[float] = (6., 7., 8., 9.)
    """
    Affinity thresholds for counting cells for the simulation history.
    """

    affinity_bins: tuple[float] = tuple(
        np.arange(E0, 16.2, 0.2)
    )
    """
    Affinity bins (right edge) for GC/EGC B cells for simulation history.
    """
    
    C0: float = 1.0
    """
    Value to normalize concentrations (nM).
    """

    naive_target_fractions: tuple = (0.8, 0.15, 0.05)
    assert np.isclose(sum(naive_target_fractions), 1)
    
    """
    Fractions of naive cells targeting each epitope.
    """

    K: float = 0.5
    """
    Stringency of Ag capture.
    """
    
    masking: int = 1
    assert masking in [0, 1]
    """
    Whether to use epitope masking.
    """

    bcell_types: tuple[str] = ('plasma_bcells_GC', 'plasma_bcells_EGC')
    """
    Types of Bcell producing Abs.
    """

    ig_types: tuple[str] = ('IgG')
    """
    Types of Igs.
    """

    initial_ka: float = 1e-3
    """
    Initial ka value (nM-1).
    """
    # No IgM in this calculation
    igm0: float = 0.01
    """
    Initial IgM concentration (nM).
    """
    
    d_igg: float = np.log(2) / 28
    """
    Decay rate of IgG (day-1).
    """
    
    conc_threshold: float = 1e-10
    """
    Concentration threshold, below which it is set to 0.
    """
    
    delay: float = 0.
    """
    Time delay before GC-derived plasma cells produce Abs (days).
    """
    
    nm_to_m_conversion: int = -9
    """
    Conversion factor from nM to M.
    """
    
    max_ka: float = 1e11
    """
    Maximum ka value allowed (nM-1).
    """
    
    production: float = 0.5
    """
    Ab production rate is 0.01 * production / n_gc. 
    production of 0.5 is equivalen to 2.5e-5 nM per day per PC, 
    which matches measurement of 174 IgG/s per PC (assumes volume of 1 mL).
    """
    
    seeding_tcells_gc: float = 10
    """
    Tcell amount for seeding GCs.
    """

    seeding_tcells_egc: float = 10
    """
    Tcell amount for seeding single EGC.
    """ 
    
    n_tfh_gc: float = 200
    """
    number of Tfh in each GC (constant)
    """

    n_tfh_egc: float = 200
    """
    number of Tfh in EGC (constant)
    """

    tspan_dt: float = 1. # XXX
    """
    Timestep to save results in history (days).
    """

    mutate_start_time: float = 6.
    """
    Time to turn on mutations (days).
    """

    simulation_time: float = 400
    """
    Number of days to run simulation for.
    """
    
    ################################################################
    # Properties are not included when calling dataclasses.fields.
    # They are not included in the parameters.json file.
    ################################################################
    @property
    def n_timesteps(self) -> int:
        """Number of timesteps for this simulation."""
        return int(self.simulation_time / self.dt)

    @property
    def history_times(self) -> tuple:
        """Timepoints to save results in history (days)."""
        return tuple(np.arange(0, self.simulation_time, self.tspan_dt))

    @property
    def n_history_timepoints(self) -> int:
        """Number of history timepoints."""
        return len(self.history_times)
    
    @property
    def n_ep(self) -> int:
        """The number of distinct epitopes in circulation across all antigens."""
        n_ep = self.n_conserved_epitopes + self.n_ag * self.n_variable_epitopes
        assert(len(self.E1hs) == n_ep)
        assert(len(self.naive_target_fractions) == n_ep)
        return n_ep

    @property
    def overlap_matrix(self) -> tuple:
        """Defines the epitope overlap matrix. Assumes conserved epitopes
        overlap with variable epitopes by an amount specified by epitope_overlap.
        
        TODO : generalize to any number of epitopes
        """
        mat = np.eye(self.n_ep)
        if self.epitope_overlap > 0:
            if self.n_ag == 2 and self.n_variable_epitopes == 1 and self.n_conserved_epitopes == 1:
                mat[0:2, 2] = self.epitope_overlap
                mat[2, 0:2] = self.epitope_overlap
            elif self.n_ag == 2 and self.n_variable_epitopes == 0 and self.n_conserved_epitopes == 3:
                mat[0, 1:] = self.epitope_overlap
                mat[1:, 0] = self.epitope_overlap
            else:
                raise ValueError(f"Epitope overlap not implemented for n_ag={self.n_ag}, n_conserved={self.n_conserved_epitopes}," +
                                 f" n_variable={self.n_variable_epitopes}")
        return tuple(map(tuple, mat))

    @property
    def r_igg(self) -> float:
        """Rate of IgG production (nM/day)."""
        return self.igm0 * self.production / self.n_gc

    @property
    def n_bcell_types(self) -> int:
        """Number of types of Bcells producing Abs."""
        if type(self.bcell_types) == str:
            #only 1 b cell type
            return 1
        return len(self.bcell_types)

    @property
    def n_ig_types(self) -> int:
        """Number of types of Igs."""
        if type(self.ig_types) == str:
            #only 1 ig type
            return 1
        return len(self.ig_types)

    @property
    def class_size(self) -> float:
        """The size of a bin in fitness_array."""
        return self.fitness_array[1] - self.fitness_array[0]

    @property
    def ag_ep_matrix(self) -> np.ndarray:
        """ Returns a n_ep by n_ag matrix with 1's and 0's specifying
        which antigens have which epitopes. """
        # TODO: read from file -> could also discretize from a matrix that specifiesx
        # rho values (conservation) of epitopes across strains n_ep_per_ag x n_ag.
        # i.e. rho[0, 1] = how similar ep0 on ag2 is to ep0 on ag 1.
        ag_ep_matrix = np.zeros((self.n_ep, self.n_ag))
        #first rows are the immunodominant variable epitopes, which each have a single 1 for each antigen
        for ag in range(self.n_ag):
            for ep in range(0, self.n_variable_epitopes):
                ag_ep_matrix[ag*self.n_variable_epitopes + ep, ag] = 1
        #last n_conserved_epitopes rows are 1 for all antigens
        ag_ep_matrix[(self.n_ep - self.n_conserved_epitopes):, :] = 1
        return ag_ep_matrix


    @property
    def sigma(self) -> np.ndarray:
        """Get sigma: covariance matrix per epitope. 
        
        For HIV project, we don't consider variants but instead
        the circulating epitope distribution. so there are no correlations
        in affinity changes between epitopes.

        XXX move to attributes"""
        sigma = np.zeros((self.n_ep, self.n_var, self.n_var))
        for ep_idx in range(self.n_ep):
            sigma[ep_idx] = np.array([
                [1.]
            ])
        return sigma   

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

    total_time = classmethod(utils.total_time)
    """Make total_time a class method to return _total_time."""


    reset_total_time = classmethod(utils.reset_total_time)
    """Make reset_total_time a class method."""


    def update_parameters_from_file(self, file_path: str | None=None) -> None:
        """Update parameter from default by reading file, if file is not None.
        
        Args:
            file_path: path to the json file containing new parameters.
        """
        self.updated_params_file = file_path
        if self.updated_params_file:
            updated_params = utils.read_json(file_path)
            for key, value in updated_params.items():
                #check that parameter is in attributes
                if not hasattr(self, key):
                    raise AttributeError(f"The parameter '{key}' is not a valid attribute of the Parameters class.")
                expected_type = self.__annotations__.get(key)
                #check if the value is an instance of the expected type
                if expected_type:
                    try:
                        # Attempt to cast the value to the expected type
                        value = expected_type(value)
                    except (TypeError, ValueError) as e:
                        # Raise TypeError with additional information if casting fails
                        raise TypeError(f"Cannot cast value '{value}' to type {expected_type} for attribute '{key}': {e}")
                setattr(self, key, value)


    def convert_tuple_to_list(self, nested_tuple: tuple) -> list:
        """Turns nested tuple into a nested list."""
        if isinstance(nested_tuple, tuple):
            return [self.convert_tuple_to_list(item) for item in nested_tuple]
        else:
            return nested_tuple


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

            file_path = os.path.join(
                self.experiment_dir, experiment, self.param_file_name
            )
            exp_params = utils.read_json(file_path)
            
            for param, value in param_dict.items():

                if param == 'updated_params_file':  # Don't consider updated_params_file
                    continue

                if param not in exp_params:
                    previous_exp = False
                    continue

                if param != 'vax_idx':
                    if isinstance(value, tuple):  # Tuples are read as lists
                        value = self.convert_tuple_to_list(value)
                    if value != exp_params[param]:
                        previous_exp = False

                else:
                    if value != exp_params[param] + 1:
                        previous_exp = False

            if previous_exp:
                return experiment
                    
        raise ValueError('Did not find previous experiment')





            