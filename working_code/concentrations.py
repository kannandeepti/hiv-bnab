import copy
from enum import Enum
from typing import Callable
import numpy as np
import utils
from parameters import Parameters
from bcells import Bcells


class ConcentrationIdx(Enum):
    """Class for recording which indices in arrays correspond to which Ag/Ab types."""
    # For masked_ag_conc
    SOLUBLE = 0
    IC_FDC = 1
    # For ab_conc and ab_ka
    IGM_NAT = 0
    IGM_IMM = 1
    IGG = 2


class Concentrations(Parameters):

    def __init__(self):
        """Initialize concentration arrays.
        
        Attributes:
            ig_type - Different Ig types
            num_ig_types - Number of different Ig types.
            ig_types_arr - Not sure. Involved in matmul with ig_new and ka_new
                (shape=(num_ig_types, n_ep))
            ag_conc - Concentrations of Ag (shape=(n_ep + 1, n_ag)). 
                ag_conc[0] corresponds to soluble Ag.
            ab_conc - Concentrations of Ab (shape=(num_ig_types, n_ep))
            ab_decay_rates - Decay rates for Ig types. Multiplied with ab_conc.
                np.ndarray (shape=(num_ig_types, np.newaxis))
            ab_ka_condense_fn - If there are multiple variant antigens being
                given at once in the GC, this function determines which Ka to use.
                Default is to use the mean, but I think Leerang/Melbourne use the
                WT at index 0.
            ab_ka - Kas for antibodies to multiple variants.
                (shape=(n_var, num_ig_types, n_ep))
        """
        super().__init__()

        self.ig_types = ['IgM', 'IgG-GCPC', 'IgG-EGCPC']
        self.num_ig_types = len(self.ig_types)
        self.ig_types_arr = np.array([[0, 0, 0], [1, 0 ,0], [0, 1, 1]])

        self.ag_conc = np.zeros((self.n_ep + 1, self.n_ag))
        self.ab_conc = np.zeros((self.num_ig_types, self.n_ep))
        self.ab_decay_rates = np.array([0, self.d_igm, self.d_igg])[:, np.newaxis]

        self.ab_ka_condense_fn: Callable[
            [np.ndarray], np.ndarray
        ] = lambda x: x.mean(axis=0)  # Take average over all n_var variants for ab_ka

        self.ab_ka = np.ones(
            (self.n_var, self.num_ig_types, self.n_ep)
        ) * self.initial_ka  # think this needs to be also n_var

        self.set_overlap_matrix()

    
    def set_overlap_matrix(self) -> None:
        """Defines the epitope overlap matrix."""
        self.overlap_matrix = np.array([
            [1, self.q12, self.q13],
            [self.q12, 1, self.q23], 
            [self.q13, self.q23, 1]
        ])

    
    def get_IC(
        self, 
        L: np.ndarray | float, 
        R: np.ndarray, 
        avg_ka: np.ndarray | float, 
        fail_if_nan: bool=False
    ) -> np.ndarray:
        """Calculate the Ag-Ab IC concentration.

        Args:
            L: Ab or 'ligand', either a np.ndarray (shape=(n_ep)) or
                float.
            R: Ag or 'receptor', nd.ndarray (shape=(n_ag))
            avg_ka: Average Ka, np.ndarray (shape=(n_ep)) or float.
            fail_if_nan: If IC conc is not real, then raise ValueError.
                Otherwise, set nan values to 0.

        Returns:
            IC: IC concentrations (shape=(n_ep, n_ag) or (n_ag))
        """
        term1 = R + L + 1/avg_ka
        term2 = np.emath.sqrt(np.square(term1) - 4 * R * L)
        IC = (term1 - term2) / 2

        if fail_if_nan:
            if ~np.isreal(IC) or np.isnan(IC):
                raise ValueError('Imaginary or nan for IC conc')
        else:
            IC = np.nan_to_num(IC, 0)
        return IC


    def get_masked_ag_conc(self) -> np.ndarray:
        """Get Ag concentration after applying epitope masking.
        
        Returns:
            masked_ag_conc: Ag concentration after epitope masking.
                np.ndarray (shape=(2, n_ep, n_ag)). masked_ag_conc[0]
                corresponds to soluble Ag and masked_ag_conc[1] corresponds
                to IC-bound Ag.
        """
        masked_ag_conc = np.zeros((2, self.n_ep, self.n_ag))

        if self.ag_conc.sum() == 0:
            return masked_ag_conc
        
        total_ab_conc_per_epitope = self.ab_conc.sum(axis=0)  # shape n_ep
        weighted_total_ab_conc_per_epitope = (
            self.ab_conc * self.ab_ka_condense_fn(self.ab_ka)
        ).sum(axis=0) # shape n_ep

        total_ab_conc_per_epitope_with_overlap = (
            total_ab_conc_per_epitope @ self.overlap_matrix
        )[:, np.newaxis] # shape n_ep

        average_ka_per_epitope_with_overlap = ((
            weighted_total_ab_conc_per_epitope @ self.overlap_matrix
        ) / total_ab_conc_per_epitope_with_overlap)[:, np.newaxis] # shape n_ep

        total_ag_conc = self.ag_conc.sum(axis=0)  # shape n_ag

        IC = self.get_IC(
            total_ab_conc_per_epitope_with_overlap / 5,  # XXX should we change from 5, shape n_ep
            total_ag_conc,                               # XXX shape n_ag
            average_ka_per_epitope_with_overlap          # XXX shape n_ep
        )

        total_free_ag_conc = total_ag_conc - self.masking * IC # shape (n_ep, n_ag)

        masked_ag_conc[ConcentrationIdx.SOLUBLE.value] = total_free_ag_conc * (
            self.ag_conc[ConcentrationIdx.SOLUBLE.value] / total_ag_conc
        )

        masked_ag_conc[ConcentrationIdx.IC_FDC.value] = total_free_ag_conc * (
            self.ag_conc[ConcentrationIdx.IC_FDC.value:].sum(axis=0) / total_ag_conc
        )

        return masked_ag_conc
    

    def get_deposit_rates(
        self, 
        L: float, 
        R: np.ndarray, 
        avg_ka: float
    ) -> np.ndarray:
        """Get the FDC-deposit rates.
        
        Args:
            L: Ab or 'ligand', float in this case.
            R: Ag or 'receptor', nd.ndarray (shape=(n_ag))
            avg_ka: Average Ka, float in this case.

        Returns:
            deposit_rates: The FDC-deposit rates. np.ndarray
            (shape=(num_ig_types, n_ep, n_ag)).
        """
        if L > 0 and R.sum() > 0:
            IC_soluble = self.get_IC(
                L, R, avg_ka, fail_if_nan=True
            )[:, np.newaxis, np.newaxis] # shape n_ag
            denom = (self.ab_ka_condense_fn(self.ab_ka) * self.ab_conc).sum()
            deposit_rates = (
                self.deposit_rate * 
                IC_soluble * 
                self.ab_ka_condense_fn(self.ab_ka) * 
                self.ab_conc
             ) / denom
            deposit_rates = deposit_rates.T
        else:
            deposit_rates = np.zeros(
                shape=self.ab_ka_condense_fn(self.ab_ka).shape + tuple([self.n_ag])
            )
        return deposit_rates
    

    def get_rescaled_rates(
        self, 
        deposit_rates: np.ndarray, 
        current_time: np.ndarray
    ) -> tuple[np.ndarray]:
        """Get rescaled deposit and Ab decay rates.

        Needed if concentrations would go to 0.

        Args:
            deposit_rates: The FDC-deposit rates. np.ndarray
            (shape=(num_ig_types, n_ep, n_ag)).
            current_time: Current simulation time from Simulation class.

        Returns:
            deposit_rates: Rescaled FDC-deposit rates.
                np.ndarray (shape=(num_ig_types, n_ep, n_ag)).
            ab_decay: Rescaled Ab decay rates.
                np.ndarray (shape=(num_ig_types, n_ep))
        """
        ab_decay = -deposit_rates.sum(axis=2) - self.ab_decay_rates * self.ab_conc # (3, n_ep)
        rescale_idx = self.ab_conc < -ab_decay * self.dt # (3, n_ep)
        rescale_factor = self.ab_conc / (-ab_decay * self.dt) # (3, n_ep)
        if rescale_idx.flatten().sum():
            print(f'Reaction rates rescaled. Time={current_time:.2f}')
            deposit_rates[rescale_idx] *= rescale_factor[rescale_idx]
            ab_decay[rescale_idx] *= rescale_factor[rescale_idx]
            if np.isnan(ab_decay.flatten()).sum():
                raise ValueError('Rescaled Ab decay rates contain Nan')
        return deposit_rates, ab_decay  # (3, n_ep, n_ag) (3, n_ep)
    

    def get_rescaled_ag_decay(self, 
        deposit_rates: np.ndarray, 
        ab_decay: np.ndarray
    ) -> tuple[np.ndarray]:
        """Get rescaled rates after checking Ag concentrations.
        
        Needed if concentrations would go to 0. XXX i think need to have rescale_idx

        Args:
            deposit_rates: FDC-deposit rates. np.ndarray
            (shape=(num_ig_types, n_ep, n_ag)).
            ab_decay: Ab decay rates. np.ndarray (shape=(num_ig_types, n_ep))

        Returns:
            deposit_rates: Rescaled FDC-deposit rates. np.ndarray
            (shape=(num_ig_types, n_ep, n_ag)).
            ag_decay_rescaled: Rescaled Ag decay rates.
                np.ndarray (shape=(n_ag))
            ab_decay: Rescaled Ab decay rates.
                np.ndarray (shape=(num_ig_types, n_ep))

        """
        ag_decay = (
            -self.d_ag * self.ag_conc[ConcentrationIdx.SOLUBLE.value] -
            deposit_rates.sum(axis=(0,1))  # (n_ag)
        )

        rescale_factor = (
            self.ag_conc[ConcentrationIdx.SOLUBLE.value] / 
            (-ag_decay * self.dt) # (n_ag)
        )

        ag_decay_rescaled = ag_decay * rescale_factor  # (n_ag)
        deposit_rates_rescaled = deposit_rates * rescale_factor  # (3, n_ep, n_ag)
        ab_decay_rescaled = (
            ab_decay + deposit_rates.sum(axis=2) - 
            deposit_rates_rescaled.sum(axis=2)  # (3, n_ep)
        )
        return deposit_rates_rescaled, ag_decay_rescaled, ab_decay_rescaled
    

    def ag_ab_update(
        self, 
        deposit_rates: np.ndarray, 
        ag_decay: np.ndarray, 
        ab_decay: np.ndarray, 
        current_time: float
    ) -> None:
        """Update the ag_conc and ab_conc arrays based on decay and deposition.
        
        Args:
            deposit_rates: FDC-deposit rates. np.ndarray
            (shape=(num_ig_types, n_ep, n_ag)).
            ag_decay_rescaled: Ag decay rates. np.ndarray (shape=(n_ag))
            ab_decay: Ab decay rates. np.ndarray (shape=(num_ig_types, n_ep))
            current_time: Current simulation time from Simulation class.
        """
        self.ag_conc[ConcentrationIdx.SOLUBLE] += (
            ag_decay * self.dt + self.F0 * 
            np.exp(self.k * current_time) * (current_time < self.T * self.dt)
        )

        self.ag_conc[ConcentrationIdx.IC_FDC:] += (
            deposit_rates.sum(axis=0) * self.dt + 
            self.ag_conc[ConcentrationIdx.IC_FDC:] * self.d_IC * self.dt
        )

        self.ab_conc += ab_decay * self.dt
        self.ab_conc[np.abs(self.ab_conc) < self.conc_threshold] = 0

        for array in [self.ag_conc, self.ab_conc]:
            if utils.any(array.flatten() < 0):
                raise ValueError('Negative concentration.')
            
    
    def get_new_arrays(
        self, 
        current_time: float, 
        plasma_bcells_gc: Bcells, 
        plasma_bcells_egc: Bcells
    ) -> tuple[np.ndarray]:
        """Get the ig_new and ka_new arrays, not sure what they do.

        Args:
            current_time: Current simulation time from Simulation class.
            plasma_bcells_gc: Plasma bcells that were tagged as being 
                GC-derived.
            plasma_bcells_egc: Plasma bcells that were tagged as being 
                EGC-derived.
        
        Returns:
            ig_new: np.ndarray (shape=(num_ig_types, n_ep))
            ka_new: np.ndarray (shape=(n_var, num_ig_types, n_ep))
        """
        threshold = current_time - self.delay
        ig_new = np.zeros((self.num_ig_types, self.n_ep))
        ka_new = np.array([ig_new for _ in range(self.n_var)]) # (n_var, n_ig_types, n_ep)
        affinity = np.empty(shape=(self.n_var, self.num_ig_types), dtype=object) # (n_var, n_ig_types)
        target = np.empty(self.num_ig_types, dtype=object)

        for var in range(self.n_var):
            #affinity[var, 0] = plasmablasts.var #XXX what were the plasmablasts doing?
            affinity[var, 1] = plasma_bcells_gc.variant_affinities[:, var] #(num_bcells,)
            affinity[var, 2] = plasma_bcells_egc.variant_affinities[:, var]
            #GC derived plasma cell antibodies only contribute to concentration after delay time of 2 days
            target[1] = (
                plasma_bcells_gc.target_epitope * 
                (plasma_bcells_gc.activated_time < threshold)
            )
            target[2] = plasma_bcells_egc.target_epitope #(num_bcells,)

        for ig_idx, ig_type in enumerate(self.ig_types):
            for ep in range(self.n_ep):
                if utils.any(target[ig_idx].flatten() == ep):
                    ig_new[ig_idx, ep] = (
                        (target[ig_idx] == ep).sum() * 
                        self.r_igm * (ig_idx == 0) + 
                        self.r_igg * (ig_idx > 0)
                    )

                    for var in range(self.n_var):
                        target_idx = target[ig_idx] == ep
                        ka_new[var, ig_idx, ep] = np.mean(
                            (10 ** affinity[var, ig_idx][target_idx] + 
                             self.nm_to_m_conversion)
                        )
        
        return ig_new, ka_new
    

    def update_ka(
        self, 
        ab_conc_copy: np.ndarray, 
        ab_decay: np.ndarray, 
        ig_new: np.ndarray, 
        ka_new: np.ndarray
    ) -> None:
        """Update the ab_ka array.
        
        Args:
            ab_conc_copy: ab_conc before any rescaling.
                (shape=(num_ig_types, n_ep))
            ab_decay: Ab decay rates. np.ndarray (shape=(num_ig_types, n_ep))
            ig_new: np.ndarray (shape=(num_ig_types, n_ep))
            ka_new: np.ndarray (shape=(n_var, num_ig_types, n_ep))
        """
        for var in range(self.n_var):
            current_sum = (ab_conc_copy + self.dt * ab_decay) * self.ab_ka[var]
            new_ka = (
                current_sum + 
                self.ig_types_arr @ (ig_new * ka_new[var] * self.dt)
            ) / self.ab_conc

            new_ka[self.ab_conc == 0] = 0

            if utils.any(new_ka.flatten() < 0):
                print('Warning: Error in Ka value, negative')
            if utils.any(np.abs(new_ka).flatten() > self.max_ka):
                print(f'Warning: Error in Ka value, greater than {self.max_ka = }')

            new_ka[np.isnan(new_ka)] = 0
            self.ab_ka[var] = new_ka


    def update_concentrations(
        self, 
        current_time: float, 
        plasma_bcells_gc: Bcells, 
        plasma_bcells_egc: Bcells
    ) -> None:
        """Update ag_conc, ab_conc, and ab_ka arrays.

        Args:
            current_time: Current simulation time from Simulation class.
            plasma_bcells_gc: Plasma bcells that were tagged as being 
                GC-derived.
            plasma_bcells_egc: Plasma bcells that were tagged as being 
                EGC-derived.
        """

        # Add concentration from dose
        if np.isclose(current_time, self.dose_time):
            self.ag_conc[ConcentrationIdx.SOLUBLE.value] += self.dose

        soluble_ag_conc = self.ag_conc[ConcentrationIdx.SOLUBLE.value] # shape n_ag
        total_ab_conc = self.ab_conc.sum()  # float

        avg_ka = (
            self.ab_ka_condense_fn(self.ab_ka) * self.ab_conc
        ).sum() / total_ab_conc  # float

        deposit_rates = self.get_deposit_rates(
            total_ab_conc, soluble_ag_conc, avg_ka
        ) # (3, n_ep, n_ag)

        ab_conc_copy = copy.deepcopy(self.ab_conc)
        
        deposit_rates, ab_decay = self.get_rescaled_rates(
            deposit_rates, current_time
        ) # (3, n_ep, n_ag) (3, n_ep)

        deposit_rates, ag_decay, ab_decay = self.get_rescaled_ag_decay(
            deposit_rates, ab_decay
        )

        self.ag_ab_update(deposit_rates, ag_decay, ab_decay, current_time)
        ig_new, ka_new = self.get_new_arrays(
            current_time, plasma_bcells_gc, plasma_bcells_egc
        )

        # Update amounts and Ka
        self.ab_conc[ConcentrationIdx.IC_FDC:] += np.vstack([
            ig_new[ConcentrationIdx.IGM_NAT, :],
            ig_new[ConcentrationIdx.IGM_IMM:, :].sum(axis=0)
        ]) * self.dt
        self.update_ka(ab_conc_copy, ab_decay, ig_new, ka_new)

        self.ab_conc[self.ab_conc < self.conc_threshold] = 0
        self.ag_conc[self.ag_conc < self.conc_threshold] = 0