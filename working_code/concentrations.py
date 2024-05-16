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
            ig_type: Different Ig types
            n_ig_types: Number of different Ig types.
            ig_types_arr: Not sure. Involved in matmul with ig_new and ka_new
                (shape=(n_ig_types, n_ep))
            ag_conc: Concentrations of Ag (shape=(n_ep + 1, n_ag)). 
                ag_conc[0] corresponds to total [Ag] (free + bound)
                ag_conc[i, j] corresponds to [IC] for epitope i on antigen j.
            ab_conc: Concentrations of free Ab (shape=(n_ig_types, n_ep))
            ab_decay_rates: Decay rates for Ig types. Multiplied with ab_conc.
                np.ndarray (shape=(n_ig_types, np.newaxis))
            ab_ka_condense_fn: If there is only one antigen in circulation, this function
                determines which one to use when computing [IC] concentrations. 
                In this case, the Ka array for ICs just keeps track of epitopes and Ig type.
                For covid work, Leerang used WT at index 0.
                For multiple antigens, this function should condense the shape of n_var down to n_ag.
            ab_ka: Kas for antibodies to multiple variants.
                (shape=(n_var, n_ig_types, n_ep))
                Keeping track of affinities to n_var > n_ag, but there are only n_ag in circulation.
        """
        super().__init__()
        #TODO: add a bnAb class (on a separate branch for HIV project)
        self.ig_types_arr = np.array([[0, 0, 0], [1, 0 ,0], [0, 1, 1]])
        self.ag_conc = np.zeros((self.n_ep + 1, self.n_ag))
        self.ab_conc = np.zeros((self.n_ig_types, self.n_ep))
        self.ab_conc[0] = self.igm0 / self.n_ep
        #BUGFIX : shouldnt IgM decay rate be d_igm? was previously 0
        self.ab_decay_rates = np.array([self.d_igm, self.d_igg, self.d_igg])[:, np.newaxis]

        # By default, take the first n_ag rows of Ka to be the n_ag variants in circulation
        self.ab_ka_condense_fn: Callable[
            [np.ndarray], np.ndarray
        ] = lambda x: x[:self.n_ag]  

        self.ab_ka = np.ones(
            (self.n_var, self.n_ig_types, self.n_ep)
        ) * self.initial_ka

    def set_ab_ka_condense_fn(
            self, 
            fn : Callable[[np.ndarray], np.ndarray]
    ):
        try:
            output = fn(self.ab_ka)
        except Exception as e:
            print("User supplied ka_condense_fn has an error")
            raise(e)
        
        if len(output.shape) != (self.n_ag, self.n_ig_types, self.n_ep):
            raise ValueError("condense fn should return an array of shape (n_ag, n_ig_types, n_ep)")
        self.ab_ka_condense_fn = fn
    
    def get_IC(
        self, 
        ig_conc: np.ndarray | float, 
        ag_conc: np.ndarray, 
        avg_ka: np.ndarray | float, 
        fail_if_nan: bool=False
    ) -> np.ndarray:
        """Calculate the Ag-Ab IC concentration.

        Args:
            ig_conc: Ab or 'ligand', either a np.ndarray (shape=(n_ep)) or
                float.
            ag_conc: Ag or 'receptor', nd.ndarray (shape=(n_ag))
            avg_ka: Average Ka, np.ndarray (shape=(n_ep)) or float.
            fail_if_nan: If IC conc is not real, then raise ValueError.
                Otherwise, set nan values to 0.

        Returns:
            IC: IC concentrations (shape=(n_ep, n_ag) or (n_ag))
        """
        term1 = ag_conc + ig_conc + 1/avg_ka
        term2 = np.emath.sqrt(np.square(term1) - 4 * ag_conc * ig_conc)
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
        ) # shape n_ep

        average_ka_per_epitope_with_overlap = ((
            weighted_total_ab_conc_per_epitope @ self.overlap_matrix
        ) / total_ab_conc_per_epitope_with_overlap) # shape n_ep

        total_ag_conc = self.ag_conc.sum(axis=0)  # shape n_ag

        IC = self.get_IC(
            total_ab_conc_per_epitope_with_overlap[:, np.newaxis] / 5,  # XXX should we change from 5, shape n_ep
            total_ag_conc,                               # XXX shape n_ag
            average_ka_per_epitope_with_overlap[:, np.newaxis]          # XXX shape n_ep
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
        ig_conc: float, 
        ag_conc: np.ndarray, 
        avg_ka: np.ndarray
    ) -> np.ndarray:
        """Get the FDC-deposit rates.
        
        Args:
            ig_conc: total free Ab or 'ligand' concentration, float in this case.
            ag_conc: total Ag or 'receptor' concentration, nd.ndarray (shape=(n_ag))
            avg_ka: Average Ka to a particular antigen (shape=(n_ag,))

        Returns:
            deposit_rates: The FDC-deposit rates. np.ndarray
                (shape=(n_ag, n_ig_types, n_ep)).
        """
        if ig_conc > 0 and ag_conc.sum() > 0:
            IC_total_per_antigen = ag_conc / (1. + 1./(avg_ka * ig_conc)) #(n_ag,)
            if np.any(~np.isreal(IC_total_per_antigen)) or np.any(np.isnan(IC_total_per_antigen)):
                raise ValueError('Imaginary or nan for IC conc')
            IC_total_per_antigen = IC_total_per_antigen.reshape((self.n_ag, 1, 1))
            #after condensing, result is (n_ag, n_ig_types, n_ep), but ab_conc is (n_ig_types, n_ep)
            #denom should be shape (n_ag, 1, 1)
            denom = (self.ab_ka_condense_fn(self.ab_ka) * self.ab_conc[np.newaxis, :, :]).sum(axis=(1, 2))
            denom = denom.reshape((self.n_ag, 1, 1))
            deposit_rates = (
                self.deposit_rate * 
                IC_total_per_antigen * 
                self.ab_ka_condense_fn(self.ab_ka) * 
                self.ab_conc[np.newaxis, :, :]
             ) / denom  # shape (n_ag, n_ig_types, n_ep)
        else:
            deposit_rates = np.zeros(
                shape=(self.n_ag, self.n_ig_types, self.n_ep)
            )
        return deposit_rates
    

    def get_rescaled_rates(
        self, 
        deposit_rates: np.ndarray, 
        current_time: float
    ) -> tuple[np.ndarray]:
        """Get rescaled deposit and Ab decay rates.

        Needed if concentrations would go to 0.

        Args:
            deposit_rates: The FDC-deposit rates. np.ndarray
                (shape=(n_ag, n_ig_types, n_ep)).
            current_time: Current simulation time from Simulation class.

        Returns:
            deposit_rates: Rescaled FDC-deposit rates.
                np.ndarray (shape=(n_ag, n_ig_types, n_ep)).
            ab_decay: Rescaled Ab decay rates.
                np.ndarray (shape=(n_ig_types, n_ep))
        """
        #Note: I removed the deposit_rates from this since ab_conc is now FREE Ig
        ab_decay = (
            self.ab_decay_rates * self.ab_conc + 
            self.epsilon
         ) # (n_ig_types n_ep)
        rescale_idx = self.ab_conc < -ab_decay * self.dt # (n_ig_types, n_ep)
        rescale_factor = self.ab_conc / (-ab_decay * self.dt) # (n_ig_types, n_ep)
        if rescale_idx.flatten().sum():
            print(f'Ab reaction rates rescaled. Time={current_time:.2f}')
            deposit_rates[:, rescale_idx] *= rescale_factor[rescale_idx]
            ab_decay[rescale_idx] *= rescale_factor[rescale_idx]
            if np.isnan(ab_decay.flatten()).sum():
                raise ValueError('Rescaled Ab decay rates contain Nan')
        return deposit_rates, ab_decay  # (shape=(n_ag, n_ig_types, n_ep)) (3, n_ep)
    

    def get_rescaled_ag_decay(self, 
        deposit_rates: np.ndarray, 
        ab_decay: np.ndarray,
        current_time: float
    ) -> tuple[np.ndarray]:
        """Get rescaled rates after checking Ag concentrations.

        Args:
            deposit_rates: FDC-deposit rates. np.ndarray
                (shape=(n_ag, n_ig_types, n_ep)).
            ab_decay: Ab decay rates. np.ndarray (shape=(n_ig_types, n_ep))
            current_time: Current simulation time from Simulation class.

        Returns:
            deposit_rates: Rescaled FDC-deposit rates. np.ndarray
                (shape=(n_ag, n_ig_types, n_ep)).
            ag_decay: Rescaled Ag decay rates.
                np.ndarray (shape=(n_ag))
            ab_decay: Rescaled Ab decay rates.
                np.ndarray (shape=(n_ig_types, n_ep))

        """
        ag_decay = (
            #Note: if ag_conc[0] is total Ag concentration, then this first term is wrong
            #should be divided by 1 + sum(ab_ka * ab_conc)...
            -self.d_ag * self.ag_conc[ConcentrationIdx.SOLUBLE.value] -
            deposit_rates.sum(axis=(1,2)) + 
            self.epsilon  # (n_ag)
        )

        rescale_idx = (
            self.ag_conc[ConcentrationIdx.SOLUBLE.value] < -ag_decay * self.dt
        )

        rescale_factor = (
            self.ag_conc[ConcentrationIdx.SOLUBLE.value] / 
            (-ag_decay * self.dt) # (n_ag)
        )

        if rescale_idx.flatten().sum():
            print(f'Ag reaction rates rescaled. Time={current_time:.2f}')
            ag_decay[rescale_idx] *= rescale_factor  # (n_ag)
            deposit_rates_rescaled = copy.deepcopy(deposit_rates)
            deposit_rates_rescaled[rescale_idx] *= rescale_factor  #(shape=(n_ag, n_ig_types, n_ep))
            #XXX not sure what rescaling does, but ab_decay now independent of deposit_rates
            #ab_decay += (
            #    deposit_rates.sum(axis=0) - 
            #    deposit_rates_rescaled.sum(axis=0)  # (3, n_ep)
            #)
            deposit_rates = deposit_rates_rescaled
        return deposit_rates, ag_decay, ab_decay
    

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
                (shape=(n_ag, n_ig_types, n_ep)).
            ag_decay_rescaled: Ag decay rates. np.ndarray (shape=(n_ag))
            ab_decay: Ab decay rates. np.ndarray (shape=(n_ig_types, n_ep))
            current_time: Current simulation time from Simulation class.
        """
        #decay of soluble antigen (n_ag,)
        self.ag_conc[ConcentrationIdx.SOLUBLE.value] += (
            ag_decay * self.dt + self.F0 * 
            np.exp(self.k * current_time) * (current_time < self.T * self.dt)
        )

        self.ag_conc[ConcentrationIdx.IC_FDC.value:] += (
            deposit_rates.sum(axis=1).T * self.dt + 
            self.ag_conc[ConcentrationIdx.IC_FDC.value:] * self.d_IC * self.dt
        )

        self.ab_conc += ab_decay * self.dt
        self.ab_conc[np.abs(self.ab_conc) < self.conc_threshold] = 0

        for array in [self.ag_conc, self.ab_conc]:
            if np.any(array.flatten() < 0):
                raise ValueError('Negative concentration.')
            
    
    def get_new_arrays(
        self, 
        current_time: float,
        plasmablasts: Bcells,
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
            ig_new: np.ndarray (shape=(n_ig_types, n_ep))
            ka_new: np.ndarray (shape=(n_var, n_ig_types, n_ep))
        """
        threshold = current_time - self.delay
        ig_new = np.zeros((self.n_ig_types, self.n_ep))
        ka_new = np.array([ig_new for _ in range(self.n_var)]) # (n_var, n_ig_types, n_ep)
        affinity = np.empty(shape=(self.n_var, self.n_ig_types), dtype=object) # (n_var, n_ig_types)
        target = np.empty(self.n_ig_types, dtype=object)

        for var in range(self.n_var):
            bcell_list = [plasmablasts, plasma_bcells_gc, plasma_bcells_egc]
            for ig_idx, bcells in enumerate(bcell_list):
                affinity[var, ig_idx] = bcells.variant_affinities[:, var]
                # GC derived plasma cell antibodies only contribute to concentration
                # after delay time of 2 days
                threshold_value = {
                    ConcentrationIdx.IGG.value: (bcells.activated_time < threshold)
                }.get(ig_idx, 1)
                target[ig_idx] = bcells.target_epitope * threshold_value

        for ig_idx, ig_type in enumerate(self.ig_types):
            for ep in range(self.n_ep):
                if np.any(target[ig_idx].flatten() == ep):
                    ig_new[ig_idx, ep] = (
                        (target[ig_idx] == ep).sum() * 
                        self.r_igm * (ig_idx == 0) + 
                        self.r_igg * (ig_idx > 0)
                    )

                    for var in range(self.n_var):
                        target_idx = target[ig_idx] == ep
                        log10_aff = (
                            affinity[var, ig_idx][target_idx] + 
                            self.nm_to_m_conversion
                        )
                        ka_new[var, ig_idx, ep] = np.mean(10 ** log10_aff)
        
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
                (shape=(n_ig_types, n_ep))
            ab_decay: Ab decay rates. np.ndarray (shape=(n_ig_types, n_ep))
            ig_new: np.ndarray (shape=(n_ig_types, n_ep))
            ka_new: np.ndarray (shape=(n_var, n_ig_types, n_ep))
        """
        for var in range(self.n_var):
            current_sum = (ab_conc_copy + self.dt * ab_decay) * self.ab_ka[var]
            new_ka = (
                current_sum + 
                self.ig_types_arr @ (ig_new * ka_new[var] * self.dt)
            ) / (self.ab_conc + self.epsilon)

            new_ka[self.ab_conc == 0] = 0

            if np.any(new_ka.flatten() < 0):
                print('Warning: Error in Ka values, negative')
            if np.any(np.abs(new_ka).flatten() > self.max_ka):
                print(f'Warning: Error in Ka value, greater than {self.max_ka = }')

            new_ka[np.isnan(new_ka)] = 0
            self.ab_ka[var] = new_ka


    def update_concentrations(
        self, 
        current_time: float,
        plasmablasts: Bcells,
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

        #average affinity across all antibody types / eptiopes
        #this needs to be shap (n_ag,)
        avg_ka = (
            self.ab_ka * self.ab_conc[np.newaxis, :, :]
        ).sum(axis=(1,2)) / total_ab_conc  # (n_ag, )

        deposit_rates = self.get_deposit_rates(
            total_ab_conc, soluble_ag_conc, avg_ka
        ) # (shape=(n_ag, n_ig_types, n_ep))

        ab_conc_copy = copy.deepcopy(self.ab_conc)
        
        deposit_rates, ab_decay = self.get_rescaled_rates(
            deposit_rates, current_time
        ) # (shape=(n_ag, n_ig_types, n_ep)) (3, n_ep)

        deposit_rates, ag_decay, ab_decay = self.get_rescaled_ag_decay(
            deposit_rates, ab_decay, current_time
        )

        self.ag_ab_update(deposit_rates, ag_decay, ab_decay, current_time)
        ig_new, ka_new = self.get_new_arrays(
            current_time, plasmablasts, plasma_bcells_gc, plasma_bcells_egc
        )

        # Update amounts and Ka
        self.ab_conc[ConcentrationIdx.IC_FDC.value:] += np.vstack([
            ig_new[ConcentrationIdx.IGM_NAT.value, :],
            ig_new[ConcentrationIdx.IGM_IMM.value:, :].sum(axis=0)
        ]) * self.dt
        self.update_ka(ab_conc_copy, ab_decay, ig_new, ka_new)

        self.ab_conc[self.ab_conc < self.conc_threshold] = 0
        self.ag_conc[self.ag_conc < self.conc_threshold] = 0