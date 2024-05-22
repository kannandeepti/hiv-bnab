"""
Simplified concentration dynamics for HIV
=========================================

For the first pass HIV calculation, we ignore the antigen presentation dynamics
and instead assume that [IC-FDC] is fixed at its carrying capacity. We further
assume that bnAbs are the only antibody type that deposit antigen onto the FDC,
and so the distribution of epitopes displayed is fixed. However, epitopes can
progressively be masked as nAbs develop.

"""

import copy
from enum import Enum
from typing import Callable
import numpy as np
from . import utils
from .parameters import Parameters
from .bcells import Bcells


class Concentrations(Parameters):

    def __init__(self):
        """Initialize concentration arrays.
        
        Attributes:
            n_ig_types: Number of different Ig types. 
            ig_types_arr: see update_ka. used to map Ig type to B cell type.
                (shape=(n_ig_types, n_bcell_types))
            ic_fdc_conc: Concentrations of total IC-FDC (shape=(n_ep, )). 
                (free + masked)
            ab_conc: Concentrations of IgG nAbs (shape=(n_ep,))
            ab_ka: Kas for neutralizing antibodies to each epitope.
                (shape=(n_ep,))
        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)

        self.n_bcell_types = 2 #GC and EGC (no plasmablasts)
        #only consider IgG antibodies in HIV setting
        self.ig_types_arr = np.array([[1, 1]])
        self.n_ig_types = 1 #only consider IgG antibodies in HIV setting

        #initialize IC-FDC at capacity according to circulating epitope distribution
        self.ic_fdc_conc = self.ag_ep_matrix @ np.array(self.f_ag) * self.fdc_capacity
        #initially nAb concentrations are zero
        self.ab_conc = np.zeros((self.n_ep,))

        self.ab_ka = np.ones(
            (self.n_ep,)
        ) * self.initial_ka

    
    def get_IC(
        self, 
        ig_conc: np.ndarray | float, 
        ag_conc: np.ndarray | float, 
        ka: np.ndarray | float, 
        fail_if_nan: bool=False
    ) -> np.ndarray:
        """Calculate the Ag-Ab IC concentration.

        Args:
            ig_conc: Ab or 'ligand', either a np.ndarray (shape=(n_ep)) or
                float.
            ag_conc: Ag or 'receptor', nd.ndarray (shape=(n_ep)) or float
            ka: affinity Ka, np.ndarray (shape=(n_ep)) or float.
            fail_if_nan: If IC conc is not real, then raise ValueError.
                Otherwise, set nan values to 0.

        Returns:
            IC: IC concentrations (shape=(n_ep, n_ag) or (n_ag))
        """
        term1 = ag_conc + ig_conc + 1/ka
        term2 = np.emath.sqrt(np.square(term1) - 4 * ag_conc * ig_conc)
        IC = (term1 - term2) / 2

        if fail_if_nan:
            if ~np.isreal(IC) or np.isnan(IC):
                raise ValueError('Imaginary or nan for IC conc')
        else:
            IC = np.nan_to_num(IC, 0)
        return IC


    def get_masked_ag_conc(self) -> np.ndarray:
        """Get [IC-FDC] free concentration after applying epitope masking.
        Assumes antigen is masked directly on FDC and that total [IC-FDC]
        per epitope is fixed.
        
        Returns:
            masked_ag_conc: [IC-FDC] after epitope masking.
                np.ndarray (shape=(n_ep,)).
        """
        if self.ic_fdc_conc.sum() == 0:
            return masked_ag_conc
        
        total_ab_conc_per_epitope = self.ab_conc # shape n_ep
        weighted_total_ab_conc_per_epitope = (
            self.ab_conc * self.ab_ka
        ) # shape n_ep

        total_ab_conc_per_epitope_with_overlap = (
            total_ab_conc_per_epitope @ self.overlap_matrix
        ) # shape n_ep

        average_ka_per_epitope_with_overlap = ((
            weighted_total_ab_conc_per_epitope @ self.overlap_matrix
        ) / total_ab_conc_per_epitope_with_overlap) # shape n_ep

        #consider [IC-FDC free] + [Ab] -> [IC-FDC masked] for each epitope separately
        masked_ic_fdc = self.get_IC(
            total_ab_conc_per_epitope_with_overlap,  # shape n_ep
            self.ic_fdc_conc,                        # shape n_ep
            average_ka_per_epitope_with_overlap      # shape n_ep
        )

        total_free_ic_fdc_conc = self.ic_fdc_conc - self.masking * masked_ic_fdc # shape (n_ep,)
        return total_free_ic_fdc_conc
    

    def get_rescaled_rates(
        self, 
        current_time: float
    ) -> tuple[np.ndarray]:
        """Get rescaled Ab decay rates.

        Needed if concentrations would go to 0.

        Args:
            current_time: Current simulation time from Simulation class.

        Returns:
            ab_decay: Rescaled Ab decay rates.
                np.ndarray (shape=(n_ep,))
        """
        ab_decay = (- self.d_igg * self.ab_conc + 
            self.epsilon
         ) 
        rescale_idx = self.ab_conc < -ab_decay * self.dt 
        rescale_factor = self.ab_conc / (-ab_decay * self.dt) 
        if rescale_idx.flatten().sum():
            import pdb; pdb.set_trace()
            print(f'Ab reaction rates rescaled. Time={current_time:.2f}')
            ab_decay[rescale_idx] *= rescale_factor[rescale_idx]
            if np.isnan(ab_decay.flatten()).sum():
                raise ValueError('Rescaled Ab decay rates contain Nan')
        return ab_decay  # (shape=(n_ep,))
    

    def ag_ab_update(
        self, 
        ab_decay: np.ndarray, 
        current_time: float
    ) -> None:
        """Update the ab_conc arrays based on decay. We assume [IC-FDC]
        is fixed and has no dynamics. Also assume only bnAbs deposit antigen
        on FDC so nAb dynamics independent of deposition rates.

        Args:
            ab_decay: Ab decay rates. np.ndarray (shape=(n_ep,))
            current_time: Current simulation time from Simulation class.
        """

        self.ab_conc += ab_decay * self.dt
        self.ab_conc[np.abs(self.ab_conc) < self.conc_threshold] = 0

        if np.any(self.ab_conc.flatten() < 0):
            raise ValueError('Negative concentration.')
            
    
    def get_ab_from_PC(
        self, 
        current_time: float,
        plasma_bcells_gc: Bcells, 
        plasma_bcells_egc: Bcells
    ) -> tuple[np.ndarray]:
        """Get the ig_PC and ka_PC arrays, i.e. the antibody concentrations
        produced by plasma cells and the affinities of those plasma cells.

        Args:
            current_time: Current simulation time from Simulation class.
            plasma_bcells_gc: Plasma bcells that were tagged as being 
                GC-derived.
            plasma_bcells_egc: Plasma bcells that were tagged as being 
                EGC-derived.
        
        Returns:
            ig_PC: np.ndarray (shape=(n_bcell_types, n_ep))
            ka_PC: np.ndarray (shape=(n_var, n_bcell_types, n_ep))
        """
        ig_PC = np.zeros((self.n_bcell_types, self.n_ep))
        ka_PC = np.array([ig_PC for _ in range(self.n_var)]) # (n_var, n_bcell_types, n_ep)
        #contains all B cell affinities in a given population 
        # so affinity[i] has shape (nbcells,)
        affinity = np.empty(shape=(self.n_var, self.n_bcell_types), dtype=object) # (n_var, n_bcell_types)
        # target[i] has shape (nbcells,)
        target = np.empty(self.n_bcell_types, dtype=object)

        for var in range(self.n_var):

            bcell_list = [plasma_bcells_gc, plasma_bcells_egc]
            assert len(bcell_list) == self.n_bcell_types

            for bcell_idx, bcells in enumerate(bcell_list):
                affinity[var, bcell_idx] = bcells.variant_affinities[:, var]
                # GC derived plasma cell antibodies only contribute to concentration
                # after delay time of 2 days
                if bcell_idx == 0:  # PC-GC
                    threshold_values = bcells.activated_time < current_time - self.delay
                else:                 # PC-EGC
                    threshold_values = 1

                target[bcell_idx] = (bcells.target_epitope + 1) * threshold_values  # Adding + 1 because 0 needs to mean nonmatching.

        for bcell_idx, _ in enumerate(self.bcell_types):
            for ep in range(self.n_ep):
                if np.any(target[bcell_idx].flatten() == ep + 1):
                    #Ig produced by bcell type bcell_idx at rate r_igg
                    ig_PC[bcell_idx, ep] = (
                        (target[bcell_idx] == ep + 1).sum() * self.r_igg)

                    for var in range(self.n_var):
                        target_idx = target[bcell_idx] == ep + 1
                        log10_aff = (
                            affinity[var, bcell_idx][target_idx] + 
                            self.nm_to_m_conversion
                        )
                        ka_PC[var, bcell_idx, ep] = np.mean(10 ** log10_aff)
        
        return ig_PC, ka_PC
    

    def update_ka(
        self, 
        ab_conc_copy: np.ndarray, 
        ab_decay: np.ndarray, 
        ig_PC: np.ndarray, 
        ka_PC: np.ndarray
    ) -> None:
        """Update the ab_ka array.
        
        Args:
            ab_conc_copy: ab_conc before any rescaling.
                (shape=(n_ep,))
            ab_decay: Ab decay rates. np.ndarray (shape=(n_ep,))
            ig_PC: np.ndarray (shape=(n_bcell_types, n_ep))
            ka_PC: np.ndarray (shape=(n_var, n_bcell_types, n_ep))
        """
        current_sum = (ab_conc_copy + self.dt * ab_decay) * self.ab_ka
        new_ka = (
            current_sum + 
            self.ig_types_arr @ (ig_PC * ka_PC[0] * self.dt) #XXX check shape
        ) / (self.ab_conc + self.epsilon)

        new_ka[self.ab_conc == 0] = 0

        if np.any(new_ka.flatten() < 0):
            print('Warning: Error in Ka values, negative')
        if np.any(np.abs(new_ka).flatten() > self.max_ka):
            print(f'Warning: Error in Ka value, greater than {self.max_ka = }')

        new_ka[np.isnan(new_ka)] = 0
        self.ab_ka = new_ka


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

        ab_conc_copy = copy.deepcopy(self.ab_conc)
        
        ab_decay = self.get_rescaled_rates(
            current_time
        ) # (shape=(n_ep,))

        self.ag_ab_update(ab_decay, current_time)

        ig_PC, ka_PC = self.get_ab_from_PC(
            current_time, plasma_bcells_gc, plasma_bcells_egc
        )

        # Update amounts and Ka
        self.ab_conc += ig_PC.sum(axis=0) * self.dt
        self.update_ka(ab_conc_copy, ab_decay, ig_PC, ka_PC)

        self.ab_conc[self.ab_conc < self.conc_threshold] = 0