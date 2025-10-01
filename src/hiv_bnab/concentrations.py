r"""
Dynamics of antibody concentrations and affinities
==================================================

This module defines the `Concentrations` class, which tracks the dynamics of
antibody concentrations and affinities. It is also used to calculate the
effective concentration of antigen on FDCs (IC-FDC) after accounting for epitope masking.

Autologous antibodies targeting epitope `i` are produced by plasma cells at a rate :math:`k_{Ig}`
and decay at a rate :math:`d_{Ig}`:

.. math::
    \frac{d[Ig]_{i}^{free}}{dt} = k_{Ig}*[PC_{i}]-d_{Ig}*[Ig]_{i}^{free}

Epitope masking is modeled via a fast binding equilibrium between autologous antibodies
and epitope `i` on FDCs

.. math::
    [IC-FDC]_{i}^{free} + [Ig]_{i}^{free} \overset{Ka_{i}}\leftrightarrow [IC-FDC]_{i}^{masked}

For the purposes of bookkeeping, we define the total concentration of antigen on FDCs
as the sum of the free and masked antigen concentrations:

.. math::
    [IC-FDC]_{i}^{tot} = [IC-FDC]_{i}^{free} + [IC-FDC]_{i}^{masked}

The total concentration of autologous antibodies targeting epitope `i` is:

.. math::
    [Ig]_{i}^{tot} = [Ig]_{i}^{free} + [IC-FDC]_{i}^{masked}

Notes:
- We assume that bnAbs deposit antigen onto FDCs prior to the start of the simulation,
  and that the distribution of epitopes displayed is fixed.
- We assume that the total [IC-FDC] is fixed over the course of the simulation.
- We assume developing aAbs do not deposit antigen onto FDCs since virus levels are low during therapy.

"""

import copy
from enum import Enum
from typing import Callable
import numpy as np
from .parameters import Parameters
from .bcells import Bcells


class Concentrations(Parameters):

    def __init__(self, updated_params_file: str | None = None):
        """Initialize concentration arrays.

        Attributes:
            n_ig_types: Number of different Ig types. Defaults to 1.
            ig_types_arr (np.ndarray of shape (n_ig_types, n_bcell_types)):
                Used to map Ig type to type of bcell that produced it (see update_ka).
            ic_fdc_conc (np.ndarray of shape (n_ep,)):
                Total concentrations of IC-FDC for each epitope (free + masked)
            ab_conc (np.ndarray of shape (n_ep,)):
                Total concentrations of IgG aAbs targeting each epitope (free + masked)
            ab_ka (np.ndarray of shape (n_ep,)):
                Kas for autologous antibodies to each epitope
        """
        super().__init__()
        self.update_parameters_from_file(updated_params_file)
        # only consider IgG antibodies in HIV setting
        self.ig_types_arr = np.array(
            [[1, 1]]
        )  # shape (1, 2) since there are 2 bcell types that produce IgG aAbs (plasma cells from GC and EGC)
        assert self.n_ig_types == 1  # only consider IgG antibodies in HIV setting

        # initialize IC-FDC according to circulating epitope distribution
        self.ic_fdc_conc = self.ag_ep_matrix @ np.array(self.f_ag) * self.fdc_capacity
        # initially aAb concentrations are zero
        self.ab_conc = np.zeros((self.n_ep,))

        self.ab_ka = np.ones((self.n_ep,)) * self.initial_ka

    def get_IC(
        self,
        ig_conc: np.ndarray | float,
        ag_conc: np.ndarray | float,
        ka: np.ndarray | float,
        fail_if_nan: bool = False,
    ) -> np.ndarray:
        """Calculate the Ag-Ab immune complex (IC) concentration based
        on a fast binding equilibrium.

        Args:
            ig_conc (np.ndarray of shape (n_ep,)): total Ab or 'ligand' (free + IC)
            ag_conc (np.ndarray of shape (n_ep,)): total Ag or 'receptor' (free + IC)
            ka (np.ndarray of shape (n_ep,)): affinity of Ab to Ag

            fail_if_nan: If IC conc is not real, then raise ValueError.
                Otherwise, set nan values to 0.

        Returns:
            IC: IC concentrations (shape=(n_ep,))
        """
        term1 = ag_conc + ig_conc + 1 / ka
        term2 = np.emath.sqrt(np.square(term1) - 4 * ag_conc * ig_conc)
        IC = (term1 - term2) / 2

        if fail_if_nan:
            if ~np.isreal(IC) or np.isnan(IC):
                raise ValueError("Imaginary or nan for IC conc")
        else:
            IC = np.nan_to_num(IC, 0)
        return IC

    def get_eff_ic_fdc_conc(self) -> np.ndarray:
        """Get unmasked [IC-FDC] concentration after applying epitope masking,
        accounting for epitope overlap. Assumes antigen is masked directly on FDC
        and that total [IC-FDC] per epitope is fixed.

        Returns:
            eff_ag_conc (np.ndarray of shape (n_ep,)): free [IC-FDC] after epitope masking.
        """
        if self.ic_fdc_conc.sum() == 0:
            return self.ic_fdc_conc

        total_ab_conc_per_epitope = self.ab_conc  # shape n_ep
        weighted_total_ab_conc_per_epitope = self.ab_conc * self.ab_ka  # shape n_ep

        total_ab_conc_per_epitope_with_overlap = total_ab_conc_per_epitope @ np.array(
            self.overlap_matrix
        )  # shape n_ep

        average_ka_per_epitope_with_overlap = (
            weighted_total_ab_conc_per_epitope @ np.array(self.overlap_matrix)
        ) / total_ab_conc_per_epitope_with_overlap  # shape n_ep

        # consider [IC-FDC free] + [Ab] -> [IC-FDC masked] for each epitope separately
        masked_ic_fdc = self.get_IC(
            total_ab_conc_per_epitope_with_overlap,  # shape n_ep
            self.ic_fdc_conc,  # shape n_ep
            average_ka_per_epitope_with_overlap,  # shape n_ep
        )

        total_free_ic_fdc_conc = (
            self.ic_fdc_conc - self.masking * masked_ic_fdc
        )  # shape (n_ep,)
        return total_free_ic_fdc_conc

    def get_rescaled_rates(self, current_time: float) -> tuple[np.ndarray]:
        """Get rescaled Ab decay rates :math:`-d_{Ig}[Ig]`.
        Rescale decay rates in cases where antibody concentrations
        may become negative in the current time step.

        Args:
            current_time: Current simulation time from Simulation class.

        Returns:
            ab_decay (np.ndarray of shape (n_ep,)): Rescaled Ab decay rates.
        """
        ab_decay = -self.d_ig * self.ab_conc + self.epsilon
        # true if ab_conc may become negative in this time step
        rescale_idx = self.ab_conc < -ab_decay * self.dt
        rescale_factor = self.ab_conc / (-ab_decay * self.dt)
        # if any ab_conc may become negative, rescale the decay rate
        if rescale_idx.flatten().sum():
            print(f"Ab reaction rates rescaled. Time={current_time:.2f}")
            ab_decay[rescale_idx] *= rescale_factor[rescale_idx]
            if np.isnan(ab_decay.flatten()).sum():
                raise ValueError("Rescaled Ab decay rates contain Nan")
        return ab_decay  # (shape=(n_ep,))

    def ag_ab_update(self, ab_decay: np.ndarray, current_time: float) -> None:
        """Update the ab_conc arrays based on decay rates using forward Euler's method.
        We assume [IC-FDC] is fixed and has no dynamics.

        Args:
            ab_decay (np.ndarray of shape (n_ep,)): Ab decay rates.
            current_time: Current simulation time from Simulation class.
        """

        self.ab_conc += ab_decay * self.dt
        self.ab_conc[np.abs(self.ab_conc) < self.conc_threshold] = 0

        if np.any(self.ab_conc.flatten() < 0):
            raise ValueError("Negative concentration.")

    def get_ab_from_PC(
        self, current_time: float, plasma_bcells_gc: Bcells, plasma_bcells_egc: Bcells
    ) -> tuple[np.ndarray]:
        """Get the ig_PC and ka_PC arrays, i.e. the antibody concentrations
        produced by GC- and EGC-derived plasma cells and the mean affinities of those plasma cells.

        Here, n_bcell_types = 2, since we only consider GC- and EGC-derived plasma cells.

        Args:
            current_time: Current simulation time from Simulation class.
            plasma_bcells_gc (Bcells): Plasma bcells that were tagged as being
                GC-derived.
            plasma_bcells_egc (Bcells): Plasma bcells that were tagged as being
                EGC-derived.

        Returns:
            ig_PC (np.ndarray of shape (n_bcell_types, n_ep)):
                [Ab] produced by GC vs EGC-derived PCs for each epitope
            ka_PC (np.ndarray of shape (n_var, n_bcell_types, n_ep)):
                Mean affinities of GC vs EGC-derived PCs for each epitope
        """
        ig_PC = np.zeros((self.n_bcell_types, self.n_ep))
        ka_PC = np.array(
            [ig_PC for _ in range(self.n_variants)]
        )  # (n_var, n_bcell_types, n_ep)
        # affinity[0, i] is an array of size (nbcells,) containing the affinities of all bcells in b cell population i
        affinity = np.empty(
            shape=(self.n_variants, self.n_bcell_types), dtype=object
        )  # (n_var, n_bcell_types)
        # target[i] is an array of size (nbcells,) containing the target epitopes of all bcells in b cell population i
        target = np.empty(self.n_bcell_types, dtype=object)

        for var in range(self.n_variants):
            bcell_list = [plasma_bcells_gc, plasma_bcells_egc]
            assert len(bcell_list) == self.n_bcell_types

            for bcell_idx, bcells in enumerate(bcell_list):
                affinity[var, bcell_idx] = bcells.variant_affinities[:, var]
                # GC derived plasma cell antibodies only contribute to concentration
                # after delay time of 2 days
                if bcell_idx == 0:  # PC-GC
                    threshold_values = bcells.activated_time < current_time - self.delay
                else:  # PC-EGC
                    threshold_values = 1

                target[bcell_idx] = (
                    bcells.target_epitope + 1
                ) * threshold_values  # Adding + 1 because 0 needs to mean nonmatching.

        for bcell_idx, _ in enumerate(self.bcell_types):
            for ep in range(self.n_ep):
                if np.any(target[bcell_idx].flatten() == ep + 1):
                    # count number of PCs that target this epitope and multiply by rate k_ig to get Ig produced by this bcell type
                    ig_PC[bcell_idx, ep] = (
                        target[bcell_idx] == ep + 1
                    ).sum() * self.k_ig

                    for var in range(self.n_variants):
                        target_idx = target[bcell_idx] == ep + 1
                        # convert from log_10(M^-1) to log_10(nM^-1)
                        log10_aff = (
                            affinity[var, bcell_idx][target_idx]
                            + self.nm_to_m_conversion
                        )
                        # take the mean of the affinities of all bcells that target this epitope (nM^-1)
                        ka_PC[var, bcell_idx, ep] = np.mean(10**log10_aff)

        return ig_PC, ka_PC

    def update_ka(
        self,
        ab_conc_copy: np.ndarray,
        ab_decay: np.ndarray,
        ig_PC: np.ndarray,
        ka_PC: np.ndarray,
    ) -> None:
        """Update the affinities of autologous antibodies to each epitope using
        forward Euler's with
        .. math::
            \\frac{dKa}{dt} = \\frac{(Ka^{PC}-Ka)k_{Ig}[PC]}{[Ig]+[IC]}

        This equation can be rewritten such that the titer at time t+dt is a weighted average of the titers from
        the pre-existing Ig that has not decayed and the titers from the newly produced Ig from plasma cells:

        .. math::
            Ka_{t+dt} = \\frac{([Ig]_t + Ig_{decay} * dt)Ka_t + k_{Ig}[PC] * dt * Ka^{PC}}{[Ig]_{t+dt}}

        This approximation prevents the Ka from become negative, which may occur when approximations for other
        concentration values are off.

        Args:
            ab_conc_copy (np.ndarray of shape (n_ep,)): ab_conc before any rescaling.
                (shape=(n_ep,))
            ab_decay (np.ndarray of shape (n_ep,)): Ab decay rates.
            ig_PC (np.ndarray of shape (n_bcell_types, n_ep)):
                [Ab] produced by GC vs EGC-derived PCs for each epitope
            ka_PC (np.ndarray of shape (n_var, n_bcell_types, n_ep)):
                Mean affinities of GC vs EGC-derived PCs for each epitope
        """
        current_sum = (ab_conc_copy + self.dt * ab_decay) * self.ab_ka
        new_ka = (
            current_sum
            + (self.ig_types_arr @ (ig_PC * ka_PC[0] * self.dt)).reshape((self.n_ep,))
        ) / (self.ab_conc + self.epsilon)

        new_ka[self.ab_conc == 0] = 0

        if np.any(new_ka.flatten() < 0):
            print("Warning: Error in Ka values, negative")
        if np.any(np.abs(new_ka).flatten() > self.max_ka):
            print(f"Warning: Error in Ka value, greater than {self.max_ka = }")

        new_ka[np.isnan(new_ka)] = 0
        self.ab_ka = new_ka

    def update_concentrations(
        self, current_time: float, plasma_bcells_gc: Bcells, plasma_bcells_egc: Bcells
    ) -> None:
        """Update ab_conc, and ab_ka arrays, i.e. the total concentrations of IgG aAbs targeting each epitope (free + masked)
        and the affinities of those antibodies to each epitope.

        Args:
            current_time: Current simulation time from Simulation class.
            plasma_bcells_gc: Plasma bcells that were tagged as being
                GC-derived.
            plasma_bcells_egc: Plasma bcells that were tagged as being
                EGC-derived.
        """

        ab_conc_copy = copy.deepcopy(self.ab_conc)

        # get rescaled Ab decay rates
        ab_decay = self.get_rescaled_rates(current_time)  # (shape=(n_ep,))

        # update ab_conc based on decay rates
        self.ag_ab_update(ab_decay, current_time)

        # get [Ab] produced by GC vs EGC-derived PCs and the mean affinities of those plasma cells
        ig_PC, ka_PC = self.get_ab_from_PC(
            current_time, plasma_bcells_gc, plasma_bcells_egc
        )

        # Update ab concentrations and affinities accounting for Ig produced from PCs
        self.ab_conc += ig_PC.sum(axis=0) * self.dt
        self.update_ka(ab_conc_copy, ab_decay, ig_PC, ka_PC)

        self.ab_conc[self.ab_conc < self.conc_threshold] = 0
