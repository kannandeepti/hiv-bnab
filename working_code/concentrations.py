import numpy as np
import utils
from parameters import Parameters


class Concentrations(Parameters):

    def __init__(self):
        super().__init__()

        self.overlap_matrix = self.get_overlap_matrix()

        self.ag_conc = {
            utils.ConcentrationNames.SOLUBLE: np.zeros(self.n_ag),
            utils.ConcentrationNames.IC_FDC: np.zeros((self.n_ag, self.n_ep))
        }

        self.ab_conc = {
            utils.ConcentrationNames.IGM_NAT: np.zeros(self.n_ep),
            utils.ConcentrationNames.IGM_IMM: np.zeros(self.n_ep),
            utils.ConcentrationNames.IGG : np.zeros(self.n_ep)
        }

        self.ab_ka = {
            utils.ConcentrationNames.IGM_NAT: np.zeros(self.n_ep),
            utils.ConcentrationNames.IGM_IMM: np.zeros(self.n_ep),
            utils.ConcentrationNames.IGG : np.zeros(self.n_ep)
        }

    
    def get_overlap_matrix(self) -> None:
        self.overlap = np.array([
            [1, self.q12, self.q13],
            [self.q12, 1, self.q23], 
            [self.q13, self.q23, 1]
        ])


    def apply_epitope_masking(self): # todo finish
        masked_ag_conc = {
            utils.ConcentrationNames.SOLUBLE: np.zeros(self.n_ep),
            utils.ConcentrationNames.IC_FDC: np.zeros(self.n_ep)
        }

        if utils.dict_sum(self.ag_conc) == 0:
            return masked_ag_conc
        
        total_ab_conc_per_epitope = utils.dict_sum_over_keys(self.ab_conc)
        weighted_ab_concs = {
            key: conc * ka
            for (key, conc), (_, ka) in zip(self.ab_conc.items(), self.ab_ka.items())
        }
        weighted_total_ab_conc_per_epitope = utils.dict_sum_over_keys(weighted_ab_concs)








