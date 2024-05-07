import numpy as np
from torch import tensor
from torch import matmul
from copy import deepcopy




class Population:
    

    def __init__(self,
                 ncell,
                 nstart,
                 nres,
                 res_low_seed,
                 res_high_seed,
                 res_low,
                 res_high,
                 n_epitope_sites,
                 mutate_probs,
                 dt,
                 death_rate,
                 birth_rate,
                 capacity,
                 size_threshold,
                 E0,
                 Esat,
                 mu,
                 sigma,
                 offset,
                 dE_min,
                 dE_max,
                 nref,
                 name):
        
        self.nstart = nstart
        self.nres = nres
        self.res_low_seed = res_low_seed
        self.res_high_seed = res_high_seed
        self.res_low = res_low
        self.res_high = res_high
        self.n_epitope_sites = n_epitope_sites
        self.nclones = np.array(ncell).astype(int)
        self.mutate_probs = mutate_probs
        self.dt = dt
        self.death_rate = death_rate
        self.birth_rate = birth_rate
        self.capacity = capacity
        self.size_threshold = size_threshold
        self.E0 = E0
        self.Esat = Esat
        self.mu = mu
        self.sigma = sigma
        self.offset = offset
        self.dE_min = dE_min
        self.dE_max = dE_max
        self.nref = nref
        self.name = name
        self.strains = np.arange(self.nclones.size)
        self.epitope_id = np.empty(shape=self.nclones.shape)
        self.recomb = np.zeros(shape=self.nclones.shape)
        self.dEs = np.zeros(shape=self.nclones.shape)
        self.big_inds = None
        self.prerecomb_E = np.empty(shape=self.nclones.shape, dtype=object)
        self.postrecomb_E = np.empty(shape=self.nclones.shape, dtype=object)
        self.recomb_times = np.empty(shape=self.nclones.shape, dtype=object)

        self.make_residues()

    
    def make_residues(self):
        self.residues = np.squeeze(np.random.uniform(
            size=(len(self.nclones), self.nres, self.n_epitope_sites),
            low=self.res_low_seed,
            high=self.res_high_seed
        ))
        

    def list_all_arrays(self):
        return [self.nclones, self.residues, self.epitope_id, 
                self.strains, self.recomb, self.dEs, self.prerecomb_E,
                self.postrecomb_E, self.recomb_times]
    
    
    def replace_all_arrays(self, nclones, residues, epitope_id, 
                           strains, recomb, dEs, prerecomb_E,
                           postrecomb_E, recomb_times):
        self.nclones = nclones
        self.residues = residues
        self.epitope_id = epitope_id
        self.strains = strains
        self.recomb = recomb
        self.dEs = dEs
        self.prerecomb_E = prerecomb_E
        self.postrecomb_E = postrecomb_E
        self.recomb_times = recomb_times
        

    def filter_small_clones(self):
        self.big_inds = np.where(self.nclones >= self.size_threshold)
        arrays = [array[self.big_inds] for array in self.list_all_arrays()]
        self.replace_all_arrays(*arrays)
        

    def kill(self):
        prob_death = self.get_death()
        self.nclones -= np.random.binomial(self.nclones,
                                           prob_death,
                                           size=self.nclones.size)
        self.filter_small_clones()


    def expand(self):
        """
        Expand members of a population.
        Some expanded clones will mutate and are added to the population.
        Clones that did not mutate or had synonymous mutations are just added
        to the nclone count.
        Clones that died due to mutation are not added to the population.
        """
        unmutated = np.random.binomial(self.tot_new,
                                    self.mutate_probs[0],
                                    size=self.tot_new.size)
        mutated = np.random.binomial(self.tot_new - unmutated,
                                     self.mutate_probs[1],
                                     size=self.tot_new.size)
        self.nclones += unmutated
        return mutated
        

    def set_nclones_for_mutated(self, x):
        max_ = np.maximum(x[x>0], self.size_threshold)
        return np.minimum(max_, self.nstart).astype(int)
    

    def sigmoid(self, x):
        Esat = self.Esat - self.E0
        return np.exp(Esat) / (1 + np.exp(Esat - x))
    

    @staticmethod
    def weighted_mean(values, weights):
        if len(values.shape) in [2, 3]:
            return (values.T * weights / weights.sum()).T.sum(axis=0)
        elif len(values.shape) == 1 and (values.shape[0] != 0):
            return (values * weights).sum() / weights.sum()
        elif len(values.shape) == 1 and (values.shape[0] == 0):
            return np.array([])
                    

    
class Bcells(Population):


    def __init__(self, 
                 pop_args, 
                 tcell, 
                 gc_seed_num,
                 nmax,
                 prob_export,
                 plasma_prob,
                 virus_epitope_sites,
                 bnab_precursor_freq,
                 ncons,
                 panel_bnabs):
        
        super().__init__(*pop_args)
        self.tcell = tcell
        self.gc_seed_num = gc_seed_num
        self.nmax = nmax
        self.prob_export = prob_export
        self.plasma_prob = plasma_prob
        self.ncons = ncons

        probs = [(1 - bnab_precursor_freq) / (virus_epitope_sites - 1)]
        probs = probs * (virus_epitope_sites - 1)
        probs = np.array([bnab_precursor_freq] + probs)
        self.epitope_id = np.random.choice(virus_epitope_sites, 
                                           p=probs,
                                           size=self.nclones.shape)
        
        if panel_bnabs:
            self.epitope_id = np.zeros(100).astype(int)
            self.residues[:,:self.ncons] = 1
            

        
    def get_death(self):
        death_rate = self.death_rate * (1 + self.nclones.sum() / self.capacity)
        return np.ones(shape=self.nclones.shape) * death_rate * self.dt
    
    
    def get_binding_energies(self, viruses, input_list=False):
        """
        Output is of dimension Bcells, viruses
        """

        virus_residues_ = viruses[1] if input_list else viruses.residues

        # group by epitope id so that a single virus residue matrix can be used
        bcell_ids = []
        bindings_ = []
        for epitope_id in np.sort(np.unique(self.epitope_id)):
            bcell_id = np.where(self.epitope_id == epitope_id)[0]
            bcell_ids.append(bcell_id)
            virus_residues = tensor(virus_residues_[:, :, epitope_id])
            bcell_residues = tensor(self.residues[bcell_id, :].T)
            bindings = matmul(virus_residues, bcell_residues).numpy().T
            bindings_.append(bindings)

        if len(bcell_ids) == 0:
            return np.array([])
        
        bcell_ids = np.concatenate(bcell_ids)

        # need to reorder bcell_ids
        reordered = np.empty_like(bcell_ids)
        tmp = np.arange(bcell_ids.size)
        for i, idx in enumerate(bcell_ids):
            reordered[idx] = tmp[i]
        bcell_ids = reordered

        binding_energies = np.concatenate(bindings_)[bcell_ids, :]
        return binding_energies

    
    def get_lambdas(self, viruses, tcell, enter_gc=False):
        if viruses.nclones.sum():
            if enter_gc:
                # take a fraction of the largest virus clones to 
                # reduce computation time
                tot = viruses.nclones.sum()
                cumsum = np.cumsum(
                    viruses.nclones[np.argsort(viruses.nclones)[::-1]]
                )
                gt_threshold = np.where(cumsum > tot * 0.5)[0][0]
                gt_threshold = 500 if gt_threshold > 500 else gt_threshold
                inds = np.argsort(viruses.nclones)[::-1][:gt_threshold]
                arrays = [arr[inds] for arr in viruses.list_all_arrays()]
                viruses.replace_all_arrays(*arrays)

            binding_energies = self.get_binding_energies(viruses)
            norm_energies = self.sigmoid(binding_energies - self.E0)
            bindings = (norm_energies * viruses.nclones).sum(axis=1) / self.nref
            activations = np.random.binomial(self.nclones, 
                                             p=np.minimum(bindings, 1))
            bindings_mean = self.weighted_mean(bindings, self.nclones)
            nact = 1 if activations.sum() == 0 else activations.sum()
            prod_ratio = tcell / nact * bindings / bindings_mean
            self.lambdas = prod_ratio / (1 + prod_ratio) * (activations / self.nclones)

        else:
            self.lambdas = np.zeros(shape=self.nclones.shape)


    def create_seeding_bcells(self, inds):
        seeding_bcells = deepcopy(self)
        seeding_bcells.replace_all_arrays(
            *[arr[inds] for arr in self.list_all_arrays()]
        )
        return seeding_bcells

        
    def seed_gc(self, viruses):
        self.get_lambdas(viruses, self.nmax)
        inds = np.random.choice(
            np.arange(self.lambdas.size),
            size=self.gc_seed_num,
            p=self.lambdas / self.lambdas.sum(),
            replace=False,
        )
        return self.create_seeding_bcells(inds)
        

    def enter_gc(self, viruses, gc_frac):
        copy_viruses = deepcopy(viruses)
        # if many bcells, take a random subset to reduce computation
        if self.nclones.size > 500:
            inds = np.random.choice(self.nclones.size, 
                                    size=500, 
                                    replace=False)
            newself = deepcopy(self)
            arrays = [arr[inds] for arr in newself.list_all_arrays()]
            newself.replace_all_arrays(*arrays)
        else:
            newself = self

        newself.get_lambdas(copy_viruses, newself.nmax*gc_frac, enter_gc=True) # XXX
        random = np.random.uniform(size=newself.lambdas.shape)
        inds = random < np.minimum(newself.lambdas * newself.dt, 1)
        return newself.create_seeding_bcells(inds)

    
    def add_bcells(self, new_bcells):
        all_ = self.list_all_arrays()
        new_ = new_bcells.list_all_arrays()
        arrays = [np.concatenate([a1, a2]) for a1, a2 in zip(all_, new_)]
        self.replace_all_arrays(*arrays)

    
    def get_selection(self, viruses):
        if self.nclones.size == 0:
            return np.empty(0)
        self.get_lambdas(viruses, self.tcell)
        assert np.where(self.lambdas < 0)[0].size == 0
        betas = self.birth_rate * self.lambdas
        pselect = np.minimum(betas * self.dt, 1)
        return pselect
    

    def export_cells(self):
        plasma = self.exported * self.plasma_prob
        plasma_int = plasma.astype(int)
        rand = (
            np.random.uniform(size=plasma_int.shape) < plasma - plasma_int
        ).astype(int)
        plasma = plasma_int + rand

        memory = self.exported - plasma
        plasma_cells = deepcopy(self)
        plasma_cells.nclones = plasma
        plasma_cells.nclones[plasma_cells.nclones > 0] = np.maximum(
            plasma_cells.nclones[plasma_cells.nclones > 0], 
            self.size_threshold
        )
        plasma_cells.filter_small_clones()

        memory_cells = deepcopy(self)
        memory_cells.nclones = memory
        memory_cells.nclones[memory_cells.nclones > 0] = np.maximum(
            memory_cells.nclones[memory_cells.nclones > 0],
            self.size_threshold
        )
        memory_cells.filter_small_clones()
        return plasma_cells, memory_cells


    def calc_selected_and_exported(self, pselect):
        self.pselect = pselect
        tot_new = self.nclones * pselect
        tot_new_int = tot_new.astype(int)
        rand = (
            np.random.uniform(size=tot_new.shape) < tot_new - tot_new_int
        ).astype(int)
        self.tot_new = tot_new_int + rand

        self.exported = self.tot_new * self.prob_export
        exported_int = self.exported.astype(int)
        exported_frac = self.exported - exported_int
        exported_frac_rand = (
            np.random.uniform(size=exported_frac.shape) < exported_frac
        ).astype(int)
        self.exported = exported_int + exported_frac_rand
        self.tot_new -= self.exported


    def add_mutated_clones(self, mutated, viruses):
        all_ = self.list_all_arrays()
        mutated_lists = []
        times_mutated = np.ceil(mutated / self.nstart)


        # iteratively add new mutated clones with nstart initial population
        # until the sampled numbers from mutated become 0
        while np.where(times_mutated > 0)[0].size > 0:
            inds = np.where(times_mutated > 0)[0]
            mutated_ = [array[inds] for array in all_]
            residues_mutate = np.random.choice(self.nres,
                                               size=inds.size)
            dEs = self.offset - np.random.lognormal(self.mu,
                                                    self.sigma, 
                                                    size=inds.size)
            virus_mean = self.weighted_mean((1 / viruses.residues), 
                                            viruses.nclones)
            change = dEs * virus_mean[residues_mutate, mutated_[2]]
            mutated_[1][np.arange(inds.size), residues_mutate] += change
            mutated_[5][np.arange(inds.size)] = dEs
            mutated_[1] = mutated_[1].clip(self.res_low, self.res_high)
            mutated_[0] = self.set_nclones_for_mutated(times_mutated)
            mutated_lists.append(mutated_)
            times_mutated -= 1

        # add bcells
        arrays = [
            np.concatenate([*items]) for items in zip(all_, *mutated_lists)
        ]
        self.replace_all_arrays(*arrays)


    def mutate(self, mutated, viruses):
        if mutated.sum():
            self.add_mutated_clones(mutated, viruses)

        

class Viruses(Population):
    

    def __init__(self, 
                 pop_args, 
                 cons_value, 
                 cons_penalty_weight,
                 ncons,
                 founder_distance,
                 recombination_fraction,
                 numGCs,
                 record_recomb,
                 panel_viruses):
        
        super().__init__(*pop_args)
        self.cons_value = cons_value
        self.cons_penalty_weight = cons_penalty_weight
        self.ncons = ncons
        self.founder_distance = founder_distance
        self.recombination_fraction = recombination_fraction
        self.numGCs = numGCs
        self.record_recomb = record_recomb
        if panel_viruses:
            self.make_conserved()
        else:
            self.select_founders()
            
    
    def get_death(self):
        ind0 = np.where(self.strains==0)
        ind1 = np.where(self.strains==1)
        death_rates = np.ones(shape=self.nclones.shape)
        capacity = self.capacity / 2
        death_rates[ind0] = self.death_rate * (1 + self.nclones[ind0].sum() / capacity)
        death_rates[ind1] = self.death_rate * (1 + self.nclones[ind1].sum() / capacity)
        return death_rates * self.dt


    def get_binding_energies(self, bcells):
        return bcells.get_binding_energies(self).T


    def get_lambdas(self, bcells):
        binding_energies = self.get_binding_energies(bcells)
        norm_energies = self.sigmoid(binding_energies - self.E0)
        bindings = (norm_energies * bcells.nclones).sum(axis=1)
        # normalize by numGCs because more GCs produces more PCs
        bindings /= (self.nref * self.numGCs)
        self.lambdas = bindings


    def get_cons_penalty(self):
        wildtype = np.ones(shape=self.residues[:, :self.ncons, 0].shape) * self.cons_value
        norm = np.linalg.norm(self.residues[:, :self.ncons, 0] - wildtype, axis=1)
        self.cons_penalty = self.cons_penalty_weight * norm


    def get_cons_penalty(self):
        """Second version, Hamming distance"""
        wildtype = np.ones(shape=self.residues[:, :self.ncons, 0].shape) * self.cons_value
        norm = ((self.residues[:, :self.ncons, 0] - wildtype) != 0).sum(1)
        self.cons_penalty = self.cons_penalty_weight * norm
    

    def get_selection(self, bcells):
        self.get_cons_penalty()
        if bcells.epitope_id.size == 0:
            betas = self.birth_rate - self.cons_penalty
        else:
            self.get_lambdas(bcells)
            betas = self.birth_rate - self.cons_penalty - self.lambdas
        pselect = np.minimum(np.maximum(betas, 0) * self.dt, 1)
        return pselect
    

    def calc_selected(self, pselect):
        tot_new = self.nclones * pselect
        tot_new_int = tot_new.astype(int)
        rand = (
            np.random.uniform(size=tot_new.shape) < (tot_new - tot_new_int)
        ).astype(int)
        self.tot_new = tot_new_int + rand
        

    def add_mutated_clones(self, mutated, bcells):
        all_ = self.list_all_arrays()
        mutated_lists = []
        times_mutated = np.ceil(mutated / self.nstart)

        # iteratively add new mutated clones with nstart initial population
        # until the sampled numbers from mutated become 0
        while np.where(times_mutated > 0)[0].size > 0:
            inds = np.where(times_mutated > 0)[0]
            mutated_ = [array[inds] for array in all_]
            residues_mutate = np.random.choice(self.nres,
                                               size=inds.size)
            epitopes_mutate = np.random.choice(self.n_epitope_sites,
                                               size=inds.size)
            
            # if no plasma cells, then just use dE as normal
            if bcells.epitope_id.size == 0:
                dEs = np.random.normal(0, 1, size=inds.size)
                dEs = dEs.clip(self.dE_min, self.dE_max)
                mutated_[1][np.arange(inds.size), 
                            residues_mutate, 
                            epitopes_mutate] += dEs
                mutated_[5][np.arange(inds.size)] = dEs

            # if plasma cells, then use plasma cell residues to get new values
            else:
                # iterate over epitopes, need to select bcells that bind
                # to the matching epitope
                for epitope in np.sort(np.unique(epitopes_mutate)):
                    matching_bcells = np.where(bcells.epitope_id == epitope)[0]
                    bcell_residues = bcells.residues[matching_bcells, :]
                    bcell_mean = self.weighted_mean(
                        (1 / bcell_residues), bcells.nclones[matching_bcells]
                    )
                    i = np.where(epitopes_mutate==epitope)[0]
                    j, k = residues_mutate[i], epitope
                    dEs = self.offset - np.random.lognormal(
                        self.mu, self.sigma, size=bcell_mean[j].shape
                    )
                    dEs = dEs.clip(self.dE_min, self.dE_max)
                    mutated_[1][i, j, k] += dEs * bcell_mean[j]
                    mutated_[5][i] = dEs

            mutated_[1] = mutated_[1].clip(self.res_low, self.res_high)
            mutated_[0] = self.set_nclones_for_mutated(times_mutated)
            mutated_lists.append(mutated_)
            times_mutated -= 1

        # add viruses
        arrays = [
            np.concatenate([*items]) for items in zip(all_, *mutated_lists)
        ]
        self.replace_all_arrays(*arrays)


#     def add_mutated_clones(self, mutated, bcells):
#         """
#         Second version, flip sign
#         """
#         all_ = self.list_all_arrays()
#         mutated_lists = []
#         times_mutated = np.ceil(mutated / self.nstart)

#         # iteratively add new mutated clones with nstart initial population
#         # until the sampled numbers from mutated become 0
#         while np.where(times_mutated > 0)[0].size > 0:
#             inds = np.where(times_mutated > 0)[0]
#             mutated_ = [array[inds] for array in all_]
#             residues_mutate = np.random.choice(self.nres,
#                                                size=inds.size)
#             epitopes_mutate = np.random.choice(self.n_epitope_sites,
#                                                size=inds.size)

#             tmp = mutated_[1][
#                 np.arange(inds.size), residues_mutate, epitopes_mutate
#             ]
#             mutated_[1][
#                 np.arange(inds.size), residues_mutate, epitopes_mutate
#             ] = -1 * tmp
#             mutated_[5][np.arange(inds.size)] = tmp # XXX

#             mutated_[1] = mutated_[1].clip(self.res_low, self.res_high)
#             mutated_[0] = self.set_nclones_for_mutated(times_mutated)
#             mutated_lists.append(mutated_)
#             times_mutated -= 1

#         # add viruses
#         arrays = [
#             np.concatenate([*items]) for items in zip(all_, *mutated_lists)
#         ]
#         self.replace_all_arrays(*arrays)


    def mutate(self, mutated, bcells):
        if mutated.sum():
            self.add_mutated_clones(mutated, bcells)

    
    def make_conserved(self):
        shape = self.residues[:, :self.ncons, 0].shape
        self.residues[:, :self.ncons, 0] = self.cons_value * np.ones(shape=shape)
            
            
    def select_founders(self):
        self.residues[0] = np.random.choice([self.res_high_seed, self.res_low_seed],
                                            size=self.residues[0].shape)
        inds = np.random.choice(self.nres, size=(self.founder_distance, self.n_epitope_sites))
        self.residues[1] = deepcopy(self.residues[0])
        self.residues[1][inds] = self.residues[1][inds] * -1
        self.make_conserved()


    def recombine(self, plasma_bcells, current_time):
        """
        Execute recombination for a fraction of the virus population.
        After selecting the fraction of the population, randomly select
        two viral clones and the number of clones to undergo recombination.

        Choose random breakpoints along the residues, and switch the 
        residues between the two clones after the breakpoint.
        
        """
        all_ = self.list_all_arrays()
        recombine_pop = self.nclones * self.recombination_fraction * self.dt
        recombine_pop = recombine_pop / self.nstart
        recombine_pop_int = recombine_pop.astype(int)
        tmp = recombine_pop - recombine_pop_int
        random = (np.random.uniform(size=tmp.shape) < tmp).astype(int)
        recombine_pop = recombine_pop_int + random
        recombined = []

        # while there are multiple clones available to recombine
        while np.where(recombine_pop > 0)[0].size >= 2:

            inds = np.random.choice(np.where(recombine_pop > 0)[0], 
                                    size=2, 
                                    replace=False)
            min_clone = np.random.choice(min(recombine_pop[inds])) + 1

            breakpoints = np.random.choice(
                self.nres * self.n_epitope_sites, size=min_clone
            )
            
            copy_self = deepcopy(self)
            arrays = [arr[inds] for arr in copy_self.list_all_arrays()]

            # iterate over breakpoints, swap residues between two viruses
            for breakpoint in np.unique(breakpoints):
                arrays_copy = deepcopy(arrays)
                
                # calculate Es to plasma cells before recombining
                if self.record_recomb:
                    prerecomb_E = plasma_bcells.get_binding_energies(
                        arrays_copy, input_list=True
                    )
                    prerecomb_E = self.weighted_mean(prerecomb_E, 
                                                     plasma_bcells.nclones)

                # Recombine
                shape = arrays_copy[1].shape
                arrays_copy[1] = arrays_copy[1].reshape((2, -1))
                virus0 = deepcopy(arrays_copy[1][0, breakpoint:])
                virus1 = deepcopy(arrays_copy[1][1, breakpoint:])
                arrays_copy[1][0, breakpoint:] = virus1
                arrays_copy[1][1, breakpoint:] = virus0
                arrays_copy[1] = arrays_copy[1].reshape(shape)
                arrays_copy[0] = np.minimum(
                    np.maximum(arrays_copy[0], self.size_threshold),
                    self.nstart
                ).astype(int)
                arrays_copy[4] += 1
                
                if self.record_recomb:
                    # calculate Es to plasma cells after recombining
                    postrecomb_E = plasma_bcells.get_binding_energies(
                        arrays_copy, input_list=True
                    )
                    postrecomb_E = self.weighted_mean(postrecomb_E, 
                                                      plasma_bcells.nclones)
                    prerecomb_E = prerecomb_E if prerecomb_E.size > 0 else np.zeros(2)
                    postrecomb_E = postrecomb_E if postrecomb_E.size > 0 else np.zeros(2)

                    # record recombination metrics
                    for i, (prev_E, E) in enumerate(zip(arrays_copy[6], prerecomb_E)):
                        if prev_E is None:
                            arrays_copy[6][i] = E
                        else:
                            arrays_copy[6][i] = np.append(prev_E, E)
                    for i, (prev_E, E) in enumerate(zip(arrays_copy[7], postrecomb_E)):
                        if prev_E is None:
                            arrays_copy[7][i] = E
                        else:
                            arrays_copy[7][i] = np.append(prev_E, E)
                    for i, prev_t in enumerate(arrays_copy[8]):
                        if prev_t is None:
                            arrays_copy[8][i] = current_time
                        else:
                            arrays_copy[8][i] = np.append(prev_t, current_time)
                    
                recombined.append(arrays_copy)

            # subtract recombined clones from initial sample
            recombine_pop[inds] -= min_clone

        # add recombined viral clones
        arrays = [
            np.concatenate([*items]) for items in zip(all_, *recombined)
        ]

        self.replace_all_arrays(*arrays)

        
#     def make_second_founder(self, first_founder, corner=False):
#         if corner:
#             random_vector = -first_founder
#         else:
#             random_vector = np.random.uniform(size=first_founder.shape) - 0.5
#         random_vector[:self.ncons, 0] = 0
#         norm = np.linalg.norm(random_vector, axis=0).mean()
#         random_vector *= self.founder_distance / norm
#         second_founder = first_founder + random_vector
#         return second_founder
    

#     def test_second_founder(self, success, corner=False):
#         for _ in range(1000):
#             first_founder = self.residues[0]
#             second_founder = self.make_second_founder(first_founder, corner=corner)
#             cond1 = second_founder.max() <= self.res_high
#             cond2 = second_founder.min() >= self.res_low
#             if cond1 and cond2:
#                 success = True
#                 self.residues[1] = second_founder
#                 if corner: # offset from the corner
#                     offset = (-first_founder - second_founder)/2
#                     self.residues[0] = first_founder + offset
#                     self.residues[1] = second_founder + offset
#                     cond1_ = self.residues[0].max() <= self.res_high
#                     cond2_ = self.residues[1].min() >= self.res_low
#                     if not (cond1_ and cond2_):
#                         raise ValueError("Founder error")
#                 break
#         return success


#     def select_founders(self):
#         self.make_conserved()
#         success = self.test_second_founder(False)
#         if success is False: # try using threshold values for founder 1
#             self.residues[0] = np.random.choice([self.res_high, self.res_low],
#                                                 size=self.residues[0].shape)
#             self.make_conserved()
#             success = self.test_second_founder(success, corner=True)
#         if success is False:
#             raise ValueError("Founder not created")
