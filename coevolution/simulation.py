import numpy as np
import pickle
from copy import deepcopy
from populations import Viruses, Bcells



class Simulation:
    # XXX is a marker to come back to this line, usually meaning that the value
    # for the parameter is not well-justified.
    

    def __init__(self,
                 bcell_nstart=10,
                 bcell_nnaive=int(2e4),           # See below
                 # https://www.sciencedirect.com/science/article/pii/S0952791517302078
                 # Whole protein specific b cells usually around 1 in 1e4
                 # EoD-GT8 vrc01 class b cells are 1 in 3e5
                 # Vrc01 are 3% of env specific b cells
                 # 1e6 naive b cells, 20000 cells per GC
                 # The bnAbs in this patient are not VRC01-class, but data on precursor frequencies on
                 # this patient's bnAbs is not available.
                 virus_nstart=100,                # Number of viruses in each starting clone
                 nres=50,                         # Number of residues
                 ncons=20,                        # Number of conserved residues. Goes in epitope 0 XXX
                 res_low_seed=-.5,                # Lower bound for seeding residues
                 res_high_seed=.5,                # Upper bound for seeding residues
                 res_low=-1.,                     # Lower bound for residues after seeding XXX
                 res_high=1.,                     # Upper bound for residues after seeding XXX
                 timesteps=20000,                 # Number of timesteps XXX 
                 dt=0.1,                          # Timestep
                 bcell_death_rate=0.5,            # Leerang used
                 bcell_birth_rate=2.5,            # Leerang used
                 bcell_capacity=np.inf,           # GC capacity, not needed so set to infinity
                 bcell_size_threshold=5,          # Threshold before removing clone 
                 bcell_mutate_probs=[0.3, 0.2],   # Silent, nonsilent
                 bcell_epitope_sites=1,           # Bcells only have one epitope site
                 bcell_nref=4e5,                  # XXX Weights how strongly Bcells bind to antigen
                 nmax=1e0,                        # Set so 10 bcells per GC per day
                 prob_export=0.1,                 # Leerang used
                 plasma_prob=0.5,                 # Leerang used
                 plasma_half_life=4,              # Leerang used 4 days
                 memory_half_life=120,            # 4 months, could be longer but population becomes big if I set to infinity
                 memory_fraction=0.1,             # Fraction of seeding B cells that are memory
                 numGCs=20,                       # Number of GCs, Leerang used 200 but takes a long time to run
                 bnab_precursor_freq=0.03,        # 3% of Env specific B cells are bnAb. See above
                 virus_death_rate=0.5,            # Leerang used
                 virus_birth_rate=80,             # 250 new virions per replicat doi:10.1128/JVI.05095-11, replication 2 days which is 8 times longer than 4 times/day for GC B cells. 2.5*250/8=80
                 virus_capacity=1e4 ,             # About same as number of GC B cells XXX
                 virus_size_threshold=5,          # Threshold before removing clone 
                 virus_mutate_probs=[0.3, 0.63],  # silent, nonsilent. .63
                 # 10% of nonsilent mutations are lethal based on Env fitness landscape E > 7.
                 # also 10% based on mutational entropies from DMS. Entropy < 2.4 
                 # (doi:10.1371/journal.ppat.1006114)
                 virus_epitope_sites=6,           # XXX
                 # 1200 aa in Env, 1200 / 50 = 24 epitopes
                 # around 4 known bnAb sites, 4/24 epitopes are bnAb = 1/6
                 virus_nref=4e1,                  # XXX Weights strength of negative selection. Used to be 1e6 or 1e2
                 founder_distance=2,              # Varied
                 recombination_fraction=0.07,     # Based on recombination freq per sequence
                 cons_value=1.,                   # XXX value of the conserved region in virus vectors
                 cons_penalty_weight=10,          # XXX used to be 30, weight for mutating conserved region of virus vectors
                 tcell=1200,                      # Leerang used
                 bcell_start_time=18,             # CDC says 18-45 days to detect HIV Abs. Avg is 32, 2 weeks before is 18.
                 gc_seed_num=20,                  # 200 B cells seed GC, 10 per clone
                 E0=0,                          
                 Esat=4,                          # Leerang used Esat 4 greater than E0
                 mu=1.5,                          # Mean of lognormal distribution
                 sigma=0.3,                       # Std dev of lognormal distribution
                 offset=3,                        # Offset of lognormal distribution
                 dE_min=-1,                       # Min dE
                 dE_max=1,                        # Max dE
                 history_timestep=10,             # Timestep for saving history
                 record_recomb=True,              # Whether to record recombination metrics
                 seed=0):                         # Random seed
        
        np.random.seed(seed)

        self.nres = nres
        self.ncons = ncons
        self.res_low_seed = res_low_seed
        self.res_high_seed = res_high_seed
        self.res_low = res_low
        self.res_high = res_high
        self.timesteps = timesteps
        self.dt = dt
        
        self.bcell_nstart = bcell_nstart
        self.bcell_nnaive = bcell_nnaive
        self.virus_nstart = virus_nstart
        self.bcell_death_rate = bcell_death_rate
        self.bcell_birth_rate = bcell_birth_rate
        self.bcell_capacity = bcell_capacity
        self.bcell_size_threshold = bcell_size_threshold
        self.bcell_mutate_probs = bcell_mutate_probs
        self.bcell_epitope_sites = bcell_epitope_sites
        self.bcell_nref = bcell_nref
        self.nmax = nmax
        self.prob_export = prob_export
        self.plasma_prob = plasma_prob
        self.plasma_half_life = plasma_half_life
        self.memory_half_life = memory_half_life
        self.memory_fraction = memory_fraction
        self.numGCs = numGCs
        self.bnab_precursor_freq = bnab_precursor_freq

        self.virus_death_rate = virus_death_rate
        self.virus_birth_rate = virus_birth_rate
        self.virus_capacity = virus_capacity
        self.virus_size_threshold = virus_size_threshold
        self.virus_mutate_probs = virus_mutate_probs
        self.virus_epitope_sites = virus_epitope_sites
        self.virus_nref = virus_nref
        self.cons_value = cons_value
        self.cons_penalty_weight = cons_penalty_weight
        self.founder_distance = founder_distance
        self.recombination_fraction = recombination_fraction

        self.tcell = tcell
        self.bcell_start_time = bcell_start_time
        self.gc_seed_num = gc_seed_num
        self.E0 = E0
        self.Esat = Esat
        self.mu = mu
        self.sigma = sigma
        self.offset = offset
        self.dE_min = dE_min
        self.dE_max = dE_max
        self.history_timestep = history_timestep
        self.random_seed = seed
        self.record_recomb = record_recomb
        
        self.seeded_gc = False
        self.current_time = 0.
        self.cum_seed_ncells = 0
        self.cum_seed_mcells = 0

        self.history = {}
        properties = ['nclones', 'epitope_id', 'strains', 'recomb',
                      'init_distance', 'binding','cons_distance',
                      'cons_value','breadth','binding_panel',
                      'prerecomb_E','postrecomb_E', 'recomb_times',
                      'residues']
        
        for pop in ['viruses', 'gc_bcells', 'plasma_bcells', 'memory_bcells']:
            self.history[pop] = {}
            for prop in properties:
                self.history[pop][prop] = []


    def create_bcells(self, nbcell, panel_bnabs=False):
        self.bcell_args = [nbcell,
                           self.bcell_nstart,
                           self.nres,
                           self.res_low_seed,
                           self.res_high_seed,
                           self.res_low,
                           self.res_high,
                           self.bcell_epitope_sites,
                           self.bcell_mutate_probs,
                           self.dt,
                           self.bcell_death_rate,
                           self.bcell_birth_rate,
                           self.bcell_capacity,
                           self.bcell_size_threshold,
                           self.E0,
                           self.Esat,
                           self.mu,
                           self.sigma,
                           self.offset,
                           self.dE_min,
                           self.dE_max,
                           self.bcell_nref,
                           'bcells']
        self.bcell_specific_args = [self.tcell,
                                    self.gc_seed_num,
                                    self.nmax,
                                    self.prob_export,
                                    self.plasma_prob,
                                    self.virus_epitope_sites,
                                    self.bnab_precursor_freq,
                                    self.ncons,
                                    panel_bnabs]
        return Bcells(self.bcell_args, *self.bcell_specific_args)


    def create_viruses(self, nvirus, panel_viruses=False):
        self.virus_args = [nvirus,
                           self.virus_nstart,
                           self.nres,
                           self.res_low_seed,
                           self.res_high_seed,
                           self.res_low,
                           self.res_high,
                           self.virus_epitope_sites,
                           self.virus_mutate_probs,
                           self.dt,
                           self.virus_death_rate,
                           self.virus_birth_rate,
                           self.virus_capacity,
                           self.virus_size_threshold,
                           self.E0,
                           self.Esat,
                           self.mu,
                           self.sigma,
                           self.offset,
                           self.dE_min,
                           self.dE_max,
                           self.virus_nref,
                           'viruses']
        self.virus_specific_args = [self.cons_value,
                                    self.cons_penalty_weight,
                                    self.ncons,
                                    self.founder_distance,
                                    self.recombination_fraction,
                                    self.numGCs,
                                    self.record_recomb,
                                    panel_viruses]
        return Viruses(self.virus_args, *self.virus_specific_args)


    def create_populations(self):
        empty = np.ones(0) * self.bcell_nstart

        self.viruses = self.create_viruses(np.ones(2) * self.virus_nstart)
        self.initial_viruses = deepcopy(self.viruses)
        self.panel_viruses = self.create_viruses(np.ones(100) * self.virus_nstart, 
                                                 panel_viruses=True)

        self.naive_bcells = self.create_bcells(
            np.ones(self.bcell_nnaive) * self.bcell_nstart
        )
        self.gc_bcells = [
            self.create_bcells(empty) for _ in range(self.numGCs)
        ]

        def get_death_rate(half_life):
            prefactor = 1 / self.dt
            exp = 2 ** (-self.dt / half_life)
            return prefactor * (1 - exp)

        self.plasma_bcells = self.create_bcells(empty)
        self.memory_bcells = self.create_bcells(empty)
        self.plasma_bcells.death_rate = get_death_rate(self.plasma_half_life)
        self.plasma_bcells.capacity = 1e20
        self.memory_bcells.death_rate = get_death_rate(self.memory_half_life)
        self.memory_bcells.capacity = 1e20
        
        self.panel_bnabs = self.create_bcells(np.ones(100) * self.bcell_nstart,
                                              panel_bnabs=True)

    
    def calc_memory_fraction(self):
        denom = self.cum_seed_mcells + self.cum_seed_ncells
        return 0 if denom == 0 else self.cum_seed_mcells / denom


    def run_gc(self, i):
        gc_size = np.sum([self.gc_bcells[i].nclones.sum() for i in range(self.numGCs)])
        gc_frac = gc_size / (self.numGCs * 1000) # assume 1000 B cells per GC
        seeding_bcells = self.naive_bcells.enter_gc(self.viruses, gc_frac)
        self.gc_bcells[i].add_bcells(seeding_bcells)
        self.print_seeding_ncells += seeding_bcells.nclones.sum()
        self.cum_seed_ncells += seeding_bcells.nclones.sum()
        if self.memory_bcells.epitope_id.size > 0:
            if self.calc_memory_fraction() < self.memory_fraction:
                seeding_memory = self.memory_bcells.enter_gc(self.viruses, gc_frac)
                self.gc_bcells[i].add_bcells(seeding_memory)
                self.print_seeding_mcells += seeding_memory.nclones.sum()
                self.cum_seed_mcells += seeding_memory.nclones.sum()

        pselect = self.gc_bcells[i].get_selection(self.viruses)
        self.gc_bcells[i].calc_selected_and_exported(pselect)
        mutated = self.gc_bcells[i].expand()
        self.gc_bcells[i].mutate(mutated, self.viruses)

        plasma_cells, memory_cells = self.gc_bcells[i].export_cells()
        self.plasma_bcells.add_bcells(plasma_cells)
        self.memory_bcells.add_bcells(memory_cells)

        self.gc_bcells[i].kill()

        
    def run_timestep(self):
        """
        Run one timestep of the simulation.
        For times before bcell_start_time, only the virus runs.
        After bcell_start_time, both virus and GC run.
        GCs are seeded once after bcell_start_time.
        """

        # virus pop
        pselect = self.viruses.get_selection(self.plasma_bcells)
        self.viruses.calc_selected(pselect)
        mutated = self.viruses.expand()
        self.viruses.mutate(mutated, self.plasma_bcells)
        self.viruses.recombine(self.plasma_bcells, self.current_time)
        self.viruses.kill()

        if self.current_time > self.bcell_start_time:
            if self.seeded_gc is False:
                for i, _ in enumerate(self.gc_bcells):
                    if self.viruses.nclones.size > 0:
                        seeding_bcells = self.naive_bcells.seed_gc(self.viruses)
                        self.gc_bcells[i].add_bcells(seeding_bcells)
                self.seeded_gc = True
                print('Seeded GCs')

            # GCs
            for i, _ in enumerate(self.gc_bcells):
                self.run_gc(i)

            # decay plasma and memory cells
            self.plasma_bcells.kill()
            self.memory_bcells.kill()


    def update_history(self):
        # Properties
        for prop in ['nclones', 'epitope_id', 'strains', 'recomb']:
            self.history['viruses'][prop].append(
                getattr(self.viruses, prop)
            )
            self.history['gc_bcells'][prop].append(
                [getattr(self.gc_bcells[i], prop) for i in range(self.numGCs)]
            )
            self.history['plasma_bcells'][prop].append(
                getattr(self.plasma_bcells, prop)
            )
            self.history['memory_bcells'][prop].append(
                getattr(self.memory_bcells, prop)
            )

        # Distance of conserved part from init
        shape = self.viruses.residues[:, :self.ncons, 0].shape
        wt = self.cons_value * np.ones(shape=shape)
        self.history['viruses']['cons_distance'].append(np.linalg.norm(
            self.viruses.residues[:, :self.ncons, 0] - wt, axis=1
        ))

        # Distance of conserved part from naive, need to filter for 
        # bnAb precursors later
        gc_cons = []
        for i in range(self.numGCs):
            naive = self.naive_bcells.residues[self.gc_bcells[i].strains]
            naive = naive[:, :self.ncons]
            gc_cons.append(np.linalg.norm(
                self.gc_bcells[i].residues[:, :self.ncons] - naive, axis=1
            ))
        self.history['gc_bcells']['cons_distance'].append(gc_cons)

        for pop in ['plasma_bcells', 'memory_bcells']:
            naive = self.naive_bcells.residues[getattr(self, pop).strains]
            naive = naive[:, :self.ncons]
            diff = np.linalg.norm(
                getattr(self, pop).residues[:, :self.ncons] - naive, axis=1
            )
            self.history[pop]['cons_distance'].append(diff)

        # Avg value of conserved part
        self.history['viruses']['cons_value'].append(
            self.viruses.residues[:, :self.ncons, 0].mean(axis=1)
        )
        gc_cons_val = []
        for i in range(self.numGCs):
            gc_cons_val.append(
                self.gc_bcells[i].residues[:, :self.ncons].mean(axis=1)
            )
        self.history['gc_bcells']['cons_value'].append(gc_cons_val)
        for pop in ['plasma_bcells', 'memory_bcells']:
            val = getattr(self, pop).residues[:, :self.ncons].mean(axis=1)
            self.history[pop]['cons_value'].append(val)
        
        # Distance from initial virus
        init_virus = self.initial_viruses.residues[self.viruses.strains]
        self.history['viruses']['init_distance'].append(
            np.linalg.norm(
            self.viruses.residues - init_virus, axis=1
            ).mean(axis=1)
        )

        # Distance from naive Bcell
        gc_diffs = []
        for i in range(self.numGCs):
            naive = self.naive_bcells.residues[self.gc_bcells[i].strains]
            diff = np.linalg.norm(self.gc_bcells[i].residues - naive, axis=1)
            gc_diffs.append(diff)
        self.history['gc_bcells']['init_distance'].append(gc_diffs)

        for pop in ['plasma_bcells', 'memory_bcells']:
            naive = self.naive_bcells.residues[getattr(self, pop).strains]
            diff = np.linalg.norm(getattr(self, pop).residues - naive, axis=1)
            self.history[pop]['init_distance'].append(diff)

        # Binding energies, virus to plasma cells
        bindings = self.viruses.get_binding_energies(self.plasma_bcells).T
        self.history['viruses']['binding'].append(
            self.viruses.weighted_mean(bindings, self.plasma_bcells.nclones)
        )

        # Binding energy before and after recombination
        self.history['viruses']['prerecomb_E'].append(self.viruses.prerecomb_E)
        self.history['viruses']['postrecomb_E'].append(self.viruses.postrecomb_E)
        self.history['viruses']['recomb_times'].append(self.viruses.recomb_times)
        

        # Binding energies, Bcells to virus
        gc_binding = []
        for i in range(self.numGCs):
            bindings = self.gc_bcells[i].get_binding_energies(self.viruses).T
            gc_binding.append(
                self.gc_bcells[i].weighted_mean(bindings, 
                                                self.viruses.nclones)
            )
        self.history['gc_bcells']['binding'].append(gc_binding)

        for pop in ['plasma_bcells', 'memory_bcells']:
            bindings = getattr(self, pop).get_binding_energies(self.viruses).T
            self.history[pop]['binding'].append(
                getattr(self, pop).weighted_mean(
                    bindings, self.viruses.nclones
                )
            )

        # Binding energies, Bcells to panel viruses
        gc_binding = []
        for i in range(self.numGCs):
            bindings = self.gc_bcells[i].get_binding_energies(self.panel_viruses).T
            gc_binding.append(
                self.gc_bcells[i].weighted_mean(bindings, 
                                                self.panel_viruses.nclones)
            )
        self.history['gc_bcells']['binding_panel'].append(gc_binding)

        for pop in ['plasma_bcells', 'memory_bcells']:
            bindings = getattr(self, pop).get_binding_energies(self.panel_viruses).T
            self.history[pop]['binding_panel'].append(
                getattr(self, pop).weighted_mean(
                    bindings, self.panel_viruses.nclones
                )
            )

        # Breadth, Bcells to panel viruses
        gc_binding = []
        for i in range(self.numGCs):
            bindings = self.gc_bcells[i].get_binding_energies(self.panel_viruses).T
            breadths = []
            for threshold in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
                if bindings.size > 0:
                    breadth = (bindings > threshold).sum(axis=0) / bindings.shape[0]
                else:
                    breadth = []
                breadths.append(breadth)
            gc_binding.append(breadths)
        self.history['gc_bcells']['breadth'].append(gc_binding)

        for pop in ['plasma_bcells', 'memory_bcells']:
            bindings = getattr(self, pop).get_binding_energies(self.panel_viruses).T
            breadths = []
            for threshold in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
                if bindings.size > 0:
                    breadth = (bindings > threshold).sum(axis=0) / bindings.shape[0]
                else:
                    breadth = []
                breadths.append(breadth)
            self.history[pop]['breadth'].append(breadths)
            
        # Binding energies, viruses to panel bnabs
        bindings = self.viruses.get_binding_energies(self.panel_bnabs).T
        self.history['viruses']['binding_panel'].append(
            self.viruses.weighted_mean(bindings, self.panel_bnabs.nclones)
        )
            
        # Residues, plasma cells
        residues = self.plasma_bcells.residues
        self.history['plasma_bcells']['residues'].append(residues)
        
        # Residues of panel viruses
        self.history['panel_viruses'] = self.panel_viruses.residues
        
        # Residues of panel bnabs
        self.history['panel_bnabs'] = self.panel_bnabs.residues
                                                         

    def run(self):

        self.create_populations()
        self.print_seeding_ncells = 0
        self.print_seeding_mcells = 0

        for _ in range(self.timesteps):

            self.run_timestep()
            self.current_time += self.dt

            if np.isclose(self.current_time, round(self.current_time)):

                if round(self.current_time) % self.history_timestep == 0:
                    self.update_history()
                    file = f'pkl/history{self.founder_distance}_{self.random_seed}.pkl'
                    pickle.dump(self.history, open(file, 'wb'))

                string = f't = {round(self.current_time)} days'
                seeding = f"{self.print_seeding_ncells} seed"
                mseeding = f"{self.print_seeding_mcells} mem"
                self.print_seeding_ncells = 0
                self.print_seeding_mcells = 0

                virus_clones = self.viruses.nclones.size
                virus_num = self.viruses.nclones.sum()
                gc_clones = sum([gc.nclones.size for gc in self.gc_bcells])
                gc_num = sum([gc.nclones.sum() for gc in self.gc_bcells])
                pc = self.plasma_bcells.nclones.sum()
                mem = self.memory_bcells.nclones.sum()
                string = (f"{string} | {seeding} | {mseeding} | "
                          f"{self.calc_memory_fraction():.3f} | "
                          f"v {virus_clones}-{virus_num} | "
                          f"gc {gc_clones}-{gc_num} | pc {pc} | mem {mem}")
                print(string)
