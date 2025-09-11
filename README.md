# Model of the humoral immune response in people with HIV

This repository contains the code and scripts in[^1] for running agent-based stochastic
simulations of the humoral immune response to broadly
neutralizing antibody therapy in people living with HIV. 
Our simulation algorithm is adapated from [^2], which investigated the 
antibody response to sequential immunization with SARS-Cov-2 vaccines. 

The module `hiv_code` contains the simulation code, and `scripts` contains a set
of scripts for running parameter sweeps on a SLURM scheduler. The
`simulations_for_paper` directory contains input yaml files for each of the parameter
sweeps run in Kannan et al. (2025). 

To reproduce our results, follow these steps:
1. Create a clean conda environment from environment.yml.
2. Modify the SWEEP_DIR, PLOT_DIR, and LOG_DIR variables in scripts/__init__.py to your
   local paths. SWEEP_DIR is the parent directory containing all simulation sweeps, i.e.
   `simulations_for_paper`.
3. Edit the scripts/submit_job.py to match your SLURM cluster specifications.
4. Run scripts/submit_job.py <sweep_name> for each sweep inside of SWEEP_DIR separately.
5. Reproduce paper figures using notebooks/paper_figures.ipynb. 

[^1]: D. Kannan, E. Wang, S. G. Deeks, S. R. Lewin, and A. K. Chakraborty. Mechanism for 
evolution of diverse autologous antibodies upon broadly neutralizing antibody therapy of people with HIV. *bioRxiv* (2025) https://doi.org/10.1101/2022.12.24.521789

[^2]: L. Yang, M. Van Beek, Z. Wang, F. Muecksch, M. Canis, T. Hatziioannou, P. D. Bieniasz, M. Nussenzweig, and A. K. Chakraborty.
Antigen presentation dynamics shape the antibody response to variants like SARS-CoV-2 Omicron after multiple vaccinations with the original strain.
*Cell Reports* (2023) https://doi.org/10.1016/j.celrep.2023.112256
