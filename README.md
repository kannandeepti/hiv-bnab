# hiv_bnab: Model of the humoral immune response in people with HIV

> Deepti Kannan, MIT (dkannan@mit.edu)

> Eric Wang, MIT 

This repository contains the code and scripts in[^1] for running agent-based stochastic
simulations of the humoral immune response to broadly
neutralizing antibody therapy in people living with HIV. 
Our simulation algorithm is adapated from [^2], which investigated the 
antibody response to sequential immunization with SARS-Cov-2 vaccines. 

The module `hiv_code` contains the simulation code, and `scripts` contains a set
of scripts for running parameter sweeps on a SLURM scheduler and analyzing simulation output. The
`simulations_for_paper` directory contains input yaml files for each of the parameter
sweeps run in Kannan et al. (2025).

## Installation

Clone the repo and navigate to the project directory. Then run
```bash
conda env create -f environment.yml
python -m pip install -e .
```

## Reproducing paper results

To reproduce our results, follow these steps:
1. Edit the configs/scripts_config.yaml and configs/shared_epitope_config.yaml with your
   local paths and information specific to your slurm HPC cluster. 
2. Run 
   ```bash
   python scripts/submit_job.py --config_file configs/scripts_config.yaml --sweep_name <sweep_name>
   ```
   for each parameter sweep listed inside `simulations_for_paper` separately.
3. To reproduce the results from varying the number of shared epitopes across two
   antigens (Figure 4), run
   ```bash
   python scripts/shared_epitopes.py --config_file configs/scripts_config.yaml --sweep_name 12_epitope_sweep
   ```
4. To generate the pie charts in Figure 5, run 
   ```bash
   python scripts/clonal_diversity.py --config_file configs/scripts_config.yaml --param_dir masking_sweep_C0_1.0/sweep_1 --title <plot_title>
   ```
   Note that this script requires more memory, so run on an exclusive node if possible.
5. To plot and animate affinity distributions (as in Figure S1), run
   ```bash
   python scripts/animate_affinity_distributions.py --config_file configs/scripts_config.yaml --param_dir masking_sweep_C0_1.0/sweep_1 --title <plot_title>
   ```
5. Reproduce paper figures using notebooks/paper_figures.ipynb (edit paths to match your
   local setup). 

## References

[^1]: D. Kannan, E. Wang, S. G. Deeks, S. R. Lewin, and A. K. Chakraborty. Mechanism for 
evolution of diverse autologous antibodies upon broadly neutralizing antibody therapy of people with HIV. *bioRxiv* (2025) https://doi.org/10.1101/2022.12.24.521789

[^2]: L. Yang, M. Van Beek, Z. Wang, F. Muecksch, M. Canis, T. Hatziioannou, P. D. Bieniasz, M. Nussenzweig, and A. K. Chakraborty.
Antigen presentation dynamics shape the antibody response to variants like SARS-CoV-2 Omicron after multiple vaccinations with the original strain.
*Cell Reports* (2023) https://doi.org/10.1016/j.celrep.2023.112256
