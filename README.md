# Model of the humoral immune response in people with HIV

This repository contains the code and scripts in[^1] for running agent-based stochastic
simulations of the humoral immune response to broadly
neutralizing antibody therapy in people living with HIV. 
Our simulation algorithm is adapated from [^2], which investigated the 
antibody response to sequential immunization with SARS-Cov-2 vaccines. 

The module `hiv_code` contains the simulation code, and `scripts` contains a set
of scripts for running parameter sweeps on a SLURM scheduler.

Use with a clean conda environment created from environment.yml.

[^1]: D. Kannan, E. Wang, S. G. Deeks, S. R. Lewin, and A. K. Chakraborty. Mechanism for 
evolution of diverse autologous antibodies upon broadly neutralizing antibody therapy of people with HIV. *bioRxiv* (2025) https://doi.org/10.1101/2022.12.24.521789

[^2]: L. Yang, M. Van Beek, Z. Wang, F. Muecksch, M. Canis, T. Hatziioannou, P. D. Bieniasz, M. Nussenzweig, and A. K. Chakraborty.
Antigen presentation dynamics shape the antibody response to variants like SARS-CoV-2 Omicron after multiple vaccinations with the original strain.
*Cell Reports* (2023) https://doi.org/10.1016/j.celrep.2023.112256
