# Model of the humoral immune response in people with HIV

This repository contains the code and scripts in[^1] for running agent-based stochastic
simulations of the humoral immune response to broadly
neutralizing antibody therapy in people living with HIV. 
Our simulation algorithm is adapated from [^2], which investigated the 
antibody response to sequential immunization with SARS-Cov-2 vaccines. 

The module `hiv_code` contains the simulation code, and `scripts` contains a set
of scripts for running parameter sweeps on a SLURM scheduler.

Use with a clean conda environment created from environment.yml.
