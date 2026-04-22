# Multi Entropy Numerics

This repository contains all code used to produce the figures in the paper "Probing chiral topological states with permutation defects" (arXiv:2512.04649). 
There are essentially two independent packages here, one for the Kitaev Honeycomb model (free fermion simulations) and one for Monte-Carlo sampling of a semion wavefunciton. Both provide functionality for computing multi-entropy expectation values on said wavefunctions. 

The code provided here was written for internal use, and is very much not properly documented. If you are interested in using this for your project, feel free to reach out for any question!

## Contents
- kitaev_honeycomb/: Julia code and notebook for Kitaev honeycomb model calculations.
    These are free-fermion calculation, and can be run effectively on a personal laptop up to large (n=10) number of replicas.
- semion/: Python scripts and notebook for semion sampling and scans.
    This provides Monte-Carlo sampling (using the NetKet interface) for a Laughlin $\nu=1/2$ wavefunction. As a result of the exponential time complexity, this calculation is feasable only for low (n=3) number of replicas, and was run on parallel GPUs.

## Structure
- kitaev_honeycomb/
  - KitaevHoneycomb.jl
  - TableOfValues.jl
  - TCIsingTransition.jl
  - plot_results.ipynb
  - Project.toml
  - Manifest.toml
- semion/
  - semion_sampler.py
  - run_semion_sampler.py
  - run_3_reps.py
  - scan_mu.py
  - plot_semion_results.ipynb
  - requirements.txt

## Quick Start
### Kitaev Honeycomb (Julia)
1. Install Julia.
2. In kitaev_honeycomb/, instantiate the environment:
   julia --project=. -e 'using Pkg; Pkg.instantiate()'

### Semion (Python)
1. Create and activate a Python environment.
2. Install dependencies:
   pip install -r semion/requirements.txt
3. Run examples:
   python semion/run_semion_sampler.py <args>
   python semion/run_3_reps.py <args>

## Provenance
Files were copied from an internal research repository on 2026-04-22.
