# Multi Entropy Numerics

This repository is a curated public subset extracted from a larger research workspace.
It is part of the paper "Probing chiral topological states with permutation defects" (arXiv:2512.04649).

## Contents
- kitaev_honeycomb/: Julia code and notebook for Kitaev honeycomb model calculations.
- semion/: Python scripts and notebook for semion sampling and scans.

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
