"""
scan_mu.py
This script scans over a range of `mu` values and system sizes (`n_spins`),
evaluating the phase of the "Renyi charged modular commutator" \mathcal{S}_{\mu,1}
using the `evaluate_list_of_perms` function from the `semion_sampler` module.
Usage:
    python scan_mu.py <base_n_samples> <n_mus> <min_size> <max_size> <d_size> <n_chains> <n_phi_samples>
Arguments:
    base_n_samples (float): The base number of samples used for the evaluation. The actual number 
                            of samples is scaled exponentially with sqrt(n_spins).
                            For n_spins=16, I used base_n_samples=4e7
    n_mus (int): The number of `mu` values to evaluate between 0 and π (excluding 0, including π).
    min_size (int): The minimum size of the system (`n_spins`) to evaluate.
    max_size (int): The maximum size of the system (`n_spins`) to evaluate.
    d_size (int): The step size for the range of system sizes (`n_spins`).
    n_chains (int): The number of Markov chains used in the evaluation.
    n_phi_samples (int): The number of φ samples used in the evaluation. This correspond
        to random rotations of the sphere, to consider different partitions of the system.
Example:
    To run the script with specific parameters:
    ```
    python3 scan_mu.py 4e7 16 16 20 2 2000 10
    ```
    This run required #TODO: insert value hours on 3 GPUs.
"""
import argparse
from typing import Union
import warnings
import semion_sampler
from semion_sampler import ValueWithError, evaluate_list_of_perms
import numpy as np
import time

if __name__ == "__main__":
    # parses input of the form
    # Argument parser
    parser = argparse.ArgumentParser(description='Topological twist')
    parser.add_argument('base_n_samples', type=float, help='Number of samples')
    parser.add_argument('n_mus', type=int, help='Number of mu values between 0 and pi')
    parser.add_argument('min_size', type=int, help='Minimum size')
    parser.add_argument('max_size', type=int, help='Maximum size')
    parser.add_argument('d_size', type=int, help='Size step')
    parser.add_argument('n_chains', type=int, help='Number of chains')
    parser.add_argument('n_phi_samples', type=int, default=1)
    args = parser.parse_args()
    
    base_n_samples = args.base_n_samples
    n_chains = args.n_chains
    n_min = args.min_size
    n_max = args.max_size
    d_n = args.d_size
    n_phi_samples = args.n_phi_samples
    n_mus = args.n_mus
    average_phase = False
    
    errors = []
    n_range = [n for n in range(n_min, n_max+1, d_n)]
    mu_range = np.linspace(0, np.pi, n_mus+1)[1:]  # Exclude 0, include pi
    results = np.zeros((len(n_range), len(mu_range)))
    errors = np.zeros((len(n_range), len(mu_range)))
    print(f"Started running with base_n_samples: {base_n_samples}, n_min: {n_min}, n_max: {n_max}, d_n: {d_n}, n_chains: {n_chains}, n_phi_samples: {n_phi_samples}, n_mus: {n_mus}")
    total_time = time.time()
    for i,n in enumerate(n_range):
        for j, mu in enumerate(mu_range):
            print(f"n_spins: {n}")
            n_samples = np.round(
                np.exp(3*(np.sqrt(n)-np.sqrt(n_min)))*base_n_samples)
            start_time = time.time()
            perms =[[[0,1],[1,0],[1,0],[0,1]]] # -> \pi_{BC} cyclic perm gives \pi_{AB}
            mus = [(mu,mu,0,0)] # -> Q_{AB} cyclic perm gives Q_{AC}
            evaluation_results = evaluate_list_of_perms(
                n, perms, n_samples=n_samples,n_chains=n_chains, n_phi_samples=n_phi_samples, mus_list=mus)

            evaluation_results = [res[0] for res in evaluation_results]
            end_time = time.time()
            if average_phase:
                phase_results = [ValueWithError(np.angle(res.value), res.error/np.abs(res.value)) for res in evaluation_results]
                warnings.warn(f"phases: {phase_results}")
                phase = sum(phase_results) / len(phase_results)
                phase.error = np.sqrt(sum([(res.value-phase.value)**2 for res in phase_results]))/len(phase_results)
            else:
                # In this case, we first average, then compute the phase
                result = sum(evaluation_results) / len(evaluation_results)
                # Compute the error as the standard error of the results
                result.error = np.sqrt(sum([(res.value-result.value)**2 for res in evaluation_results]))/len(evaluation_results)
                warnings.warn(f"results: {evaluation_results}")
                phase = ValueWithError(np.angle(result.value), result.error/np.abs(result.value))

            print(phase)
            print(f"angle: {phase.value} +- {phase.error}")
            # The cluster only prints warnings during the computation
            warnings.warn(f"n_spins = {n}, mu={mu}, angle: {phase.value} +- {phase.error}, time: {end_time - start_time:.2f} seconds")
            results[i, j] = phase.value
            errors[i, j] = phase.error
    print(f"Total time taken: {time.time() - total_time:.2f} seconds")
    print("******************Results******************")
    print(f'mus: {mu_range}')
    print(f'results: {results}')
    print(f'errors: {errors}')