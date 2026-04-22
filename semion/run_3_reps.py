import argparse
from typing import Union
import warnings
import semion_sampler
from semion_sampler import ValueWithError, evaluate_list_of_perms
import numpy as np
import time

if __name__ == "__main__":
    # parses input of the form
    # python run_3_reps.py <base_n_samples> <min_size> <max_size> <d_size> <n_chains> [<n_phi_samples>]
    # Argument parser
    parser = argparse.ArgumentParser(description='Topological twist')
    parser.add_argument('base_n_samples', type=float, help='Number of samples')
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
    average_phase = False
    
    results = []
    errors = []
    print(f"Started running with base_n_samples: {base_n_samples}, n_min: {n_min}, n_max: {n_max}, d_n: {d_n}, n_chains: {n_chains}, n_phi_samples: {n_phi_samples}")
    total_time = time.time()
    for n in range(n_min, n_max+1, d_n):
        print(f"n_spins: {n}")
        n_samples = np.round(
            np.exp(3*(np.sqrt(n)-np.sqrt(n_min)))*base_n_samples)
        start_time = time.time()
        evaluation_results = evaluate_list_of_perms(
            n, [[[1,2,0], [1,0,2], [0,2,1], [0, 1, 2]]], n_samples=n_samples,n_chains=n_chains, n_phi_samples=n_phi_samples)
                    # n, [[[1, 0, 2], [2, 1, 0], [1, 2, 0], [0, 1, 2]]], n_samples=n_samples,n_chains=n_chains, n_phi_samples=n_phi_samples)

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
            warnings.warn(f"results: {evaluation_results}")
            phase = ValueWithError(np.angle(result.value), result.error/np.abs(result.value))

        print(phase)
        print(f"angle: {phase.value} +- {phase.error}")
        # The cluster only prints warnings during the computation
        warnings.warn(f"n_spins = {n}, angle: {phase.value} +- {phase.error}, time: {end_time - start_time:.2f} seconds")
        results.append(phase.value)
        errors.append(phase.error)
    print(f"Total time taken: {time.time() - total_time:.2f} seconds")
    print("******************Results******************")
    print(results)