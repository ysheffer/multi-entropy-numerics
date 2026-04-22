import argparse
import semion_sampler

if __name__ == "__main__":
    # parses input of the form
    # python run_semion_sampler.py 2 2 10000 10 0.3 10 20 2 10 20 2 .3
    # Argument parser
    parser = argparse.ArgumentParser(description='Topological twist')
    parser.add_argument('r', type=int, help='Number of replicas')
    parser.add_argument('base_n_samples', type=float, help='Number of samples')
    parser.add_argument('n_chains', type=int, help='Number of chains')
    parser.add_argument('min_size', type=int, help='Minimum size')
    parser.add_argument('max_size', type=int, help='Maximum size')
    parser.add_argument('d_size', type=int, help='Size step')
    parser.add_argument('norms_n_samples_factor', type=float, default=.3,
                        help='Factor to scale the number of samples for the norms')
    args = parser.parse_args()

    # Run the scaling
    sizes, results = semion_sampler.topo_twist_size_scaling(
        args.r, args.min_size, args.max_size, args.d_size, args.base_n_samples,
        norms_n_samples_factor=args.norms_n_samples_factor,
        n_samples=args.base_n_samples, n_chains=args.n_chains)
    print(sizes, results)
