from collections import namedtuple
import argparse
from collections.abc import Iterable
from functools import partial
import gc
from typing import List, NamedTuple, Union
import warnings
from tqdm import tqdm
import time
from netket.utils import struct
import jax
import jax.experimental
import jax.experimental.checkify
import netket as nk
import json
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import flax
import flax.linen as nn


class ValueWithError:
    def __init__(self, value, error):
        self.value = value
        self.error = error

    def __str__(self):
        return f"{self.value} +- {self.error}"

    def __repr__(self):
        return f"{self.value} +- {self.error}"

    def __add__(self, other):
        if isinstance(other, ValueWithError):
            return ValueWithError(self.value + other.value, np.sqrt(self.error**2 + other.error**2))
        else:
            return ValueWithError(self.value + other, self.error)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ValueWithError):
            return ValueWithError(self.value - other.value, np.sqrt(self.error**2 + other.error**2))
        else:
            return ValueWithError(self.value - other, self.error)
    
    def rsub__(self, other):
        if isinstance(other, ValueWithError):
            return ValueWithError(other.value - self.value, np.sqrt(self.error**2 + other.error**2))
        else:
            return ValueWithError(other - self.value, self.error)

    def __mul__(self, other):
        if isinstance(other, ValueWithError):
            return ValueWithError(self.value * other.value, np.sqrt(np.abs(self.error*other.value)**2 + np.abs(other.error*self.value)**2))
        else:
            return ValueWithError(self.value * other, self.error * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ValueWithError):
            return ValueWithError(self.value / other.value, np.sqrt(np.abs(self.error/other.value)**2 + np.abs(other.error*self.value/other.value**2)**2))
        else:
            return ValueWithError(self.value / other, self.error / other)

    def __pow__(self, other):
        if isinstance(other, ValueWithError):
            # Did not really check this part
            return ValueWithError(self.value ** other.value, np.sqrt(np.abs(other.value*self.value**(other.value-1)*self.error)**2 + np.abs(other.error*self.value**other.value*np.log(self.value))**2))
        else:
            return ValueWithError(self.value ** other, np.abs(other*self.value**(other-1)*self.error))


def get_angles(n_spins, random_points=False, rotation_mat=None):
    """
    Returns the angles for the spins on the sphere.

    Args:
        n_spins: int, number of spins on the sphere.

    Returns:
        theta: array of shape (n_spins) containing the theta angles.
        phi_i: array of shape (n_spins) containing the phi angles.
    """
    if random_points:
        # Fix the seed for reproducibility
        np.random.seed(0)
        # Generate random points on the sphere
        r = np.random.normal(size=(n_spins, 3))
        r = r/np.linalg.norm(r, axis=-1)[:, None]
        theta = np.arccos(r[:, 2])
        phi_i = np.arctan2(r[:, 1], r[:, 0])
    else:
        phi = np.pi*(3-np.sqrt(5))
        theta = np.arccos(1-2*np.arange(n_spins)/(n_spins-1))
        phi_i = np.mod(phi*np.arange(n_spins), 2*np.pi)
        if rotation_mat is not None:
            # Rotate the angles using the rotation matrix
            theta, phi_i = _rotate_angles(theta, phi_i, rotation_mat)
    return theta, phi_i

def _rotate_angles(theta,phi,rotation_mat):
    """
    Rotates the angles theta and phi using the rotation matrix.
    
    Args:
        theta: array of shape (n_spins) containing the theta angles.
        phi: array of shape (n_spins) containing the phi angles.
        rotation_mat: 2D array of shape (3, 3) representing the rotation matrix.
    
    Returns:
        rotated_theta: array of shape (n_spins) containing the rotated theta angles.
        rotated_phi: array of shape (n_spins) containing the rotated phi angles.
    """
    vecs_3d = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]).T  # shape (n_spins, 3)
    rotated_vecs_3d = vecs_3d @ rotation_mat.T  # shape (n_spins, 3)
    rotated_theta = np.arccos(rotated_vecs_3d[:, 2])  # theta from z-axis
    rotated_phi = np.arctan2(rotated_vecs_3d[:, 1], rotated_vecs_3d[:, 0])  # phi from x-axis
    # ensure that phi is in the range (0,2pi)
    rotated_phi = np.mod(rotated_phi, 2*np.pi)

    return rotated_theta, rotated_phi


def get_sphere_distance(theta1, phi1, theta2, phi2):
    """
    Returns the distance between two points on the sphere.

    Args:
        theta1: float, theta angle of the first point.
        phi1: float, phi angle of the first point.
        theta2: float, theta angle of the second point.
        phi2: float, phi angle of the second point.
    """
    return np.arccos(np.round(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2), 6))


class LaughlinSphere(nn.Module):
    n_sites: int
    n_replicas: int
    rotation_mat: Union[jnp.ndarray, None] = struct.field(
        pytree_node=False,
        hash=False,          # <-- prevent hashing
        compare=False,       # <-- keep out of __eq__
        repr=False,
        default=None,
    )
    # all the options above are used to allow ndarray without hashing it 
    
    def setup(self):
        self.z_vals = self._z_vals_def()
        self.log_z_mat = self._log_z_mat_def()
    
    def __hash__(self):
        return hash((self.n_sites, self.n_replicas, tuple(self.rotation_mat.flatten())))

    def _z_vals_def(self):
        """
        Returns the z value of the i-th site on the sphere."""
        theta, phi_i = get_angles(self.n_sites, rotation_mat=self.rotation_mat)
        z_vals = np.sin(theta)/(1+np.cos(theta))*np.exp(1j*phi_i)
        return z_vals

    def _log_z_mat_def(self):
        z = self.z_vals
        log_z_mat = np.log(z[:, np.newaxis] - z[np.newaxis, :]+1e-15)
        # zero elements below the diagonal such that each pair is counted once
        log_z_mat = np.where(np.tril(np.ones_like(log_z_mat), -1),
                             log_z_mat, 0.)
        return log_z_mat

    @nn.compact
    def __call__(self, x):
        """
        Returns the log probability of the configuration x.

        Args:
            x: array of shape (n_replicas*n_sites)
                containing the configuration.
        """
        x = jnp.reshape(x, (-1, self.n_replicas, self.n_sites))
        log_z_mat = self.log_z_mat

        # Remove configurations where the sum of the spins in a replica is not zero
        infs = jnp.where(jnp.any(jnp.sum(x, axis=-1) != 0, axis=-1),
                         jnp.inf, 0.)
        # The sum is multiplied by alpha=1/2 to obtain a Laughlin wavefunction
        out = jnp.sum(x[:, :, :, None]*x[:, :, None, :] *
                      log_z_mat[None, None, :, :], axis=(-1, -2, -3))/2 - infs

        # Returns the log probability
        return out


def get_hilbert_space(n_replicas, n_spins):
    # Define the hilbert space
    # the hilbert space is constrained to have total_sz=0 in each replica
    hilberts = [nk.hilbert.Spin(s=1/2, N=n_spins, total_sz=0)
                for _ in range(n_replicas)]
    return nk.hilbert.TensorHilbert(*hilberts)


def get_sampler(n_replicas, n_spins, hilbert, n_chains=20, rotation_mat=None):
    # define clusters for the samples
    clusters = []
    dist_const = 1.2*np.pi
    thetas, phis = get_angles(n_spins, rotation_mat=rotation_mat)
    inds_list = np.reshape(
        np.arange(n_replicas * n_spins), (n_replicas, n_spins))
    for r in range(n_replicas):
        for i in range(n_spins):
            for j in range(i, n_spins):
                if get_sphere_distance(thetas[i], phis[i], thetas[j], phis[j]) < dist_const/np.sqrt(n_spins):
                    clusters.append((inds_list[r, i], inds_list[r, j]))

    # Define the sampler
    rule = nk.sampler.rules.ExchangeRule(clusters=clusters)
    sampler = nk.sampler.MetropolisSampler(
        hilbert, rule, n_chains=n_chains)
    return sampler


class RegionSwapOperator(nk.operator.AbstractOperator):
    def __init__(self, n_spins, n_replicas, hilbert, perms=(None, None, None, None), mus=(None,None,None,None),
                 max_theta=None, min_theta=None, rotation_mat=None):
        self.n_spins = n_spins
        self.n_replicas = n_replicas
        self.max_theta = max_theta if max_theta is not None else np.pi*2/3
        self.min_theta = min_theta if min_theta is not None else 0
        self.rotation_mat = rotation_mat
        assert self.min_theta < self.max_theta
        self.perms = self._define_perms(perms)
        self.inv_perms = self._define_inv_perms(self.perms)
        self.mus = tuple(mu if mu is not None else 0. for mu in mus)
        self.size = n_spins*n_replicas
        self.regions = self._define_regions()
        self._hilbert = hilbert

    @property
    def dtype(self):
        return float

    def _define_regions(self):
        theta, phi_i = get_angles(self.n_spins, rotation_mat=self.rotation_mat)
        max_theta = self.max_theta
        # mimial theta allows the regions to be defined as an annulus
        min_theta = self.min_theta
        in_a = (theta >= min_theta) & (theta <= max_theta) & (
            0 <= phi_i) & (phi_i <= np.pi*2/3)
        in_b = (theta >= min_theta) & (theta <= max_theta) & (np.pi *
                                                              2/3 < phi_i) & (phi_i <= np.pi*4/3)
        in_c = (theta >= min_theta) & (
            theta <= max_theta) & (np.pi*4/3 < phi_i)
        in_d = np.logical_or(theta > max_theta, theta < min_theta)

        # make sure that all spins are in one of the regions
        # and that regions are mutually exclusive
        assert np.all(in_a | in_b | in_c | in_d)
        assert np.all(in_a == np.bitwise_not(in_b | in_c | in_d))
        assert np.all(in_b == np.bitwise_not(in_a | in_c | in_d))
        assert np.all(in_c == np.bitwise_not(in_a | in_b | in_d))
        assert np.all(in_d == np.bitwise_not(in_a | in_b | in_c))
        print(f"Obtained {sum(in_a)} spins in region A, {sum(in_b)} spins in region B, ",
              f"{sum(in_c)} spins in region C, {sum(in_d)} spins in region D")
        return in_b + 2*in_c + 3*in_d

    def _define_perms(self, perms):
        id = np.arange(self.n_replicas)
        return tuple(perm if perm is not None else id for perm in perms)
    
    def _define_inv_perms(self, perms):
        inv_perms = []
        for perm in perms:
            inv_perm = np.zeros_like(perm,dtype=int)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            assert np.all(perm[inv_perm] == np.arange(len(perm)))
            inv_perms.append(inv_perm)
        return tuple(inv_perms)

    def _define_perm_array(self):
        # Define the array that will be used to permute the spins
        # the spin vector will be permuted by x = x[perm_array]
        n_replicas = self.n_replicas
        n_spins = self.n_spins
        arr_in = np.reshape(np.arange(n_replicas*n_spins,
                         dtype=int), (n_replicas, n_spins))
        arr_out = np.zeros_like(arr_in)

        for i in range(n_spins):
            # Use the inverse permutation to define the permuted array
            # This is because we have x_new = x_old[perm], so to get x_old we need to use the inverse perm
            arr_out[:, i] = arr_in[self.inv_perms[self.regions[i]], i]
        arr_out = arr_out.flatten()
        return arr_out
    
    def _define_mus_array(self):
        n_replicas = self.n_replicas
        n_spins = self.n_spins
        arr_in = np.reshape(np.arange(n_replicas*n_spins,
                         dtype=int), (n_replicas, n_spins))
        mus_arr = np.zeros_like(arr_in, dtype=float)
        # gauge is applied only on the last replica
        for i in range(n_spins):
            mus_arr[-1, i] = self.mus[self.regions[i]]
        mus_arr = mus_arr.flatten()
        return mus_arr

    @property
    def vars(self):
        return [self.n_replicas, self.n_spins, self.perms, self.size, self.regions]

    @property
    def op_fn(self):
        perm_array = self._define_perm_array()
        mus_array = self._define_mus_array()

        @jax.vmap
        def call_swap_and_charge(x):
            """
            Applies the swap and charge operator to the configuration x. Using the information of the operator.
            This operator first applies swap, then charge, corresponding to <\psi|e^{mu Q_R1} \rho_R2|\psi>.
            """
            assert x.ndim == 1
            x = x[perm_array]
            # A spin +1 has charge 1 no spin
            phase = jnp.exp(jnp.sum(mus_array*(x/2), axis=-1))

            return x, phase
        return call_swap_and_charge


# def e_loc(logpsi, pars, sigma, *extra_args, **kwargs):
#     eta = extra_args[0]
#     assert sigma.ndim == 2
#     assert eta.ndim == 2
#     # let's write the local energy assuming a single sample, and vmap it

#     @jax.vmap
#     def _loc_vals(sigma, eta):
#         lp_sigma = logpsi(pars, sigma)
#         lp_eta = logpsi(pars, eta)
#         return jnp.where(jnp.isfinite(lp_eta), jnp.exp(lp_eta - lp_sigma), 0.0)

#     return _loc_vals(sigma, eta)


# @nk.vqs.get_local_kernel.dispatch
# def get_local_kernel(vstate: nk.vqs.MCState, op: RegionSwapOperator, chunk_size: int):
#     return e_loc


# @nk.vqs.get_local_kernel_arguments.dispatch
# def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: RegionSwapOperator):
#     sigma = vstate.samples
#     # get the connected elements. Reshape the samples because that code only works
#     # if the input is a 2D matrix
#     extra_args = op.op_fn(sigma.reshape(-1, vstate.hilbert.size))
#     return sigma, extra_args

@partial(jax.vmap, in_axes=(None, None, 0, 0))
def e_loc(logpsi, pars, sigma, eta):
    # jax.debug.print("eta: {eta}, Logpsi eta: {x}",
    #                 eta=eta, x=logpsi(pars, eta))
    lp_eta = logpsi(pars, eta)
    lp_sigma = logpsi(pars, sigma)
    return jnp.where(jnp.isfinite(lp_eta), jnp.exp(lp_eta - lp_sigma), 0.0)


class Results:
    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.error_of_mean = 0

    def __repr__(self):
        return f"{self.mean} +- {self.error_of_mean} (Variance: {self.variance}, relative error: {self.error_of_mean/abs(self.mean)})"


# @nk.vqs.expect.dispatch
# def expect_swap_operator(vstate: nk.vqs.MCState, op: RegionSwapOperator, chunk_size: None):
#     op_fn = op.op_fn
#     n_samples = vstate.n_samples
#     # chunk_size = n_samples  # min(10000, n_samples)
#     chunk_size = min(1000*vstate.sampler.n_chains, n_samples)
#     assert n_samples % chunk_size == 0

#     results = Results()
#     n_chunks = n_samples//chunk_size
#     err_sqrd = 0.0
#     for i in range(n_chunks):
#         sigma = vstate.sample(n_samples=chunk_size)
#         res_i = _expect(vstate._apply_fun, op_fn, {}, sigma)
#         results.mean += res_i.mean
#         results.variance += res_i.variance
#         err_sqrd += res_i.error_of_mean**2
#     results.mean /= n_chunks
#     results.variance /= n_chunks
#     results.error_of_mean = jnp.sqrt(err_sqrd)/n_chunks
#     return results

@nk.vqs.expect.dispatch
def expect_swap_operator(vstate: nk.vqs.MCState, op: RegionSwapOperator, chunk_size: None):
    op_fn = op.op_fn
    n_samples = vstate.n_samples
    # chunk_size = n_samples  # min(10000, n_samples)
    chunk_size = min(1000*vstate.sampler.n_chains, n_samples)

    results = Results()
    remaining_samples = n_samples
    while remaining_samples > 0:
        current_chunk_size = min(chunk_size, remaining_samples)
        sigma = vstate.sample(n_samples=current_chunk_size)
        res_i = _expect(vstate._apply_fun, op_fn, {}, sigma)
        results.mean += res_i.mean * current_chunk_size
        results.variance += res_i.variance * current_chunk_size
        remaining_samples -= current_chunk_size
    # TODO: delete following
    assert remaining_samples == 0
    results.mean /= n_samples
    results.variance /= n_samples
    results.error_of_mean = jnp.sqrt(results.variance/n_samples)
    return results


@partial(jax.jit, static_argnums=(0, 1))
def _expect(logpsi, op_fn, variables, sigma):
    """Computes the expectation value of the operator on the given samples.
    Args:
        logpsi: function, the wavefunction to evaluate.
        op_fn: function, the operator function to apply.
        variables: dict, the variables for the wavefunction.
        sigma: array(n_chains, n_spins), the samples to evaluate.
    """
    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    # flatten all batches
    sigma = sigma.reshape(-1, N)

    eta, phase = op_fn(sigma)

    E_loc = e_loc(logpsi, variables, sigma, eta)*phase

    # reshape back into chains to compute statistical information
    E_loc = E_loc.reshape(-1, n_chains)

    # this function computes things like variance and convergence information.
    return nk.stats.statistics(E_loc)

def _get_random_rotation():
    # Generates a random rotation/reflection matrix using the QR decomposition
    random_matrix = np.random.randn(3, 3)
    q, r = np.linalg.qr(random_matrix)
    return q  # The orthogonal matrix q is the rotation matrix

def evaluate_list_of_perms(n_spins, perms_list, n_samples=1e4, n_chains=10, max_theta=np.pi*2/3, n_phi_samples=1,
                           min_theta=None, mus_list=None, **args) -> List[ValueWithError]:
    """
    Evaluates the expectation value of the swap operator for a list of permutations.

    Args:
        n_spins: int, number of spins on the sphere.
        n_replicas: int, number of replicas.
        perms_list: list of permutations. Each element of the list is a 4-tuple of permutations.
            Each permutation is a list of n_replicas elements.
        mus_list: list of mu (phases of charge operator) values, given as a list of 4-tuple of floats.
        n_samples: (int|list), number of samples to evaluate the expectation value, or a list of number of samples for each permutation.
        n_phi_samples: int, number of samples for the random rotations.

    Returns:
        list of expectation values for each permutation.
    """
    res = []
    if not isinstance(n_samples, Iterable):
        n_samples = [n_samples]*len(perms_list)

    for phi_iter in range(n_phi_samples):
        # rotation matrix is generated randomly for each iteration
        # need to ensure that all permutations are calculated with the same rotation
        rotation_mat = _get_random_rotation()
        res.append([])
        for i, perms in enumerate(perms_list):
            # the number of replicas is obtained from the specified permutation
            print("Permutations: ", perms, "Number of samples: ", n_samples[i])
            start_time = time.time()
            n_replicas = len(perms[0])
            print(perms)
            print(n_replicas)
            mus = mus_list[i] if mus_list is not None else (None, None, None, None)
            model = LaughlinSphere(n_sites=n_spins, n_replicas=n_replicas,rotation_mat=rotation_mat)
            hilbert = get_hilbert_space(n_replicas, n_spins)
            sampler = get_sampler(n_replicas, n_spins, hilbert, n_chains=n_chains, rotation_mat=rotation_mat)
            operator = RegionSwapOperator(
                n_spins, n_replicas, hilbert, perms, mus=mus, max_theta=max_theta, min_theta=min_theta, rotation_mat=rotation_mat)
            state = nk.vqs.MCState(sampler, model=model, variables={
                "params": {}}, n_samples=int(n_samples[i]))
            result = state.expect(operator)
            print(
                f"Evaluation result: {result}, acceptance rate: {state.sampler_state.acceptance}")
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
            res[phi_iter].append(ValueWithError(result.mean, result.error_of_mean))
        gc.collect()
    return res


def get_topo_twist(n_sites, r, only_abc=False, norms_n_samples_factor=1., n_samples=1e4, norm_v2=True, **args):
    # There is currently no support for the random rotations here, and stuff will break
    raise NotImplementedError("This function is broken and needs to be fixed.")
    perm_mat = np.reshape(np.arange(2*r), (2, r))
    perm_a = np.array(
        [np.roll(perm_mat[0], 1), np.roll(perm_mat[1], -1)]).flatten()
    perm_b = np.array([perm_mat[1], perm_mat[0]]).flatten()
    perm_c = np.array(
        [np.roll(perm_mat[1], -1), np.roll(perm_mat[0], 1)]).flatten()
    perm_triv = np.arange(2*r)

    if only_abc:
        perms = [phi_abc]
        return evaluate_list_of_perms(n_sites, perms, **args)[0]

    if not norm_v2:
        phi_abc = [perm_a, perm_b, perm_c, perm_triv]
        phi_ab = [perm_a, perm_b, perm_triv, perm_triv]
        phi_ac = [perm_a, perm_triv, perm_c, perm_triv]
        phi_bc = [perm_triv, perm_b, perm_c, perm_triv]
        phi_a = [perm_a, perm_triv, perm_triv, perm_triv]
        phi_b = [perm_triv, perm_b, perm_triv, perm_triv]
        phi_c = [perm_triv, perm_triv, perm_c, perm_triv]
        n_samples = [n_samples]+[int(n_samples*norms_n_samples_factor)]*6
        perms = [phi_abc, phi_ab, phi_ac, phi_bc, phi_a, phi_b, phi_c]
        [s_abc, s_ab, s_ac, s_bc, s_a, s_b, s_c] = evaluate_list_of_perms(
            n_sites, perms, n_samples=n_samples, **args)
        s_0 = ValueWithError(1, 0)
    else:
        phi_abc = [perm_a[perm_b], perm_b[perm_b], perm_c[perm_b], perm_triv]
        phi_ab = [perm_a[perm_b], perm_b[perm_b], perm_c, perm_triv]
        phi_ac = [perm_a[perm_b], perm_b, perm_c[perm_b], perm_triv]
        phi_bc = [perm_a, perm_b[perm_b], perm_c[perm_b], perm_triv]
        phi_a = [perm_a[perm_b], perm_b, perm_c, perm_triv]
        phi_b = [perm_a, perm_b[perm_b], perm_c, perm_triv]
        phi_c = [perm_a, perm_b, perm_c[perm_b], perm_triv]
        phi_0 = [perm_a, perm_b, perm_c, perm_triv]
        normed_n_samples = int(n_samples*norms_n_samples_factor)
        n_samples = [normed_n_samples]*2 + [n_samples] + \
            [normed_n_samples]*4 + [n_samples]
        perms = [phi_abc, phi_ab, phi_ac, phi_bc, phi_a, phi_b, phi_c, phi_0]
        [s_abc, s_ab, s_ac, s_bc, s_a, s_b, s_c, s_0] = evaluate_list_of_perms(
            n_sites, perms, n_samples=n_samples, **args)

    topological_twist_op = s_0 / (s_a*s_b*s_c)*(s_ab * s_ac * s_bc)/s_abc
    warnings.warn(f'n_sites: {n_sites}, n_samples: {n_samples[0]}, '
                  f'Twist operator: {topological_twist_op}, '
                  f'Relative error: {topological_twist_op.error/abs(topological_twist_op.value)}')
    print('*****************')
    print(f"Twist operator: {topological_twist_op}")
    print(f'n_sites: {n_sites}')
    print(f'n_samples: {n_samples[0]}')
    print(
        f'Relative error: {topological_twist_op.error/abs(topological_twist_op.value)}')
    print('*****************')
    return topological_twist_op


def get_topo_renyi_entropy(n_sites, r=2, norms_n_samples_factor=1., n_samples=1e4, **args):
    perm_0 = np.roll(np.arange(r), 1)
    perm_a = perm_0
    perm_b = perm_0
    perm_c = perm_0
    perm_triv = np.arange(r)
    phi_abc = [perm_a, perm_b, perm_c, perm_triv]
    phi_ab = [perm_a, perm_b, perm_triv, perm_triv]
    phi_ac = [perm_a, perm_triv, perm_c, perm_triv]
    phi_bc = [perm_triv, perm_b, perm_c, perm_triv]
    phi_a = [perm_a, perm_triv, perm_triv, perm_triv]
    phi_b = [perm_triv, perm_b, perm_triv, perm_triv]
    phi_c = [perm_triv, perm_triv, perm_c, perm_triv]
    n_samples = [n_samples]+[int(n_samples*norms_n_samples_factor)]*6

    perms = [phi_abc, phi_ab, phi_ac, phi_bc, phi_a, phi_b, phi_c]
    [s_abc, s_ab, s_ac, s_bc, s_a, s_b, s_c] = evaluate_list_of_perms(
        n_sites, perms, n_samples=n_samples, **args)
    topological_purity = s_abc * (s_a*s_b*s_c)/(s_ab * s_ac * s_bc)

    print("Mean TEE: ", topological_purity)
    return s_abc, s_ab, s_ac, s_bc, s_a, s_b, s_c


def topo_twist_size_scaling(r, min_size, max_size, d_size, base_n_samples=None, alpha=None, **args):
    sizes = np.arange(min_size, max_size+1, d_size, dtype=int)
    results = []
    for s in sizes:
        # If base_n_samples is defined, the number of samples is scaled
        if base_n_samples is not None:
            if alpha is None:
                # magical value obtained from experiment
                # should ensure that the error is roughly constant across sizes
                alpha = 2.4
            n_samples = base_n_samples * \
                np.round(np.exp(alpha*r * (np.sqrt(s)-np.sqrt(min_size))), 1)
            args["n_samples"] = n_samples
        s = int(s)
        results.append(get_topo_twist(s, r, **args))
    print(sizes)
    print([r.value for r in results])
    print([r.error for r in results])
    return sizes, results


def renyi_entropy_scaling(n_spins, min_theta, max_theta, n_thetas, n_samples, r=2, n_chains=10):
    thetas = np.linspace(min_theta, max_theta, n_thetas)
    region_n_spins = []
    res = []
    perm_0 = np.roll(np.arange(r), 1)
    perm_a = perm_0
    perm_b = perm_0
    perm_c = perm_0
    perm_triv = np.arange(r)
    phi_abc = [perm_a, perm_b, perm_c, perm_triv]
    for theta in tqdm(thetas):
        # the number of replicas is obtained from the specified permutation
        model = LaughlinSphere(n_sites=n_spins, n_replicas=r)
        hilbert = get_hilbert_space(r, n_spins)
        sampler = get_sampler(r, n_spins, hilbert, n_chains=n_chains)
        operator = RegionSwapOperator(
            n_spins, r, hilbert, phi_abc, max_theta=theta)
        state = nk.vqs.MCState(sampler, model=model, variables={
            "params": {}}, n_samples=int(n_samples))
        result = state.expect(operator)
        region_n_spins_i = sum(operator.regions != 3)
        print(
            f"Evaluation result: {result}, acceptance rate: {state.sampler_state.acceptance}")
        print(f"N spins: {region_n_spins_i}, "
              f"Log result: {-np.log(result.mean.real)}")
        res.append(-np.log(result.mean.real))
        region_n_spins.append(region_n_spins_i)
    print(region_n_spins)
    print(res)
    return res, region_n_spins


if __name__ == "__main__":
    base_n_samples = 1e5
    n_min = 4
    n_max = 14
    d_n = 2
    results = []
    for n in range(n_min, n_max+1, d_n):
        print(f"n_spins: {n}")
        n_samples = np.round(
            np.exp(3*(np.sqrt(n)-np.sqrt(n_min)))*base_n_samples)
        phi = evaluate_list_of_perms(
            n, [[[1, 0, 2], [2, 1, 0], [1, 2, 0], [0, 1, 2]]], n_samples=n_samples)[0]
        print(phi)
        print(np.angle(phi.value))
        results.append(np.angle(phi.value))
