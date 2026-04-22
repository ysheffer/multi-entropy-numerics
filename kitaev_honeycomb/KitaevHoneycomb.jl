module KitaevHoneycomb
using LinearAlgebra
using IterTools
using Plots

function print_in_python_format(arr)
    println("np.array([")
    for row in eachrow(arr)
        println("    [", join(real.(row), ", "), "],")
    end
    println("])")
end

abstract type AbstractModel end

struct KitaevHoneycombModel <: AbstractModel
    J::Vector{Float64}
    c_idx_to_one_d::Array{Int}
    b_idx_to_one_d::Array{Int}
    K::Float64
    nx::Int
    ny::Int
    n_reps::Int
    regions_mat::Matrix{Int}
end

struct ChernInsulatorModel <: AbstractModel
    J::Vector{Float64}
    c_idx_to_one_d::Array{Int}
    K::Float64
    S::Float64
    nx::Int
    ny::Int
    n_reps::Int
    regions_mat::Matrix{Int}
end

function Base.size(model::KitaevHoneycombModel)
    # Returns the size of the matrix A constructed
    # for the Kitaev honeycomb model.
    return maximum(model.b_idx_to_one_d)
end

function Base.size(model::ChernInsulatorModel)
    # Returns the size of the matrix A constructed
    # for the Chern insulator model.
    return maximum(model.c_idx_to_one_d)
end

function shrange(i::Int, n::Int)
    # Shifts a range
    return (i % n) + 1
end

function make_A(model::KitaevHoneycombModel)
    """
    Constructs the matrix A for Hamiltonian of the Kitaev honeycomb model.

    The matrix A is constructed such that the Hamiltonian can be expressed as
    H = -1/2 * sum_{i,j} A_{ij} c_i c_j, where c_i are the Majorana fermions.
    Most of the matrix is for the c-fermions of the model, but we also include
    the b-fermions that are used to keep track of the gauge field.

    Args:
        model::KitaevHoneycombModel: The model for which to construct the matrix A.
    Returns:
        A::Matrix{Float64}: The constructed matrix A.
    """

    nx = model.nx
    ny = model.ny
    n_reps = model.n_reps
    J = model.J
    K = model.K
    # c_idx_to_one_d is a 5D array of shape n_reps x 4 x nx x ny x 2
    c_idx_to_one_d = model.c_idx_to_one_d

    A = zeros(size(model), size(model))

    for r in 1:n_reps, i in 1:nx, j in 1:ny
        # nn-links
        A[c_idx_to_one_d[r, i, j, 1], c_idx_to_one_d[r, i, j, 2]] = J[1]
        A[c_idx_to_one_d[r, i, shrange(j, ny), 1], c_idx_to_one_d[r, i, j, 2]] = J[2]
        A[c_idx_to_one_d[r, shrange(i, nx), j, 1], c_idx_to_one_d[r, i, j, 2]] = J[3]

        # nnn-links
        # bb
        A[c_idx_to_one_d[r, shrange(i, nx), j, 1], c_idx_to_one_d[r, i, j, 1]] = K
        A[c_idx_to_one_d[r, i, j, 1], c_idx_to_one_d[r, i, shrange(j, ny), 1]] = K
        A[c_idx_to_one_d[r, i, shrange(j, ny), 1], c_idx_to_one_d[r, shrange(i, nx), j, 1]] = K
        # ww
        A[c_idx_to_one_d[r, i, j, 2], c_idx_to_one_d[r, shrange(i, nx), j, 2]] = K
        A[c_idx_to_one_d[r, i, shrange(j, ny), 2], c_idx_to_one_d[r, i, j, 2]] = K
        A[c_idx_to_one_d[r, shrange(i, nx), j, 2], c_idx_to_one_d[r, i, shrange(j, ny), 2]] = K
    end

    # b projection
    # We add b (gauge field) fermions to the hilbert space, to keep track of 
    # changes in the gauge field as a result of the permutation defects. 
    # We add two pairs of fermions for each boundary between two regions (A, B, C, \Lambda) and 
    # each replica.
    for r in 1:n_reps, f1 in 1:4, f2 in (f1+1):4, s in 1:2
        A[model.b_idx_to_one_d[r, f1, f2, s], model.b_idx_to_one_d[r, f2, f1, s]] = 1.0
    end

    # anti-symmetrise
    A = A - transpose(A)

    return 2.0 .* A
end

function make_A(model::ChernInsulatorModel)
    """
    Constructs the matrix A for Hamiltonian of the Chern insulator model.

    The matrix A is constructed such that the Hamiltonian can be expressed as
    H = -1/2 * sum_{i,j} A_{ij} c_i c_j, where c_i are the Majorana fermions.
    In constrast with the Kitaev honeycomb model, the Chern insulator model
    does not use b-fermions, so the matrix A is only for the c-fermions.

    Args:
        model::ChernInsulatorModel: The model for which to construct the matrix A.
    Returns:
        A::Matrix{Float64}: The constructed matrix A.
    """

    nx = model.nx
    ny = model.ny
    n_reps = model.n_reps
    J = model.J
    K = model.K
    S = model.S # particle-hole breaking term
    # c_idx_to_one_d is a 5D array of shape n_reps x 2 x 4 x nx x ny x  2
    c_idx_to_one_d = model.c_idx_to_one_d

    A = zeros(size(model), size(model))

    for r in 1:n_reps, i in 1:nx, j in 1:ny
        for cop in 1:2
            sn = 1
            # nn-links
            A[c_idx_to_one_d[r, cop, i, j, 1], c_idx_to_one_d[r, cop, i, j, 2]] = J[1] * sn
            A[c_idx_to_one_d[r, cop, i, shrange(j, ny), 1], c_idx_to_one_d[r, cop, i, j, 2]] = J[2] * sn
            A[c_idx_to_one_d[r, cop, shrange(i, nx), j, 1], c_idx_to_one_d[r, cop, i, j, 2]] = J[3] * sn

            # nnn-links
            # bb
            A[c_idx_to_one_d[r, cop, shrange(i, nx), j, 1], c_idx_to_one_d[r, cop, i, j, 1]] = K * sn
            A[c_idx_to_one_d[r, cop, i, j, 1], c_idx_to_one_d[r, cop, i, shrange(j, ny), 1]] = K * sn
            A[c_idx_to_one_d[r, cop, i, shrange(j, ny), 1], c_idx_to_one_d[r, cop, shrange(i, nx), j, 1]] = K * sn
            # ww
            A[c_idx_to_one_d[r, cop, i, j, 2], c_idx_to_one_d[r, cop, shrange(i, nx), j, 2]] = K * sn
            A[c_idx_to_one_d[r, cop, i, shrange(j, ny), 2], c_idx_to_one_d[r, cop, i, j, 2]] = K * sn
            A[c_idx_to_one_d[r, cop, shrange(i, nx), j, 2], c_idx_to_one_d[r, cop, i, shrange(j, ny), 2]] = K * sn
        end

        # onsite particle-hole breaking term
        for s in 1:2
            A[c_idx_to_one_d[r, 1, i, j, s], c_idx_to_one_d[r, 2, i, j, s]] += S * (-1)^s
        end
        # charge conservation breaking term
        A[c_idx_to_one_d[r, 1, i, j, 1], c_idx_to_one_d[r, 2, i, j, 2]] += S * (-1)
    end

    # anti-symmetrise
    A = A - transpose(A)

    return 2.0 .* A
end


function get_gs_proj(model::AbstractModel)
    # Returns the projection matrix onto the ground state of the KitaevHoneycombModel.
    # That is, we project on the state that has only excitations with negative enrgy.
    h = make_A(model) * im
    eigvals, eigvecs = eigen(h)
    return eigvecs[:, eigvals.<0]
end

function evaluate_operator_for_gs(model::AbstractModel, M)::ComplexF64
    """
    Evaluates the expectation value of an operator of the form exp(1/4 log(M)_ij c_i c_j)
    for the ground state of the KitaevHoneycombModel.

    Args:
        gs_proj::Matrix{ComplexF64}: The projection matrix onto the ground state.
        M::Matrix{ComplexF64}: The operator to evaluate, given as a matrix.
    Returns:
        ComplexF64: The expectation value of the operator for the ground state.
        
    NOTE: In general, this expectation value suffers from a sign ambiguity. 
    This is not an issue for all operators that we checked. We verified this by checking this 
    when the model is equivalent to the toric code model (Jx=Jy=1, Jz<<1), where the expectation value is known,
    and tracking the result adiabatically as we change Jz.
    """

    # create projections on the ground state
    gs_proj = get_gs_proj(model)

    return evaluate_operator_for_gs(gs_proj, M)
end

function evaluate_operator_for_gs(gs_proj, M)::ComplexF64
    res = sqrt(det(gs_proj' * M * gs_proj))
    return res
end

function define_regions_mat(nx, ny)
    # returns a matrix of shape nx x ny 
    # where each element is an integer specifying whether the site is in region A,B,C or \Lambda
    alpha = sqrt(3 / 4)
    regions = zeros(nx, ny)
    function in_A(i, j)
        return (i < nx * alpha) && (j < ny * alpha / 2)
    end
    function in_B(i, j)
        return (i < nx * alpha / 2) && (j >= ny * alpha / 2) && (j < ny * alpha)
    end
    function in_C(i, j)
        return (i >= nx * alpha / 2) && (i < nx * alpha) && (j >= ny * alpha / 2) && (j < ny * alpha)
    end

    for i in 1:nx, j in 1:ny
        regions[i, j] = in_A(i, j) ? 1 : in_B(i, j) ? 2 : in_C(i, j) ? 3 : 4
    end
    return regions
end

function gauge_matrix_in_regions!(m, model::KitaevHoneycombModel, regions_to_apply::Matrix{Bool})
    # returns the matrix G such that exp(1/4 log(G)_ij c_i c_j) for m=log(G) applies c_i b^x_i b^y_i b^z_i
    # on all sites in the region
    # regions_to_apply is a matrix of shape 4 x n_reps, where each element is a boolean specifying whether to apply the gauge transformation in a given region and replica

    # set m to the identity
    for i in 1:size(m, 1)
        m[i, i] = 1
    end
    resolved_idx_to_one_d = model.c_idx_to_one_d

    # gauge transformation on c fermions
    for i in 1:model.nx, j in 1:model.ny
        for r in 1:model.n_reps
            reg = model.regions_mat[i, j]
            if regions_to_apply[reg, r] == 1
                m[resolved_idx_to_one_d[r, i, j, 1], resolved_idx_to_one_d[r, i, j, 1]] = -1
                m[resolved_idx_to_one_d[r, i, j, 2], resolved_idx_to_one_d[r, i, j, 2]] = -1
            end
        end
    end
    # gauge transformation on b fermions
    for reg in 1:4
        for r in 1:model.n_reps
            if regions_to_apply[reg, r] == 1
                for reg2 in 1:4
                    if reg2 != reg
                        m[model.b_idx_to_one_d[r, reg, reg2, 1], model.b_idx_to_one_d[r, reg, reg2, 1]] = -1
                        m[model.b_idx_to_one_d[r, reg, reg2, 2], model.b_idx_to_one_d[r, reg, reg2, 2]] = -1
                    end
                end
            end
        end
    end
end

function perm_vec_to_matrix(permutation::Vector{Int}; correct_phases::Bool=false)
    # Converts a permutation vector to a matrix that represents the majorana permutation
    # the matrix P is such that for m=log(p), exp(m_ij c_i c_j) applies the permutation

    n = length(permutation)
    P = zeros(n, n)
    sgn = (-1)^correct_phases
    # If correct_phases is true, apply "Jordan-Wigner"-like signs to ensure that the permutation matrix acts correctly on fermions
    # For the Kitaev honeycomb model, this is taken care of by the gauge transformations
    for i in 1:n
        j = permutation[i]
        P[j, i] = j > i ? sgn^(j - i) : 1
    end
    return P
end

function mu_to_charge_op(mu)
    # returns a real rotation matrix with angle mu
    return [cos(mu) -sin(mu); sin(mu) cos(mu)]
end

function permutation_per_region(model::KitaevHoneycombModel, perms_per_region::Vector{Vector{Int}})
    # Returns the matrix P such that exp(1/4 log(P)_ij c_i c_j) applies the permutation of the sites in the region
    # m = Matrix(I * 1.0, size(model), size(model))
    m = zeros(size(model), size(model))
    P = [perm_vec_to_matrix(perm) for perm in perms_per_region]
    n_reps = model.n_reps
    c_idx_to_one_d = model.c_idx_to_one_d
    b_idx_to_one_d = model.b_idx_to_one_d

    regions_mat = model.regions_mat
    # Permute c fermions
    for i in 1:model.nx, j in 1:model.ny, s in 1:2
        for r1 in 1:n_reps, r2 in 1:n_reps
            m[c_idx_to_one_d[r1, i, j, s],
                c_idx_to_one_d[r2, i, j, s]] = P[regions_mat[i, j]][r1, r2]
        end
    end
    # Permute b fermions
    for r1 in 1:n_reps, r2 in 1:n_reps, f1 in 1:4, f2 in 1:4, s in 1:2
        if f1 != f2
            m[b_idx_to_one_d[r1, f1, f2, s], b_idx_to_one_d[r2, f1, f2, s]] = P[f1][r1, r2]
        end
    end
    return m
end

function charge_operator_per_region(model::ChernInsulatorModel, mu_per_region::Vector; rep_to_apply=nothing)
    rep_to_apply = something(rep_to_apply, 1)
    # Returns the matrix P such that exp(1/4 log(P)_ij c_i c_j) applies the charge operator
    m = Matrix{ComplexF64}(I * 1.0, size(model), size(model))
    c_idx_to_one_d = model.c_idx_to_one_d
    charge_mat_per_region = [mu_to_charge_op(mu) for mu in mu_per_region]

    regions_mat = model.regions_mat
    # Apply mu to c fermions, only on the replica given by rep_to_apply
    for i in 1:model.nx, j in 1:model.ny, s in 1:2, cop1 in 1:2, cop2 in 1:2
        m[c_idx_to_one_d[rep_to_apply, cop1, i, j, s], c_idx_to_one_d[rep_to_apply, cop2, i, j, s]] = charge_mat_per_region[regions_mat[i, j]][cop1, cop2]
    end
    # Assert that the matrix is orthogonal
    # if !isapprox(m * m', I, atol=1e-6)
    #     error("The charge operator matrix is not orthogonal")
    # end
    return m
end

function permutation_per_region(model::ChernInsulatorModel, perms_per_region::Vector{Vector{Int}})
    # Returns the matrix P such that exp(1/4 log(P)_ij c_i c_j) applies the permutation of the sites in the region
    # m = Matrix(I * 1.0, size(model), size(model))
    m = zeros(size(model), size(model))
    P = [perm_vec_to_matrix(perm, correct_phases=true) for perm in perms_per_region]
    n_reps = model.n_reps
    c_idx_to_one_d = model.c_idx_to_one_d

    regions_mat = model.regions_mat
    # Permute c fermions
    for cop in 1:2, i in 1:model.nx, j in 1:model.ny, s in 1:2
        for r1 in 1:n_reps, r2 in 1:n_reps
            m[c_idx_to_one_d[r1, cop, i, j, s],
                c_idx_to_one_d[r2, cop, i, j, s]] = P[regions_mat[i, j]][r1, r2]
        end
    end
    return m
end

function _check_gauge_regions_to_apply(perms_per_region, regions_to_apply, boundary_perms)
    # Checks that the gauge transformations are applied on the correct regions
    # perms_per_region is a vector of length 4, where each element is a permutation vector
    # regions_to_apply is a matrix of shape 4 x n_reps, where each element is a boolean specifying whether to apply the gauge transformation in a given region and replica
    if !all(regions_to_apply[4, :] .== false)
        return false
    end

    # Don't sum over configurations in which two regions with the same permutation
    # are acted on by a different gauge transformation
    for reg1 in 1:4, reg2 in 1:4
        if perms_per_region[reg1] == perms_per_region[reg2] && regions_to_apply[reg1, :] != regions_to_apply[reg2, :]
            return false
        end
    end

    # check that an even number of gauge transformations are applied in
    # each permutation orbit
    n_reps = size(regions_to_apply, 2)
    for reg in 1:4
        for rep in 1:n_reps
            # gather reps in the same orbit
            orbit = [rep]
            next_rep = rep
            while true
                next_rep = perms_per_region[reg][next_rep]
                if next_rep == rep
                    break
                end
                push!(orbit, next_rep)
            end
            if sum(true .- regions_to_apply[reg, orbit]) == 0
                return false
            end
        end
    end
    n_reps = size(regions_to_apply, 2)
    for reg1 in 1:4, reg2 in reg1+1:4
        for rep in 1:n_reps
            # Gather reps in the same orbit
            orbit = [rep]
            next_rep = rep
            while true
                next_rep = boundary_perms[reg1, reg2][next_rep]
                if next_rep == rep
                    break
                end
                push!(orbit, next_rep)
            end
            if (sum(regions_to_apply[reg1, orbit] .+ regions_to_apply[reg2, orbit]) + length(orbit)) % 2 == 0
                return false
            end
        end
    end
    return true
end

function _get_boundary_perms(perms_per_region)::Dict{Tuple,Vector}
    # Returns a dictionary of boundary permutations
    # where the keys are tuples of the form (reg1, reg2) and the
    # values are the permutations that map the sites in region reg2 to the sites in region reg1,
    # i.e. perm1*perm2^{-1}
    boundary_perms = Dict{Tuple,Vector}()
    perm_inverses = [[findfirst(v -> v == i, perm) for i in 1:length(perm)] for perm in perms_per_region]
    for reg1 in 1:4, reg2 in 1:4
        if reg1 != reg2
            boundary_perms[reg1, reg2] = perms_per_region[reg1][perm_inverses[reg2]]
        end
    end
    return boundary_perms
end

function evaluate_permutation_and_charge_operators(model::ChernInsulatorModel, perms_per_region, mu_per_region, gs_proj; rep_to_apply_charge=nothing)
    perm_mat = permutation_per_region(model, perms_per_region)
    charge_op_mat = charge_operator_per_region(model, mu_per_region; rep_to_apply=rep_to_apply_charge)

    # Evaluate the permutation matrix for each gauge transformation that can be applied on a region

    # Evaluate the operator for the ground state
    result_for_gauge = evaluate_operator_for_gs(gs_proj, charge_op_mat * perm_mat)
    result = result_for_gauge
    println("Perms: ", perms_per_region, "mus", mu_per_region, ", Result: ", result)
    return result
end

function evaluate_permutation_and_charge_operators(model::ChernInsulatorModel, perms_per_region, mu_per_region)
    # Get gs_proj once
    gs_proj = get_gs_proj(model)
    return evaluate_permutation_and_charge_operators(model, perms_per_region, mu_per_region, gs_proj)
end

function evaluate_permutation_operator(model::KitaevHoneycombModel, perms_per_region; single_gauge=false)
    perm_mat = permutation_per_region(model, perms_per_region)

    # Evaluate the permutation matrix for each gauge transformation that can be applied on a region

    result = 0.0im
    # Get gs_proj once
    # This is a relatively expensive operation, as we need to diagonalize a large matrix
    # The rest of the operations only require computing a determinant
    gs_proj = get_gs_proj(model)
    zs = zeros(Bool, 1, model.n_reps)
    gauge_mat = zeros(size(model), size(model))
    boundary_perms = _get_boundary_perms(perms_per_region)
    for regions_to_apply in product(fill([false, true], model.n_reps * 3)...)
        regions_to_apply = vcat(reshape([i for i in regions_to_apply], 3, model.n_reps), zs)
        # Run some checks to see if the gauge transformation is valid (gives nonzero result)
        if !_check_gauge_regions_to_apply(perms_per_region, regions_to_apply, boundary_perms)
            continue
        end

        # Obtain that matrix that implements the gauge transformation
        # on the regions specified by regions_to_apply
        gauge_matrix_in_regions!(gauge_mat, model, regions_to_apply)
        # Evaluate the operator for the ground state, with the gauge transformation applied
        result_for_gauge = evaluate_operator_for_gs(gs_proj, perm_mat * gauge_mat)
        println("Perms: ", perms_per_region, ", Regions: ", regions_to_apply, "Result: ", result_for_gauge)
        result += result_for_gauge
        if single_gauge
            # If we are only interested in a single single value of the gauge fields, we can break here
            break
        end
    end
    println("Perms: ", perms_per_region, ", Result: ", result)
    return result
end

function evaluate_multiple_perm_ops(model::KitaevHoneycombModel, list_of_perms_per_region; single_gauge=false)
    return [evaluate_permutation_operator(model, perms_per_region; single_gauge=single_gauge) for perms_per_region in list_of_perms_per_region]
end

function get_topo_entanglement_entropy(nx, ny, n_reps; J=[1.0, 1.0, 1.0], K=0.1)
    """
    Returns the topological entanglement (Renyi) entropy for the KitaevHoneycombModel,
    as calculated using the Kitaev-Preskill prescription.

    Args:
        nx::Int: The number of unit cells in the x direction.
        ny::Int: The number of unit cells in the y direction.
        n_reps::Int: The number of replicas (the Renyi index).
        J::Vector{Float64}: A vector [Jx, Jy, Jz] representing the coupling 
            constants for the Kitaev honeycomb model.
        K::Float64: The magnetic field strength.
    Returns:
        ComplexF64: The product of the expectation values of the permutation operators,
                which correspond to exp((1-n)*S_topo).
    """

    model = create_honeycomb_model(J, K, nx, ny, n_reps)
    perm_triv = [i for i in 1:n_reps]
    perm = [i % n_reps + 1 for i in 1:n_reps]


    phi_abc = [perm, perm, perm, perm_triv]
    phi_ab = [perm, perm, perm_triv, perm_triv]
    phi_ac = [perm, perm_triv, perm, perm_triv]
    phi_bc = [perm_triv, perm, perm, perm_triv]
    phi_a = [perm, perm_triv, perm_triv, perm_triv]
    phi_b = [perm_triv, perm, perm_triv, perm_triv]
    phi_c = [perm_triv, perm_triv, perm, perm_triv]
    phi_0 = [perm_triv, perm_triv, perm_triv, perm_triv]

    pers_list = [phi_abc, phi_ab, phi_ac, phi_bc, phi_a, phi_b, phi_c, phi_0]
    s_abc, s_ab, s_ac, s_bc, s_a, s_b, s_c, s_0 = evaluate_multiple_perm_ops(model, pers_list)

    return s_abc * (s_a * s_b * s_c) / (s_ab * s_ac * s_bc * s_0)
end

function get_lens_space_multi_entropy(nx, ny, r; J=[1.0, 1.0, 1.0], K=0.1, arg_only=false)
    """
    Returns the value of the "Lens space multi-entropy" for the KitaevHoneycombModel, 
        as defined in [1]: arXiv:2502.12259
    This is a topological invariant that can be used to distinguish between different
    topological phases with the same topological entanglement entropy.

    Args:
        nx::Int: The number of unit cells in the x direction.
        ny::Int: The number of unit cells in the y direction.
        r::Int: The number of regions in the lens space.
        J::Vector{Float64}: A vector [Jx, Jy, Jz] representing the coupling 
            constants for the Kitaev honeycomb model.
        K::Float64: The magnetic field strength.
        arg_only::Bool: If true, only returns the argument of the result, otherwise returns both argument and absolute value.
    Returns:
        ComplexF64: The argument of expectation value of the permutation operators.
        ComplexF64: The absolute value of the normalized expectation value of the permutation operators,
                with the normalization defined in [1].
    """


    model = create_honeycomb_model(J, K, nx, ny, 2 * r)
    perm_triv = [i for i in 1:(2*r)]
    permA = [mod1.((1:r) .- 1, r); mod1.((1:r) .+ 1, r) .+ r]
    permB = [(1:r) .+ r; (1:r)]
    permC = [mod1.((1:r) .+ 1, r) .+ r; mod1.((1:r) .- 1, r)]
    phi11 = [permA, permB, permC, perm_triv]
    phi12 = [permC, permB, permA, perm_triv]
    phi13 = [permA, perm_triv, permA, perm_triv]
    phi14 = [permC, perm_triv, permC, perm_triv]
    phi21 = [permA, perm_triv, permC, perm_triv]
    phi22 = [permC, perm_triv, permA, perm_triv]
    phi23 = [permA, permB, permA, perm_triv]
    phi24 = [permC, permB, permC, perm_triv]
    pers_list = [phi11, phi12, phi13, phi14, phi21, phi22, phi23, phi24]
    if !arg_only
        s11, s12, s13, s14, s21, s22, s23, s24 = evaluate_multiple_perm_ops(model, pers_list)
        s_arg = angle(s11)
        s_abs = (s11 * s12 * s13 * s14) / (s21 * s22 * s23 * s24)
    else
        single_gauge = (mod(r, 2) == 1)
        s11 = evaluate_permutation_operator(model, phi11, single_gauge=single_gauge)
        s_arg = angle(s11)
        s_abs = 1.0
    end


    return s_arg, s_abs
end

function create_chern_insulator_model(J, K, nx, ny, n_reps; S=0.0)
    c_idx_to_one_d = reshape(range(1, 2 * nx * ny * n_reps * 2), n_reps, 2, nx, ny, 2)
    # Two fermions for each region (A,B,C,\Lambda) in each replica

    regions_mat = define_regions_mat(nx, ny)
    return ChernInsulatorModel(J, c_idx_to_one_d, K, S, nx, ny, n_reps, regions_mat)
end

function create_honeycomb_model(J, K, nx, ny, n_reps)
    """Creates a KitaevHoneycombModel struct with the specified parameters.
    Args:
        J::Vector{Float64}: A vector [Jx, Jy, Jz] representing the coupling 
            constants for the Kitaev honeycomb model.
        K::Float64: The magnetic field strength.
        nx::Int: The number of unit cells in the x direction.
        ny::Int: The number of unit cells in the y direction.
        n_reps::Int: The number of replicas (the Renyi index).
    Returns:
        KitaevHoneycombModel: The created Kitaev honeycomb model.
    """
    c_idx_to_one_d = reshape(range(1, nx * ny * n_reps * 2), n_reps, nx, ny, 2)
    # Two fermions for each region (A,B,C,\Lambda) in each replica
    b_idx_to_one_d = zeros(n_reps, 4, 4, 2)
    ind = length(c_idx_to_one_d) + 1
    for r in 1:n_reps, f1 in 1:4, f2 in 1:4, s in 1:2
        if f1 == f2
            continue
        end
        b_idx_to_one_d[r, f1, f2, s] = ind
        ind += 1
    end

    regions_mat = define_regions_mat(nx, ny)
    return KitaevHoneycombModel(J, c_idx_to_one_d, b_idx_to_one_d, K, nx, ny, n_reps, regions_mat)
end

end # module KitaevHoneycomb

# using .KitaevHoneycomb
# using LinearAlgebra
# using Plots

# Loop over Renyi modular commutator calculations


# J_list = collect(range(1, 1, length=1))
# # J_list = [1.0]
# resarg = zeros(ComplexF64, length(J_list))
# n_r = 7
# for (j, Jyz) in enumerate(J_list)
#     model = KitaevHoneycomb.create_honeycomb_model([Jyz, Jyz, 1.0], 0.3, 16, 16, n_r)
#     perm_b = vcat(collect(2:n+1), [1], collect(n+2:2*n+1))
#     perm_c = vcat(collect(1:n), collect(n+2:2*n+1), [n + 1])
#     perm_a = perm_b[perm_c]
#     # @time res = KitaevHoneycomb.evaluate_permutation_operator(model, [[2, 1, 3], [3, 2, 1], [2, 3, 1], [1, 2, 3]])
#     # perm1 = [2, 1, 3]
#     # perm2 = [1, 3, 2]
#     # perm1 = [2, 3, 4, 5, 1, 6, 7, 8, 9]
#     # perm2 = [1, 2, 3, 4, 6, 7, 8, 9, 5]
#     perm1 = [2, 3, 4, 1, 5, 6, 7]
#     perm2 = [5, 2, 3, 4, 6, 7, 1]
#     perm3 = [2, 7, 3, 4, 5, 6, 1]
#     # perm1 = [2, 3, 1, 4, 5]
#     # perm2 = [4, 2, 3, 5, 1]
#     # perm3 = [2, 5, 3, 4, 1]
#     # perm3 = [4, 2, 1, 3, 5]
#     perm_triv = collect(1:n_r)
#     # perm3 = perm1[perm2]
#     @time res = KitaevHoneycomb.evaluate_permutation_operator(model, [perm1, perm2, perm3, perm_triv])
#     resarg[j] = -angle(res)
#     println("Jz = ", Jyz, ", arg = ", angle(res))
# end

# # rs = collect(2:4)
# # J_list = collect(range(0.2, 1.0, length=30))
# # resarg = zeros(ComplexF64, length(rs), length(J_list))
# # resmag = zeros(length(rs), length(J_list))
# # for (i, r) in enumerate(rs)
# #     for (j, Jyz) in enumerate(J_list)
# #         @time res = KitaevHoneycomb.get_topo_twist_operator(12, 12, r, K=0.1, J=[1.0, Jyz, Jyz])
# #         resarg[i, j] = res[1]
# #         resmag[i, j] = abs(res[2])
# #         println("Jz = ", Jyz, ", arg = ", round(res[1], digits=3), ", mag = ", round(res[2], digits=3))
# #     end
# # end

# # using DelimitedFiles

# # # Export results to a CSV file
# # writedlm("results_arg.csv", real(resarg), ',')
# # writedlm("results_mag.csv", resmag, ',')
# # writedlm("rs.csv", rs, ',')
# # writedlm("jlist.csv", J_list, ',')
# # @time res = KitaevHoneycomb.get_topo_twist_operator(8, 8, 4, K=0.3, J=[1.0, 1.0, 1.0])
# # @time res = KitaevHoneycomb.get_topo_twist_operator(6, 6, 3, K=0.3, J=[1.0, 0.1, 0.1])
# # plot(J_list, abs.(resmag), xlabel="Jz", ylabel="Twist Operator", title="Twist Operator vs Jz")
# # plot(J_list, resarg, xlabel="Jz", ylabel="Twist Operator", title="Twist Operator vs Jz")

# # @time println(KitaevHoneycomb.get_topo_twist_operator(8, 8, 2, K=0.3, J=[1.0, 1.0, 1.0]))
# # @time println(KitaevHoneycomb.get_topo_twist_operator(8, 8, 2, K=0.3, J=[0.1, 1.0, 1.0]))
# # @time println(KitaevHoneycomb.get_topo_entanglement_entropy(6, 6, 3, K=1.0, J=[1.0, 1.0, 1.0]))
# # @time println(KitaevHoneycomb.get_topo_entanglement_entropy(7, 7, 4, K=0.5))

# # result = []
# # jz_res = []
# # perms_per_region = [[2, 3, 4, 1, 8, 5, 6, 7], [5, 6, 7, 8, 1, 2, 3, 4], [8, 5, 6, 7, 2, 3, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8]]
# # model = KitaevHoneycomb.create_honeycomb_model([1.0, 1.0, 1.0], 0.3, 20, 20, 8)
# # perm_mat = KitaevHoneycomb.permutation_per_region(model, perms_per_region)
# # gs_proj = KitaevHoneycomb.get_gs_proj(model)
# # zs = zeros(Bool, 1, model.n_reps)
# # gauge_mat = zeros(size(model), size(model))
# # regions_to_apply = collect(Bool.([0 1 0 0 1 0 0 0; 1 1 1 1 0 0 0 0; 1 0 1 1 1 0 0 0; 0 0 0 0 0 0 0 0]))
# # KitaevHoneycomb.gauge_matrix_in_regions!(gauge_mat, model, regions_to_apply)
# # result_for_gauge1 = KitaevHoneycomb.evaluate_operator_for_gs(gs_proj, perm_mat * gauge_mat, n_iters=1)
# # gauge_mat = zeros(size(model), size(model))
# # regions_to_apply = collect(Bool.([1 1 1 0 1 0 1 1; 1 1 0 1 0 0 1 0; 1 1 1 1 0 0 0 0; 0 0 0 0 0 0 0 0]))
# # KitaevHoneycomb.gauge_matrix_in_regions!(gauge_mat, model, regions_to_apply)
# # result_for_gauge2 = KitaevHoneycomb.evaluate_operator_for_gs(gs_proj, perm_mat * gauge_mat, n_iters=1)
# # res = result_for_gauge1 + result_for_gauge2
# # println(res / abs(res))
# # for eps in -0.3:0.02:0.3
# #     jz = 0.5 + 0.0001 + eps
# #     push!(jz_res, jz)
# #     model = KitaevHoneycomb.create_honeycomb_model([jz, jz, 1.0], 0.01, 8, 8, 6)
# #     perm_mat = KitaevHoneycomb.permutation_per_region(model, perms_per_region)
# #     gs_proj = KitaevHoneycomb.get_gs_proj(model)
# #     zs = zeros(Bool, 1, model.n_reps)
# #     gauge_mat = zeros(size(model), size(model))
# #     # regions_to_apply = collect(Bool.([0 1 1 0 0 0; 1 1 1 0 0 0; 1 0 0 0 0 0; 0 0 0 0 0 0]))
# #     regions_to_apply = collect(Bool.([0 0 0 0 0 0; 0 1 1 1 0 0; 0 1 1 1 0 0; 0 0 0 0 0 0]))
# #     KitaevHoneycomb.gauge_matrix_in_regions!(gauge_mat, model, regions_to_apply)
# #     result_for_gauge = KitaevHoneycomb.evaluate_operator_for_gs(gs_proj, perm_mat * gauge_mat, n_iters=1)
# #     push!(result, result_for_gauge)
# # end
# # plot(jz_res, angle.(result))
# # println("Perms: ", perms_per_region, ", Result: ", result)