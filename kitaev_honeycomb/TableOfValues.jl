include("KitaevHoneycomb.jl")
using .KitaevHoneycomb
using Plots

Jx = 1.
K = 0.3
ns = 16
max_n = 2
max_r = 3

# Obtain the phases of J_n (Renyi modular commutator)
j_n_results = []
for n in 1:max_n
    n_reps = 2 * n + 1
    model = KitaevHoneycomb.create_honeycomb_model([Jx, Jx, Jx], K, ns, ns, n_reps)
    perm_b = vcat(collect(1:n), mod1.(collect(2:n+2), n + 1) .+ n)
    perm_c = vcat(mod1.((1:n+1) .+ 1, n + 1), collect(n+2:2*n+1))
    perm_a = vcat(collect(2:2*n+1), [1])
    println(perm_a, " ", perm_b, " ", perm_c)
    perm_lambda = collect(1:2*n+1)
    @time res = KitaevHoneycomb.evaluate_permutation_operator(model,
        [perm_a, perm_b, perm_c, perm_lambda]; single_gauge=true)
    res = res / abs(res)
    expected_res = exp(-pi * 1im * 2 / 24 * n^2 / (2 * n + 1) / (n + 1))
    print(res, " vs ", expected_res, " for n = ", n, "\n")
    println(angle(res))
    push!(j_n_results, [res, expected_res])
end

# Obtain the phases of K_n (also some measure of the chiral central charge)
k_n_results = []
for n in 1:max_n
    n_reps = 2 * n + 1
    model = KitaevHoneycomb.create_honeycomb_model([Jx, Jx, Jx], K, ns, ns, n_reps)
    perm_b = vcat([[2 * i, 2 * i - 1] for i in 1:n]..., [2 * n + 1])
    perm_c = vcat([1], [[2 * i + 1, 2 * i] for i in 1:n]...)
    perm_a = perm_b[perm_c]
    perm_lambda = collect(1:2*n+1)
    @time res = KitaevHoneycomb.evaluate_permutation_operator(model,
        [perm_a, perm_b, perm_c, perm_lambda]; single_gauge=true)
    res = res / abs(res)
    expected_res = exp(-pi * 1im / 24 * n * (2 * n - 1) / (2 * n + 1))
    print(res, " vs ", expected_res, " for n = ", n, "\n")
    push!(k_n_results, [res, expected_res])
end


# Obtain the phases of Phi_r (Lens-space multi-entropy)
phi_r_results = []
rs = collect(2:max_r)
for r in rs
    resarg, resmag = KitaevHoneycomb.get_lens_space_multi_entropy(ns, ns, r;
        J=[Jx, Jx, Jx], K=K, arg_only=true)
    resarg = exp(1im * resarg)  # Convert to complex exponential form
    expected_res = exp(-2 * pi * 1im * (r + 2 / r) / 24 / 2) * (1 + (-1)^r + 2 * exp(2 * pi * 1im / 16 * r))
    expected_res /= abs(expected_res)
    print(resarg, " vs ", expected_res, " for r = ", r, "\n")
    push!(phi_r_results, [resarg, expected_res])
end


# Print the results for the error of J_n
print("Results for J_n:\n")
for (n, (res, expected_res)) in enumerate(j_n_results)
    argres = angle(res)
    expected_arg = angle(expected_res)
    err = abs(argres - expected_arg)
    println("n = $n: result = $(argres) Error = $err")
end

# Print the results for the error of K_n
print("Results for K_n:\n")
for (n, (res, expected_res)) in enumerate(k_n_results)
    argres = angle(res)
    expected_arg = angle(expected_res)
    err = abs(argres - expected_arg)
    println("n = $n: result = $(argres) Error = $err")
end

# Print the results for the error of Phi_r
print("Results for Phi_r:\n")
for (r, (resarg, expected_res)) in zip(rs, phi_r_results)
    argres = angle(resarg)
    expected_arg = angle(expected_res)
    err = abs(argres - expected_arg)
    println("r = $r: result = $(argres) Error = $err")
end

println("J_n error arr:")
println([abs(angle(resarg) - angle(expected_res)) for (resarg, expected_res) in j_n_results])
println("K_n error arr:")
println([abs(angle(resarg) - angle(expected_res)) for (resarg, expected_res) in k_n_results])
println("Phi_r error arr:")
println([abs(angle(resarg) - angle(expected_res)) for (resarg, expected_res) in phi_r_results[2:end]])


# Multiple usages of charged operators and permutation operators
n_s = 8
reps_rng = 2:4
mus = range(-5, 5, length=40)
results = zeros(ComplexF64, length(mus), length(reps_rng))
p1 = plot(xlabel="μ", ylabel="arg")
p2 = plot(xlabel="μ", ylabel="2π * arg / μ²")
for n_reps in reps_rng
    model = KitaevHoneycomb.create_chern_insulator_model([1.0, 1.0, 1.0], 0.3, n_s, n_s, n_reps, S=0.0)
    gs_proj = KitaevHoneycomb.get_gs_proj(model)
    for (i, mu) in enumerate(mus)
        perm_triv = [i for i in 1:n_reps]
        perm_cyclic = [i % n_reps + 1 for i in 1:n_reps]
        mu_per_region = [mu, 0, mu, 0] .* 1im
        perms_per_region = [perm_cyclic, perm_cyclic, perm_triv, perm_triv]
        res = KitaevHoneycomb.evaluate_permutation_and_charge_operators(model, perms_per_region, mu_per_region, gs_proj)
        res = angle(res)
        results[i, n_reps-1] = res
        println("mu = ", mu, ", n_reps = ", n_reps, ", res = ", res, ", normalized res = ", 2 * pi * res / (mu^2))
    end
    plot!(p1, mus, real.(results[:, n_reps-1]), label="n_reps=$(n_reps)")
    plot!(p2, mus, 2 * pi * real.(results[:, n_reps-1]) ./ (mus .^ 2), label="n_reps=$(n_reps)")
end
display(p1)
display(p2)
# print in python-like format
print("[")
for i in 1:length(mus)
    print("[")
    for j in 1:length(reps_rng)
        print(round(real(results[i, j]), digits=6))
        if j < length(reps_rng)
            print(", ")
        end
    end
    println("],")
end
println("]")

# Renyi modular commutator with charge, as a function of 
# PH symmetry breaking parameter S
begin
    n_s = 19
    max_n = 2
    Ss = range(0, 0.3, length=4)
    results = zeros(length(Ss), max_n)
    mu0 = [1, 0, 1, 0]
    mu = pi
    p1 = plot(xlabel="S", ylabel="arg", title="$(mu0)")
    # p2 = plot(xlabel="μ", ylabel="2π * arg / μ²", title="$(mu0)")
    for n in 1:max_n
        n_reps = n + 1
        for (i, S) in enumerate(Ss)
            model = KitaevHoneycomb.create_chern_insulator_model([1.0, 1.0, 1.0], 0.3, n_s, n_s, n_reps, S=S)
            gs_proj = KitaevHoneycomb.get_gs_proj(model)
            perm_triv = [i for i in 1:n_reps]
            perm_cyclic = [i % n_reps + 1 for i in 1:n_reps]
            mu_per_region = [mu, 0, mu, 0]
            perms_per_region = [perm_cyclic, perm_cyclic, perm_triv, perm_triv]
            res = KitaevHoneycomb.evaluate_permutation_and_charge_operators(model, perms_per_region, mu_per_region, gs_proj,
                rep_to_apply_charge=1)
            res = real.(angle(res))
            results[i, n] = res
            println("S = ", S, ", n_reps = ", n_reps, ", res = ", res, ", normalized res = ", 2 * pi * res / (mu^2) / n * (n + 1) * 2)
        end
        plot!(p1, Ss, (results[:, n]), label="n_reps=$(n_reps)")
        # expected_jn = results[findfirst(Ss .== 0.0), n]
        # normalized = (results[:, n] .- expected_jn) ./ (Ss .^ 2) * 2 * 2 * pi
        # plot!(p2, Ss, normalized, label="n_reps=$(n_reps)")
    end
    display(p1)
    # display(p2)
end

# Renyi TEE 
begin
    n_s = 12
    max_n = 2
    results = zeros(max_n)
    for n in 1:max_n
        n_reps = n + 1
        model = KitaevHoneycomb.create_chern_insulator_model([1.0, 1.0, 1.0], 0.3, n_s, n_s, n_reps, S=0.1)
        gs_proj = KitaevHoneycomb.get_gs_proj(model)
        perm_triv = [i for i in 1:n_reps]
        perm_cyclic = [i % n_reps + 1 for i in 1:n_reps]
        mu_per_region = [0, 0, 0, 0]
        p123 = [perm_cyclic, perm_cyclic, perm_cyclic, perm_triv]
        p12 = [perm_cyclic, perm_cyclic, perm_triv, perm_triv]
        p23 = [perm_triv, perm_cyclic, perm_cyclic, perm_triv]
        p13 = [perm_cyclic, perm_triv, perm_cyclic, perm_triv]
        p1 = [perm_cyclic, perm_triv, perm_triv, perm_triv]
        p2 = [perm_triv, perm_cyclic, perm_triv, perm_triv]
        p3 = [perm_triv, perm_triv, perm_cyclic, perm_triv]
        (res123, res12, res23, res13, res1, res2, res3) = (KitaevHoneycomb.evaluate_permutation_and_charge_operators(model, p, mu_per_region, gs_proj,
            rep_to_apply_charge=1) for p in (p123, p12, p23, p13, p1, p2, p3))
        res = res123 * res1 * res2 * res3 / (res12 * res23 * res13)
        results[n] = real.(res)
        println(" n_reps = ", n_reps, ", res = ", res)
    end
end
begin
    include("KitaevHoneycomb.jl")
    using .KitaevHoneycomb
    max_ns = 12
    min_ns = 5
    ns_arr = collect(min_ns:2:max_ns)
    mus = range(-0.05, 0.05, length=3)
    results = zeros(length(ns_arr))
    p1 = plot(xlabel="μ", ylabel="arg")
    n = 2
    for (i, n_s) in enumerate(ns_arr)
        n_reps = 2n + 1
        model = KitaevHoneycomb.create_chern_insulator_model([1.0, 1.0, 1.0], 0.2, n_s, n_s, n_reps, S=0.0)
        gs_proj = KitaevHoneycomb.get_gs_proj(model)
        perm_b = vcat(collect(1:n), mod1.(collect(2:n+2), n + 1) .+ n)
        perm_c = vcat(mod1.((1:n+1) .+ 1, n + 1), collect(n+2:2*n+1))
        perm_a = vcat(collect(2:2*n+1), [1])
        perm_lambda = collect(1:2*n+1)
        mu_per_region = [0, 0, 0, 0] .* 1im
        perms_per_region = [perm_a, perm_b, perm_c, perm_lambda]
        # res = KitaevHoneycomb.evaluate_permutation_and_charge_operators(model, perms_per_region, mu_per_region, gs_proj,
        #     rep_to_apply_charge=1)
        res = KitaevHoneycomb.evaluate_permutation_and_charge_operators(model, perms_per_region, mu_per_region, gs_proj,
            rep_to_apply_charge=1)
        res = real.(angle(res))
        results[i] = res
        # println("mu = ", ", n_reps = ", n_reps, ", res = ", res, ", normalized res = ", 2 * pi * res / (mu^2))
    end
    plot(p1, ns_arr, results, label="n_reps=$(n_reps)")
end