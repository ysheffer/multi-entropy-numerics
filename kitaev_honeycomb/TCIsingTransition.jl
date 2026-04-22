# Obtain the phases J_n, Phi_r as a function of J_x

include("KitaevHoneycomb.jl")
using .KitaevHoneycomb
using ProgressBars

K = 0.3
ns = 10
max_n = 3
max_r = 5
n_Jx = 20

Jx_rng = range(0.7, 1.0, length=n_Jx)
j_n_results = zeros(Complex{Float64}, n_Jx, max_n)
phi_r_results = zeros(Complex{Float64}, n_Jx, max_r - 2)

for (i, Jx) in ProgressBar(enumerate(Jx_rng))
    for n in 1:max_n
        n_reps = 2 * n + 1
        model = KitaevHoneycomb.create_honeycomb_model([Jx, Jx, 1], K, ns, ns, n_reps)
        perm_b = vcat(collect(1:n), mod1.(collect(1:n+1) .+ 1, n + 1) .+ n)
        perm_c = vcat(mod1.(collect(1:n+1) .+ 1, n + 1), collect(1:n) .+ (n + 1))
        perm_a = mod1.(collect(1:2*n+1) .+ 1, 2 * n + 1)
        perm_lambda = collect(1:2*n+1)
        @time res = KitaevHoneycomb.evaluate_permutation_operator(model,
            [perm_a, perm_b, perm_c, perm_lambda]; single_gauge=true)
        j_n_results[i, n] = angle(res)
    end

    for (j, r) in enumerate(3:max_r)
        resarg, _ = KitaevHoneycomb.get_lens_space_multi_entropy(ns, ns, r;
            J=[Jx, Jx, 1], K=K, arg_only=true)
        phi_r_results[i, j] = resarg
    end
end

# print results in Python format
println("J_n results:")
KitaevHoneycomb.print_in_python_format(j_n_results)

println("Phi_r results:")
KitaevHoneycomb.print_in_python_format(phi_r_results)
