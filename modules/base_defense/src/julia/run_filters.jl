using Printf
using NPZ
include("util.jl")
include("kmeans_filters.jl")
include("quantum_filters.jl")

input_folder = ARGS[1]
output_path = ARGS[2]
defense_type = ARGS[3]
target_label = ARGS[4]
eps = parse(Int16,ARGS[5])

reps = npzread(input_folder * "$(target_label).npy")'
n = size(reps)[2]
removed = round(Int, 1.5*eps)

if defense_type == "pca"
    println("Running PCA filter")
    reps_pca, U = pca(reps, 1)
    pca_poison_ind = k_lowest_ind(-abs.(mean(reps_pca[1, :]) .- reps_pca[1, :]), round(Int, 1.5*eps))
    poison_removed = sum(pca_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @printf("%d poisons removed, %d cleans removed\n", poison_removed, clean_removed)
    npzwrite(output_path, pca_poison_ind)
elseif defense_type == "kmeans"
    println("Running kmeans filter")
    kmeans_poison_ind = .! kmeans_filter2(reps, eps)
    poison_removed = sum(kmeans_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @printf("%d poisons removed, %d cleans removed\n", poison_removed, clean_removed)
    npzwrite(output_path, kmeans_poison_ind)
elseif defense_type == "quantum"
    println("Running quantum filter")
    quantum_poison_ind = .! rcov_auto_quantum_filter(reps, eps)
    poison_removed = sum(quantum_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @printf("%d poisons removed, %d cleans removed\n", poison_removed, clean_removed)
    npzwrite(output_path, quantum_poison_ind)
else
    throw("unimplemented")
end
