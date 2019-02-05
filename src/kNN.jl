module kNN
    export knn, kernelregression

    using StatsBase
    using Distances
    using NearestNeighbors
	using SmoothingKernels
    using LinearAlgebra
    using Statistics
    
    include("bandwidth.jl")
    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
    include("regression.jl")
    include("anomaly.jl")
end
