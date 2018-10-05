module kNN
    export knn, kernelregression

    using StatsBase
    using Distances
    using NearestNeighbors

    include("kernels.jl")
    include("bandwidth.jl")
    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
    include("regression.jl")
    include("anomaly.jl")
end
