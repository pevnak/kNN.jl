struct kNNRegression
    t
    y::Vector
    kernel::Function
    h::Float64
end

function kNNRegression(X::Matrix, y::Vector; kernel::Symbol = :epanechnikov, bandwidth::Real = 1.0, metric::Metric = Euclidean())
    return kNNRegression(KDTree(X, metric), y,SmoothingKernels.kernels[kernel], bandwidth)
end


function StatsBase.predict(m::kNNRegression, x::Matrix, kernel, bandwidth, k::Integer = 1)
    inds, dists = NearestNeighbors.knn(m.t, x, k)
    map(zip(inds,dists)) do z   
        i, d = z
        w = bandwidth .* kernel.(bandwidth .* d)
        sum(w .* m.y[i])/sum(w)
    end
end


StatsBase.predict(m::kNNRegression, x::Matrix, k::Integer = 1) = predict(m, x, m.kernel, m.h, k)