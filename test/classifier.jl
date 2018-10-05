module TestClassifier
	using Test
	using kNN
	using DataFrames
    using RDatasets
    using Distances
    using StatsBase
	using Statistics

    iris = dataset("datasets", "iris")
    X = collect(convert(Array, iris[1:4])')
    y = convert(Array, iris[5])
    model = knn(X, y, metric = Euclidean())

    predict_k1 = predict(model, X, 1)
    predict_k2 = predict(model, X, 2)
    predict_k3 = predict(model, X, 3)
    predict_k4 = predict(model, X, 4)
    predict_k5 = predict(model, X, 5)

    mean(predict_k1 .== y)
    mean(predict_k2 .== y)
    mean(predict_k3 .== y)
    mean(predict_k4 .== y)
    mean(predict_k5 .== y)
end
