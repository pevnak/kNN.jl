uniform(u::Real) = 1//2 * Float64(abs(u) <= 1.0)
uniform_unnormalized(u::Real) = 1//2 * Float64(abs(u) <= 1.0)

triangular(u::Real) = (1 - abs(u)) * Float64(abs(u) <= 1.0)
triangular_unnormalized(u::Real) = (1 - abs(u)) * Float64(abs(u) <= 1.0)

epanechnikov(u::Real) = 3//4 * (1 - u^2) * Float64(abs(u) <= 1.0)
epanechnikov_unnormalized(u::Real) = (1 - u^2) * Float64(abs(u) <= 1.0)

biweight(u::Real) = 15//16 * (1 - u^2)^2 * Float64(abs(u) <= 1.0)
biweight_unnormalized(u::Real) = (1 - u^2)^2 * Float64(abs(u) <= 1.0)

triweight(u::Real) = 35//32 * (1 - u^2)^3 * Float64(abs(u) <= 1.0)
triweight_unnormalized(u::Real) = (1 - u^2)^3 * Float64(abs(u) <= 1.0)

tricube(u::Real) = 70//81 * (1 - abs(u)^3)^3 * Float64(abs(u) <= 1.0)
tricube_unnormalized(u::Real) = (1 - abs(u)^3)^3 * Float64(abs(u) <= 1.0)

gaussian(u::Real) = (1 / sqrt(2 * pi)) *  exp(-1//2 * u^2)
gaussian_unnormalized(u::Real) = exp(-1//2 * u^2)

cosine(u::Real) = (pi / 4) * cos((pi / 2) * u) * Float64(abs(u) <= 1.0)
cosine_unnormalized(u::Real) = cos((pi / 2) * u) * Float64(abs(u) <= 1.0)

logistic(u::Real) = 1 / (exp(u) + 2 + exp(-u))
logistic_unnormalized(u::Real) = 1 / (exp(u) + 2 + exp(-u))

kernels = Dict(
            :uniform => uniform,
            :triangular => triangular,
            :epanechnikov => epanechnikov,
            :biweight => biweight,
            :triweight => triweight,
            :tricube => tricube,
            :gaussian => gaussian,
            :cosine => cosine,
            :logistic => logistic
          )

unnormalized_kernels = Dict(
                        :uniform => uniform_unnormalized,
                        :triangular => triangular_unnormalized,
                        :epanechnikov => epanechnikov_unnormalized,
                        :biweight => biweight_unnormalized,
                        :triweight => triweight_unnormalized,
                        :tricube => tricube_unnormalized,
                        :gaussian => gaussian_unnormalized,
                        :cosine => cosine_unnormalized,
                        :logistic => logistic_unnormalized
                      )
