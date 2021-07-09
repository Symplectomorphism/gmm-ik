using Random, Distributions
using Clustering
using LinearAlgebra

struct ThreeLink
    θ::Matrix{Float64}
    x::Matrix{Float64}
    ξ::Matrix{Float64}                                  # (x, θ) || (input, output)
    M::Int              # Number of components
    N::Int              # Number of data points
    π::Array{Float64, 1}                                 # Mixing proportions
    μ::Matrix{Float64}                                   # Mean of Gaussians
    Σ::Array{Matrix{Float64}, 1}                         # Variance of Gaussians
    h::Matrix{Float64}
    kμ::KmeansResult{Matrix{Float64}, Float64, Int64}    # K-means results
end

function ThreeLink(θ, x, ξ, M, N)
    Σ = Array{Matrix{Float64}, 1}()
    h = zeros(N, M)

    for j = 1:M
        push!(Σ, diagm(ones(5)))
    end

    π = 1/M*ones(M)
    temp = kmeans(ξ, M)
    μ = zeros(5, M)
    for i = 1:M
        μ[:,i] = temp.centers[:,i]
    end
    ThreeLink(θ, x, ξ, M, N, π, μ, Σ, h, temp)
end

function ThreeLink(;N::Int=10)
    d = Uniform(-90*π/180, 90*π/180)
    θ = rand(d, 3, N)
    x = zeros(2, N)
    ξ = zeros(5, N)
    M = max(1, N ÷ 100)

    for i = 1:N
        fk!(x[:,i], θ[:,i])
        ξ[:,i] = vcat(x[:,i], θ[:,i])
    end

    robot = ThreeLink(θ, x, ξ, M, N)
    return robot
end

function ThreeLink(;N::Int=100, M::Int=10)
    d = Uniform(-90*π/180, 90*π/180)
    θ = rand(d, 3, N)
    x = zeros(2, N)
    ξ = zeros(5, N)
    # M = max(1, N ÷ 100)

    for i = 1:N
        # fk!(x[:,i], θ[:,i])
        x[:,i] = fk(θ[:,i])
        ξ[:,i] = vcat(x[:,i], θ[:,i])
    end

    robot = ThreeLink(θ, x, ξ, M, N)
    return robot
end

function fk!(x::Vector, θ::Vector)
    x[1] = cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(sum(θ))
    x[2] = sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(sum(θ))
end

function fk(θ::Vector)
    x = zeros(2)
    x[1] = cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(sum(θ))
    x[2] = sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(sum(θ))
    return x
end

function forward_kinematics!(r::ThreeLink, N::Int)
    for i = 1:N
        fk!(r.x[:,i], r.θ[:,i])
        r.ξ[:,i] = vcat(r.x[:,i], r.θ[:,i])
    end
end


function _E_step(r::ThreeLink)
    for i = 1:r.N
        for j = 1:r.M
            numerator = 1/sqrt(det(r.Σ[j])) * 
                exp(-1/2 * dot((r.ξ[:,i] - r.μ[:,j]), r.Σ[j]\(r.ξ[:,i] - r.μ[:,j])) )
            denominator = 0.0
            for l = 1:r.M
                denominator += 1/sqrt(det(r.Σ[l])) * 
                    exp(-1/2 * dot((r.ξ[:,i] - r.μ[:,l]), r.Σ[l]\(r.ξ[:,i] - r.μ[:,l])) )
            end
            r.h[i,j] = numerator / denominator
        end
    end
end

function _M_step(r::ThreeLink)
    for j = 1:r.M
        numerator_μ = zeros(5)
        numerator_Σ = zeros(5, 5)

        normalizer = 0.0

        for i = 1:r.N
            numerator_μ += r.h[i,j] * r.ξ[:,i]
            normalizer += r.h[i,j]
        end
        r.μ[:,j] = numerator_μ / normalizer
        
        for i = 1:r.N
            numerator_Σ += r.h[i,j] * (r.ξ[:,i] - r.μ[:,j]) * (r.ξ[:,i] - r.μ[:,j])'
        end

        r.π[j] = normalizer / r.N
        r.Σ[j] = numerator_Σ / normalizer
        r.Σ[j] = 1/2 * (r.Σ[j] + r.Σ[j]')           # Make sure it is Hermitian.
    end
end

function execute_em!(r::ThreeLink; maxiter::Int=10)
    μ_error = Inf
    Σ_error = Inf
    k = 1
    while μ_error > 1e-3 || Σ_error > 1e-2
        μ = zeros(5,r.M)
        Σ = Array{Matrix{Float64}, 1}()
        for i = 1:r.M
            μ[:,i] = r.μ[:,i]
            push!(Σ, r.Σ[i])
        end

        # EM step
        _E_step(r)
        _M_step(r)

        μ_error = sum(norm(μ[:,i] - r.μ[:,i]) for i = 1:r.M) / sum(norm(r.μ[:,i]) for i = 1:r.M)
        Σ_error = sum(norm(Σ[i] - r.Σ[i]) for i = 1:r.M) / sum(norm(r.Σ[i]) for i = 1:r.M)
        println("Iteration: $k, |Δμ| = $(round(μ_error, digits=4)), |ΔΣ| = $(round(Σ_error, digits=4))")

        k += 1
        k > maxiter ? break : nothing
    end
end

function prediction(r::ThreeLink, x::Vector)
    μ_θ_tilde = zeros(3, r.M)
    Σ_θθ_tilde = Array{Matrix{Float64}, 1}()
    d = Array{MvNormal, 1}()
    β = zeros(r.M)
    denominator = 0.0
    θ_tilde = zeros(3)
    Σ_θθ_tilde_final = zeros(3,3)

    for i = 1:r.M
        μ_θ_tilde[:,i] = r.μ[3:5,i] + r.Σ[i][3:5,1:2]*(r.Σ[i][1:2,1:2]\(x - r.μ[1:2,i]))
        push!(Σ_θθ_tilde, 
            r.Σ[i][3:5,3:5] - r.Σ[i][3:5,1:2]*(r.Σ[i][1:2,1:2]\r.Σ[i][1:2,3:5]))

        Σ = r.Σ[i][1:2,1:2]
        push!(d, MvNormal(r.μ[1:2,i], 1/2*(Σ + Σ')))
        denominator += r.π[i] * pdf(d[i], x)
    end

    for i = 1:r.M
        β[i] = r.π[i] * pdf(d[i], x) / denominator
    end


    for i = 1:r.M
        θ_tilde += β[i] * μ_θ_tilde[:,i]
        Σ_θθ_tilde_final += β[i]*β[i]*Σ_θθ_tilde[i]
    end
    return θ_tilde, Σ_θθ_tilde_final

    # # SLSE
    # value, ind = findmax([pdf(d[i], x) for i =1:r.M])
    # return μ_θ_tilde[:,ind], Σ_θθ_tilde[ind]
end


function use_gmm!(r::ThreeLink; nIter::Int=100)
    # This function uses the Julia package GaussianMixtures
    # It executes much faster than my code!

    chol = Array{UpperTriangular{Float64, Matrix{Float64}}, 1}()
    his = History[]
    push!(his, History(0.0, "aha"))

    for i = 1:r.M
        # push!(chol, cholesky(1/2*(r.Σ[i]+r.Σ[i]')).U)
        push!(chol, cholesky(r.Σ[i]).U)
    end

    gmm = GMM(r.π, convert(Matrix, r.μ'), chol, his, 0)
    em!(gmm, convert(Matrix, r.ξ'), nIter=nIter)

    r.μ[:] = gmm.μ[:]
    for i = 1:r.M
        r.Σ[i] = gmm.Σ[i]'*gmm.Σ[i]
    end
end