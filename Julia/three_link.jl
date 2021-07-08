using Random, Distributions
using Clustering
using LinearAlgebra
using Setfield

struct ThreeLink
    θ::Matrix{Float64}
    x::Matrix{Float64}
    ξ::Matrix{Float64}
    M::Int              # Number of components
    N::Int              # Number of data points
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

    temp = kmeans(ξ, M)
    μ = zeros(5, M)
    for i = 1:M
        μ[:,i] = temp.centers[:,i]
      end
    ThreeLink(θ, x, ξ, M, N, μ, Σ, h, temp)
end

function ThreeLink(;N::Int=10)
    d = Uniform(-π, π)
    θ = rand(d, 3, N)
    x = rand(d, 2, N)
    ξ = rand(d, 5, N)
    M = max(1, N ÷ 100)

    for i = 1:N
        fk!(x[:,i], θ[:,i])
        ξ[:,i] = vcat(x[:,i], θ[:,i])
    end

    robot = ThreeLink(θ, x, ξ, M, N)
    return robot
end

function fk!(x::Vector, θ::Vector)
    x[1] = cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(sum(θ))
    x[2] = sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(sum(θ))
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
            numerator = 1/sqrt(det(Σ[j])) * 
                exp(-1/2* dot((r.ξ[:,i] - r.μ[:,j]), Σ[j]\(r.ξ[:,i] - r.μ[:,j])) )
            denominator = 0.0
            for l = 1:r.M
                denominator += 1/sqrt(det(Σ[l])) * 
                    exp(-1/2* dot((r.ξ[:,i] - r.μ[:,l]), Σ[l]\(r.ξ[:,i] - r.μ[:,l])) )
            end
            r.h[i,j] = numerator / denominator
        end
    end
end

function _M_step(r::ThreeLink)
    for j = 1:r.M
        numerator_μ = zeros(5)
        numerator_Σ = zeros(5, 5)
        denominator = 0.0

        for i = 1:r.N
            numerator_μ += r.h[i,j] * r.ξ[:,i]
            denominator += r.h[i,j]
        end

        r.μ[:,j] = numerator_μ / denominator

        for i = 1:r.N
            numerator_Σ += r.h[i,j] * (r.ξ[:,i] - r.μ[:,j]) * (r.ξ[:,i] - r.μ[:,j])'
        end
        r.Σ[j] = numerator_Σ / denominator
    end
end

function execute_EM(r::ThreeLink; maxiter::Int=100)
    for i = 1:maxiter
        _E_step(r)
        _M_step(r)
    end
end

function prediction(r::ThreeLink, x::Vector)
    
end