using Random, Distributions
using Clustering
using LinearAlgebra
using Setfield

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

function ThreeLink(;N::Int=100, M::Int=10)
    d = Uniform(-π, π)
    θ = rand(d, 3, N)
    x = rand(d, 2, N)
    ξ = rand(d, 5, N)
    # M = max(1, N ÷ 100)

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
                exp(-1/2* dot((r.ξ[:,i] - r.μ[:,j]), r.Σ[j]\(r.ξ[:,i] - r.μ[:,j])) )
            denominator = 0.0
            for l = 1:r.M
                denominator += 1/sqrt(det(r.Σ[l])) * 
                    exp(-1/2* dot((r.ξ[:,i] - r.μ[:,l]), r.Σ[l]\(r.ξ[:,i] - r.μ[:,l])) )
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

        r.π[j] = denominator / r.N

        r.μ[:,j] = numerator_μ / denominator

        for i = 1:r.N
            numerator_Σ += r.h[i,j] * (r.ξ[:,i] - r.μ[:,j]) * (r.ξ[:,i] - r.μ[:,j])'
        end
        r.Σ[j] = numerator_Σ / denominator
    end
end

function execute_EM(r::ThreeLink; maxiter::Int=10)
    for i = 1:maxiter
        μ = zeros(5,r.M)
        Σ = Array{Matrix{Float64}, 1}()
        for i = 1:r.M
            μ[:,i] = r.μ[:,i]
            push!(Σ, r.Σ[i])
        end
        _E_step(r)
        _M_step(r)
        @info sum(norm(μ[:,i] - r.μ[:,i]) for i = 1:r.M)
        @info sum(norm(Σ[i] - r.Σ[i]) for i = 1:r.M)
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
        θ_tilde += β[i] * μ_θ_tilde[:,i]
        Σ_θθ_tilde_final += β[i]*β[i]*Σ_θθ_tilde[i]
    end

    return θ_tilde, Σ_θθ_tilde_final
end