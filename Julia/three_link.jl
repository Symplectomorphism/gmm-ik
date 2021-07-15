using Random, Distributions
using Clustering
using LinearAlgebra
using JuMP, Ipopt
using ForwardDiff


"""
The following book and papers were used as references:
Book: McLachlan, Krishnan, "The EM Algorithm and Extensions," ISBN: 9780470191606, 0470191600
Papers: Ghahramani, "Solving Inverse Problems Using an EM Approach To Density Estimation"
        Xu, et al., "Data-driven methods towards learning the highly nonlinear 
                     inverse kinematics of tendon-driven surgical manipulators"
                     DOI: 10.1002/rcs.1774
"""

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
    model = Model(Ipopt.Optimizer)
    ThreeLink(θ, x, ξ, M, N, π, μ, Σ, h, temp)
end

function ThreeLink(;N::Int=10)
    d = Uniform(-180*π/180, 180*π/180)
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
    d = Uniform(-180*π/180, 180*π/180)
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

function jacobian!(J::Matrix, θ::AbstractArray)
    a = [1., 1., 1/2]

    J[1,1] = -a[1]*sin(θ[1]) - a[2]*sin(θ[1]+θ[2]) - a[3]*sin(θ[1]+θ[2]+θ[3])
    J[1,2] = -a[2]*sin(θ[1]+θ[2]) - a[3]*sin(θ[1]+θ[2]+θ[3])
    J[1,3] = -a[3]*sin(θ[1]+θ[2]+θ[3])
    J[2,1] = a[1]*cos(θ[1]) + a[2]*cos(θ[1]+θ[2]) + a[3]*cos(θ[1]+θ[2]+θ[3])
    J[2,2] = a[2]*cos(θ[1]+θ[2]) + a[3]*cos(θ[1]+θ[2]+θ[3])
    J[2,3] = a[3]*cos(θ[1]+θ[2]+θ[3])
end

function jacobians(θ::AbstractArray)
    id = Matrix(I, 3, 3)
    temp = zeros(2,3);
    jacobian!(temp, θ)

    J = Array{Matrix{Float64}, 1}()
    for i = 1:2
        push!(J,
            temp * (I - sum(id[:,j]*id[:,j]' for j in (i+1):3 ) )
        )
    end
    push!(J, temp)

    # dJ = Array{Array{Matrix{Float64}, 1}, 1}()
    # dJ11 = zeros(2,3)
    # dJ11[1,1] = -a[1]*cos(θ[1]) - a[2]*cos(θ[1]+θ[2]) - a[3]*cos(θ[1]+θ[2]+θ[3])
    # dJ11[2,1] = -a[1]*sin(θ[1]) - a[2]*sin(θ[1]+θ[2]) - a[3]*sin(θ[1]+θ[2]+θ[3])

    return J
end

function energy(θ::AbstractArray{T, 1}, θdot::AbstractArray{T, 1}) where T
    m = [2, 1, 1/2]
    J = jacobians(θ)
    return 1/2*dot(θdot, sum(m[i]*J[i]'*J[i] for i = 1:3), θdot)
end

function power(θ::AbstractArray{T, 1}, θdot::AbstractArray{T, 1}, 
                θddot::AbstractArray{T, 1}, ) where T
    # This is incomplete!...
    
    m = [2, 1, 1/2]
    J = jacobians(θ)
    H = sum(m[i]*J[i]'*J[i] for i = 1:3)
    term1 = dot(θdot, H, θddot)

    return term1
end

function ik_optimization(x::Vector, y::Vararg{AbstractArray, 4};
                            start::Vector=rand(-π:0.1:π, 3)) where {N}

    SUCCESS = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    n_failures::Int=0
    model = Model(Ipopt.Optimizer)

    A_eq, b_eq, lb, ub = y
    isempty(lb) ? lb = -Inf*ones(3) : nothing
    isempty(ub) ? ub = Inf*ones(3) : nothing

    @variable(model, θ[1:3])
    @constraint(model, [i=1:length(b_eq)], dot(A_eq[i], θ) == b_eq[i])
    @constraint(model, con[i=1:length(lb)], lb[i] <= θ[i] <= ub[i])
    set_start_value.(θ, start)
    @NLobjective(model, Min, 
        (cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(θ[1]+θ[2]+θ[3]) - x[1])^2 + 
        (sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(θ[1]+θ[2]+θ[3]) - x[2])^2
    )
    JuMP.set_silent(model)
    for i = 1:10
        optimize!(model)
        if any( termination_status(model) .== SUCCESS ) 
            break
        else
            set_start_value.(all_variables(model), rand(-π:0.1:π, 3))
        end
    end
    n_failures == 10 ? (@warn "Optimizer did not convert: IK solution may be wrong.") : nothing
    return rem2pi.(value.(θ), RoundNearest)
end

ik_elbow_down_optimization(x::Vector; start::Vector=rand(-π:0.1:π, 3)) = 
    ik_optimization(x, [], [], [-Inf, 0, -Inf], []; start=start)

ik_elbow_down_optimization(x::Vector, θ3::Float64; start::Vector=rand(-π:0.1:π, 3)) = 
    ik_optimization(x, [[0, 0, 1]], [θ3], [-Inf, 0, -Inf], []; start=start)

ik_elbow_up_optimization(x::Vector; start::Vector=rand(-π:0.1:π, 3)) = 
    ik_optimization(x, [], [], [], [Inf, 0, Inf]; start=start)

ik_elbow_up_optimization(x::Vector, θ3::Float64; start::Vector=rand(-π:0.1:π, 3)) = 
    ik_optimization(x, [[0, 0, 1]], [θ3], [], [Inf, 0, Inf]; start=start)