using MeshCat, CoordinateTransformations, Rotations
using GeometryBasics
using Colors: RGBA, RGB
using Blink
# Blink.AtomShell.install()

using Random, Distributions
using BenchmarkTools
using Clustering
using LaTeXStrings
using LinearAlgebra
using GaussianMixtures
using PyCall, PyPlot
using JuMP, Ipopt
using BSON: @save, @load     # enable if you want to load one of the .bson files
using CSV, DataFrames


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


mutable struct TLVisualizer
    vis::Visualizer
    win::Window
end

function TLVisualizer()
    vis = Visualizer()
    win = Blink.Window()
    open(vis, win)

    TLVisualizer(vis, win)
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
    d = [MvNormal(r.μ[:,j], r.Σ[j]) for j = 1:r.M]
    for i = 1:r.N
        for j = 1:r.M
            num = pdf(d[j], r.ξ[:,i])
            den = (num + sum( pdf(d[l], r.ξ[:,i]) for l = Base.Iterators.filter(λ -> λ!=j, 1:r.M) ))
            r.h[i,j] = num / den
        end
    end
end

function _M_step(r::ThreeLink; ε::Float64=1e-6)
    for j = 1:r.M
        sum_h_over_data = sum(r.h[:,j])
        r.π[j] = sum_h_over_data / r.N
        r.μ[:,j] = sum(transpose(r.h[:,j] .* r.ξ'); dims=2) / sum_h_over_data
        r.Σ[j] = Hermitian(
            sum( r.h[i,j] * (r.ξ[:,i] - r.μ[:,j]) * (r.ξ[:,i] - r.μ[:,j])' for i = 1:r.N ) / sum_h_over_data
        ) + ε*I   # Make sure it is Hermitian and positive definite
    end
end

function execute_em!(r::ThreeLink; 
    maxiter::Int=10, tol_μ::Float64=1e-3, tol_Σ::Float64=1e-2, verbose::Bool=true)

    μ_error = Inf
    Σ_error = Inf
    k = 1
    counter = 1
    while μ_error > tol_μ || Σ_error > tol_Σ
        μ = zeros(5,r.M)
        Σ = Array{Matrix{Float64}, 1}()
        for i = 1:r.M
            μ[:,i] = r.μ[:,i]
            push!(Σ, r.Σ[i])
        end

        # EM step
        _E_step(r)
        _M_step(r; ε=min(1e-6, tol_Σ))

        μ_error = sum(norm(μ[:,i] - r.μ[:,i]) for i = 1:r.M) / sum(norm(r.μ[:,i]) for i = 1:r.M)
        Σ_error = sum(norm(Σ[i] - r.Σ[i]) for i = 1:r.M) / sum(norm(r.Σ[i]) for i = 1:r.M)
        
        if verbose && (counter % 10 == 0)
            println("Iteration: $k, |Δμ| = $(round(μ_error, digits=6)), |ΔΣ| = $(round(Σ_error, digits=6))")
            counter = 0
        end

        counter += 1
        k += 1
        k > maxiter ? break : nothing
    end
    println("Iteration: $k, |Δμ| = $(round(μ_error, digits=6)), |ΔΣ| = $(round(Σ_error, digits=6))")
end

function conditionalize(r::ThreeLink, x::Vector)
    π_θ_tilde = zeros(r.M)              # π_θ_tilde = β (in the paper Xu, et al.)
    μ_θ_tilde = zeros(3, r.M)           # this will be the conditional mean of each Gaussian
    Σ_θθ_tilde = Array{Matrix{Float64}, 1}()    # this will be the conditional covariance of each Gaussian
    d = Array{MvNormal, 1}()

    denominator = 0.0
    for i = 1:r.M
        μ_θ_tilde[:,i] = r.μ[3:5,i] + r.Σ[i][3:5,1:2]*(r.Σ[i][1:2,1:2]\(x - r.μ[1:2,i]))

        Σ_temp = r.Σ[i][3:5,3:5] - r.Σ[i][3:5,1:2]*(r.Σ[i][1:2,1:2]\r.Σ[i][1:2,3:5])
        push!(Σ_θθ_tilde, Hermitian(Σ_temp))

        Σ = r.Σ[i][1:2,1:2]
        push!(d, MvNormal(r.μ[1:2,i], Matrix(Hermitian(Σ))))
        denominator += r.π[i] * pdf(d[i], x)
    end

    for i = 1:r.M
        π_θ_tilde[i] = r.π[i] * pdf(d[i], x) / denominator
    end

    return π_θ_tilde, μ_θ_tilde, Σ_θθ_tilde
end

function predict_most_likely(r::ThreeLink, x::Vector)
    π_θ_tilde, μ_θ_tilde, Σ_θθ_tilde = conditionalize(r, x)

    # Single component least-squares estimation
    # From "Ghahramani, Solving Inverse Problems Using an EM Approach To Density Estimation"
    # value, ind = findmax([pdf(d[i], x) for i =1:r.M]),                        where d[i] = N(x | μ_{x,i}, Σ_{xx,i}) comes from conditonalize(r, x)
    value, ind = findmax(π_θ_tilde)   # this is equivalent to the line above.
    return μ_θ_tilde[:,ind], Σ_θθ_tilde[ind]
end

function predict_full_posterior(r::ThreeLink, x::Vector)
    π_θ_tilde, μ_θ_tilde, Σ_θθ_tilde = conditionalize(r, x)

    # This one is from the paper: Xu, et al. "Data-driven ..." DOI: 10.1002/rcs.1774
    μ_θ_tilde_final = zeros(3)                  
    Σ_θθ_tilde_final = zeros(3,3)
    for i = 1:r.M
        μ_θ_tilde_final += π_θ_tilde[i] * μ_θ_tilde[:,i]    
        Σ_θθ_tilde_final += π_θ_tilde[i]*π_θ_tilde[i]*Σ_θθ_tilde[i]
    end
    return μ_θ_tilde_final, Matrix(Hermitian(Σ_θθ_tilde_final))
end

function predict(r::ThreeLink, x::Vector; mode::Symbol=:slse)
    θ_hat = zeros(3)
    Σ_hat = zeros(3,3)
    if mode == :slse             # Do maximum-likelihood estimation over the most likely Gaussian that generated x: (SLSE)
        μ_θ_tilde, Σ_θθ_tilde = predict_most_likely(r,x)
        θ_hat = μ_θ_tilde
        Σ_hat = Σ_θθ_tilde
    else
        μ_θ_tilde_final, Σ_θθ_tilde_final = predict_full_posterior(r, x)
        Σ_hat = Σ_θθ_tilde_final
        if mode == :lse              # Do maximum-likelihood over the full posterior (mixture of Gaussians): (LSE)
            θ_hat = μ_θ_tilde_final
        elseif mode == :stoch        # Stochastically sample from the full posterior (mixture of Gaussians): (STOCH)
            d = MvNormal(μ_θ_tilde_final, Σ_θθ_tilde_final)
            θ_hat = rand(d)
        else
            error("Unknown method for prediction.")
        end
    end

    return θ_hat, Σ_hat
end

function predict_cond(r::ThreeLink, x::Vector, θ3::Float64; mode::Symbol=:slse)
    μ, Σ = predict(r, x; mode=mode)
    d = MvNormal(μ, Σ)
    θ = rand(d)
    counter = 1
    while θ[3] >= θ3 + 0.01 || θ[3] <= θ3 - 0.01
        θ = rand(d)
        if counter > 1000 
            @warn "Satisfactory approximate solution not found." 
            break
        end
        counter += 1
    end
    return θ
end

function predict_elbow_down(r::ThreeLink, x::Vector; mode::Symbol=:slse)
    μ, Σ = predict(r, x; mode=mode)
    d = MvNormal(μ, Σ)
    θ = rand(d)
    counter = 1
    while θ[2] <= 0
        θ = rand(d)
        if counter > 1000 
            @warn "Satisfactory approximate solution not found." 
            break
        end
        counter += 1
    end
    return θ
end

function predict_elbow_up(r::ThreeLink, x::Vector; mode::Symbol=:slse)
    μ, Σ = predict(r, x; mode=mode)
    d = MvNormal(μ, Σ)
    θ = rand(d)
    counter = 1
    while θ[2] >= 0
        θ = rand(d)
        if counter > 1000 
            @warn "Satisfactory approximate solution not found." 
            break
        end
        counter += 1
    end
    return θ
end

function predict_elbow_down_cond(r::ThreeLink, x::Vector, θ3::Float64; mode::Symbol=:slse)
    μ, Σ = predict(r, x; mode=mode)
    d = MvNormal(μ, Σ)
    θ = rand(d)
    counter = 1
    while θ[2] <= 0 || θ[3] >= θ3 + 0.01 || θ[3] <= θ3 - 0.01
        θ = rand(d)
        if counter > 1000 
            @warn "Satisfactory approximate solution not found." 
            break
        end
        counter += 1
    end
    return θ
end

function predict_elbow_up_cond(r::ThreeLink, x::Vector, θ3::Float64; mode::Symbol=:slse)
    μ, Σ = predict(r, x; mode=mode)
    d = MvNormal(μ, Σ)
    θ = rand(d)
    counter = 1
    while θ[2] >= 0 || θ[3] >= θ3 + 0.01 || θ[3] <= θ3 - 0.01
        θ = rand(d)
        if counter > 1000 
            @warn "Satisfactory approximate solution not found." 
            break
        end
        counter += 1
    end
    return θ
end

function rand_predict(r::ThreeLink, x::Vector)
    μ, Σ = predict(r, x)
    d = MvNormal(μ, Σ)
    return rand(d)
end


function use_gmm!(r::ThreeLink; nIter::Int=100)
    # This function uses the Julia package GaussianMixtures
    # It executes much faster than my code!
    # It doesn't seem to work as well, though...

    chol = Array{UpperTriangular{Float64, Matrix{Float64}}, 1}()
    his = History[]
    push!(his, History(0.0, "aha"))

    for i = 1:r.M
        # push!(chol, cholesky(Matrix(Hermitian(r.Σ[i]))).U)
        push!(chol, cholesky(r.Σ[i]).U)
    end

    gmm = GMM(r.π, convert(Matrix, r.μ'), chol, his, 0)
    em!(gmm, convert(Matrix, r.ξ'), nIter=nIter)

    r.μ[:] = gmm.μ[:]
    for i = 1:r.M
        r.Σ[i] = gmm.Σ[i]'*gmm.Σ[i]
    end
end


function test_training(r::ThreeLink; nPoints::Int=200, mode::Symbol=:slse)
    xmin, xmax = (minimum(r.ξ[1,:]), maximum(r.ξ[1,:]))
    ymin, ymax = (minimum(r.ξ[2,:]), maximum(r.ξ[2,:]))
    dx = Uniform(xmin, xmax)
    dy = Uniform(ymin, ymax)

    test_x = convert(Matrix, hcat(rand(dx, nPoints), rand(dy, nPoints))')
    pred_θ = hcat([predict(r, test_x[:,i]; mode)[1] for i = 1:nPoints]...)
    cost = [norm(fk(pred_θ[:,i]) - test_x[:,i]) for i = 1:nPoints]

    return mean(cost)
end


function solve_optimization(x::Vector, y::Vararg{AbstractArray, 4};
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

solve_elbow_down_optimization(x::Vector; start::Vector=rand(-π:0.1:π, 3)) = 
    solve_optimization(x, [], [], [-Inf, 0, -Inf], []; start=start)

solve_elbow_down_optimization(x::Vector, θ3::Float64; start::Vector=rand(-π:0.1:π, 3)) = 
    solve_optimization(x, [[0, 0, 1]], [θ3], [-Inf, 0, -Inf], []; start=start)

solve_elbow_up_optimization(x::Vector; start::Vector=rand(-π:0.1:π, 3)) = 
    solve_optimization(x, [], [], [], [Inf, 0, Inf]; start=start)

solve_elbow_up_optimization(x::Vector, θ3::Float64; start::Vector=rand(-π:0.1:π, 3)) = 
    solve_optimization(x, [[0, 0, 1]], [θ3], [], [Inf, 0, Inf]; start=start)


function hypertrain_M(;N::Int=1001, M_span::AbstractArray=2:10:102, record::Bool=false)
    avg_cost = Float64[]
    for M in M_span
        r = ThreeLink(N=N, M=M)
        execute_em!(r; maxiter=100, tol_μ=1e-4, tol_Σ=1e-3, verbose=true)
        push!(avg_cost, test_training(r; nPoints=200))
        @info "Average Cost(M=$M) = $(avg_cost[end])"
        println()
    end
    fig = figure(100)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    # ax.plot(M_span, avg_cost, linestyle="-", marker="o")
    ax.bar(M_span, avg_cost, width=diff(M_span)[1]*0.8)
    ax.set_ylabel(L"Average $\ell_2$ error", fontsize=15)
    ax.set_xlabel(LaTeXString("M: component size"), fontsize=15)
    ax.set_title(LaTeXString("Training data size: N = $N"), fontsize=16)
    ax.set_xticks(M_span)

    if record
        fig.savefig("../TeX/figures/hyperparam_M.svg", dpi=600, 
            bbox_inches="tight", format="svg")
    end

    return avg_cost, fig
end

function hypertrain_N(;M::Int=61, N_span::AbstractArray=1001:500:5001, record::Bool=false)
    avg_cost = Float64[]
    for N in N_span
        r = ThreeLink(N=N, M=M)
        execute_em!(r; maxiter=100, tol_μ=1e-4, tol_Σ=1e-3, verbose=true)
        push!(avg_cost, test_training(r; nPoints=200))
        @info "Average Cost(N=$N) = $(avg_cost[end])"
        println()
    end
    fig = figure(101)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    # ax.plot(M_span, avg_cost, linestyle="-", marker="o")
    ax.bar(N_span, avg_cost, width=diff(N_span)[1]*0.8)
    ax.set_ylabel(L"Average $\ell_2$ error", fontsize=15)
    ax.set_xlabel(LaTeXString("N: training data size"), fontsize=15)
    ax.set_title(LaTeXString("Component size: M = $M"), fontsize=16)
    ax.set_xticks(N_span)

    if record
        fig.savefig("../TeX/figures/hyperparam_N.svg", dpi=600, 
            bbox_inches="tight", format="svg")
    end

    return avg_cost, fig
end


function hypertrain_MN(;M_span::AbstractArray=Int.(round.(2 .^range(log(2, 10); stop=log(2, 320), length=6))), 
                        N_span::AbstractArray=Int.(round.(2 .^range(log(2, 100); stop=log(2, 3200), length=6))), 
                        record::Bool=false)

    Z = zeros(length(N_span), length(M_span))
    i,j = (1,1)
    for M in M_span
        for N in N_span
            if M <= N
                r = ThreeLink(N=N, M=M)
                execute_em!(r; maxiter=150, tol_μ=1e-4, tol_Σ=1e-3, verbose=true)
                Z[i,j] = test_training(r; nPoints=1000)
            else
                Z[i,j] = maximum(Z)
                continue
            end
            @info "Average Cost(M=$M, N=$N) = $(Z[i,j])"
            println()
            i += 1
        end
        j += 1
        i = 1
    end
    Z[Z.==0.0] .= maximum(Z)


    fig = figure(101)
    fig.clf()
    ax = fig.add_subplot(1,1,1)

    # levels = sort( range(maximum(z); stop=0.05, length=10) )
    # cs = ax.contour(θ1, θ2, z, levels=levels, cmap=PyPlot.cm.coolwarm)
    # ax.clabel(cs, cs.levels, inline=true, fontsize=8)
    cs = ax.contourf(M_span, N_span, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(LaTeXString("M: component size"), fontsize=30)
    ax.set_ylabel(LaTeXString("N: training data size"), fontsize=30)
    ax.set_title(L"Average $\ell_2$ error", fontsize=30)
    # ax.set_aspect("equal")
    cbar = fig.colorbar(cs, ticks=0.1:0.15:1.3)
    cbar.ax.tick_params(labelsize=30)

    ax.set_xticks(M_span)
    ax.set_yticks(N_span)
    ax.tick_params(labelsize=30)
    fig.tight_layout()


    if record
        fig.savefig("../TeX/figures/hyperparam_MN.svg", dpi=600, 
            bbox_inches="tight", format="svg")
    end

    return Z, fig
end


function generate_cartesian_distribution(r::ThreeLink; nPoints::Int=100)
    xmin, xmax = (minimum(r.ξ[1,:]), maximum(r.ξ[1,:]))
    ymin, ymax = (minimum(r.ξ[2,:]), maximum(r.ξ[2,:]))
    dx = Uniform(xmin, xmax)
    dy = Uniform(ymin, ymax)
    x = [rand(dx), rand(dy)]
    @info "x = $(round.(x; digits=3))"

    μ, Σ = predict(r, x; mode=:slse)
    d = MvNormal(μ, Σ)
    θ_dist = rand(d, nPoints)
    # θ_dist = hcat([predict_cond(r, x, 0.0; mode=:slse) for i = 1:nPoints]...)
    x_dist = hcat([fk(θ_dist[:,i]) for i = 1:nPoints]...)
    
    fig = figure(1);
    fig.clf()
    plot_manipulator!(fig, x, θ_dist, x_dist)
    return nothing
end


function generate_cartesian_distribution(r::ThreeLink, x::Vector; nPoints::Int=100, record::Bool=false)
    μ, Σ = predict(r, x; mode=:slse)
    d = MvNormal(μ, Σ)
    θ_dist = rand(d, nPoints)
    # θ_dist = hcat([predict_cond(r, x, 0.0; mode=:slse) for i = 1:nPoints]...)
    x_dist = hcat([fk(θ_dist[:,i]) for i = 1:nPoints]...)
    
    fig = figure(1);
    fig.clf()
    plot_manipulator!(fig, x, θ_dist, x_dist)

    if record
        fig.savefig("../TeX/figures/sample_solution.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
    return nothing
end

function plot_manipulator!(fig::Figure, θ_dist::Matrix{Float64}, x_dist::Matrix{Float64})
    
    nPoints = size(θ_dist)[2]
    ax = fig.add_subplot(1,1,1)

    line1 = Array{PyCall.PyObject, 1}()
    line2 = Array{PyCall.PyObject, 1}()
    line3 = Array{PyCall.PyObject, 1}()
    for i = 1:nPoints
        p1 = [cos(θ_dist[1,i]), sin(θ_dist[1,i])]
        p2 = p1 + [cos(θ_dist[1,i]+θ_dist[2,i]), sin(θ_dist[1,i]+θ_dist[2,i])]
        p3 = p2 + 1/2*[cos(sum(θ_dist[:,i])), sin(sum(θ_dist[:,i]))]

        l2 = ax.plot(0, 0, marker="^", markersize=7, color="green", alpha=0.7)
        l1 = ax.plot([0,p1[1]], [0, p1[2]], linewidth=2, color="orange", alpha=0.5)
        ax.plot(p1[1], p1[2], marker="^", markersize=7, color="green", alpha=0.2)
        ax.plot([p1[1], p2[1]], [p1[2],p2[2]], linewidth=2, color="orange", alpha=0.2)
        ax.plot(p2[1], p2[2], marker="^", markersize=7, color="green", alpha=0.2)
        ax.plot([p2[1], p3[1]], [p2[2],p3[2]], linewidth=2, color="orange", alpha=0.2)

        l3 = plot(x_dist[1,i], x_dist[2,i], marker="*", markersize=10, color="black", 
                alpha=0.75)
        
        push!(line1, l1[1])
        push!(line2, l2[1])
        push!(line3, l3[1])
    end
    line1[1].set_label("Robot links")
    line2[1].set_label("Robot joints")
    line3[1].set_label("GMM solution")
    
    ax.set_xlabel(L"x", fontsize=16)
    ax.set_ylabel(L"y", fontsize=16)
    ax.legend()
end

function plot_manipulator!(fig::Figure, x::Vector, θ_dist::Matrix{Float64}, x_dist::Matrix{Float64})

    nPoints = size(θ_dist)[2]
    ax = fig.add_subplot(1,1,1)

    line1 = Array{PyCall.PyObject, 1}()
    line2 = Array{PyCall.PyObject, 1}()
    line3 = Array{PyCall.PyObject, 1}()
    for i = 1:nPoints
        p1 = [cos(θ_dist[1,i]), sin(θ_dist[1,i])]
        p2 = p1 + [cos(θ_dist[1,i]+θ_dist[2,i]), sin(θ_dist[1,i]+θ_dist[2,i])]
        p3 = p2 + 1/2*[cos(sum(θ_dist[:,i])), sin(sum(θ_dist[:,i]))]

        l2 = ax.plot(0, 0, marker="^", markersize=7, color="green", alpha=0.7)
        l1 = ax.plot([0,p1[1]], [0, p1[2]], linewidth=2, color="orange", alpha=0.5)
        ax.plot(p1[1], p1[2], marker="^", markersize=7, color="green", alpha=0.2)
        ax.plot([p1[1], p2[1]], [p1[2],p2[2]], linewidth=2, color="orange", alpha=0.2)
        ax.plot(p2[1], p2[2], marker="^", markersize=7, color="green", alpha=0.2)
        ax.plot([p2[1], p3[1]], [p2[2],p3[2]], linewidth=2, color="orange", alpha=0.2)

        l3 = plot(x_dist[1,i], x_dist[2,i], marker="*", markersize=10, color="black", 
                alpha=0.75)
        
        push!(line1, l1[1])
        push!(line2, l2[1])
        push!(line3, l3[1])
    end
    line1[1].set_label("Robot links")
    line2[1].set_label("Robot joints")
    line3[1].set_label("GMM solution")
    plot(x[1], x[2], marker="o", markersize=16, label="End-effector location")
    
    ax.set_xlabel(L"x", fontsize=16)
    ax.set_ylabel(L"y", fontsize=16)
    ax.legend()
end


function plot_posterior(r::ThreeLink, x::Vector=[-1.5, -0.4]; record::Bool=false)
    # Plot most likely posterior N*(θ | x) by marginalizing θ3
    μ, Σ = predict(r, x)

    μ12 = μ[1:2]
    Σ12 = Σ[1:2,1:2]
    d = MvNormal(μ12, Σ12)


    fig = figure(2)
    fig.clf()
    fig.suptitle(L"$\mathcal{N}(\theta_1, \theta_2 \mid x = {%$(round.(x; digits=2))})$ (marginalized over $\theta_3$)", fontsize=16)
    
    ax = fig.add_subplot(1,2,1)
    ax.cla()

    res = svd(Σ12)
    μ_view_min = μ12 - 5*res.S[1] * res.U[:,1]
    μ_view_max = μ12 + 5*res.S[1] * res.U[:,1]

    θ1 = range(μ_view_min[1]; stop=μ_view_max[1], length=201)
    θ2 = range(μ_view_min[2]; stop=μ_view_max[2], length=99*2)
    z = zeros(length(θ2), length(θ1))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            z[i,j] = pdf(d, [θ1[j], θ2[i]])
        end
    end
    # levels = sort( range(maximum(z); stop=0.05, length=10) )
    # cs = ax.contour(θ1, θ2, z, levels=levels, cmap=PyPlot.cm.coolwarm)
    # ax.clabel(cs, cs.levels, inline=true, fontsize=8)
    cs = ax.contourf(θ1, θ2, z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.set_aspect("equal")


    PyPlot.PyObject(PyPlot.axes3D)      # PyPlot.pyimport("mpl_toolkits.mplot3d.axes3d")
    ax = fig.add_subplot(1,2,2, projection="3d")
    ax.cla()

    X = θ1' .* ones(length(θ2))
    Y = ones(length(θ1))' .* θ2
    Z = zeros(size(X))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            Z[i,j] = pdf(d, [X[i,j], Y[i,j]])
        end
    end
    ax.plot_surface(X, Y, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.view_init(elev=38, azim=-15)

    if record
        fig.savefig("../TeX/figures/posterior_marginal_theta3.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
end

function plot_posterior(r::ThreeLink, θ3::Float64, x::Vector=[-1.5, -0.4]; record::Bool=false)
    # Plot most likely posterior N*(θ | x) by conditioning θ3
    μ, Σ = predict(r, x)


    μ_cond_θ3 = μ[1:2] + Σ[1:2,3] * 1/Σ[3,3] * (θ3 - μ[3])
    temp = Σ[1:2,1:2] - Σ[3,1:2]*1/Σ[3,3]*Σ[1:2,3]'
    Σ_cond_θ3 = Matrix(Hermitian(temp))
    d = MvNormal(μ_cond_θ3, Σ_cond_θ3)


    fig = figure(2)
    fig.clf()
    fig.suptitle(L"$\mathcal{N}(\theta_1, \theta_2 \mid x = {%$(round.(x; digits=2))})$ (conditioned at $\theta_3 = {%$(round(θ3; digits=2))}$)", fontsize=16)
    
    ax = fig.add_subplot(1,2,1)
    ax.cla()

    res = svd(Σ_cond_θ3)
    μ_view_min = μ_cond_θ3 - 5*res.S[1] * res.U[:,1]
    μ_view_max = μ_cond_θ3 + 5*res.S[1] * res.U[:,1]

    θ1 = range(μ_view_min[1]; stop=μ_view_max[1], length=201)
    θ2 = range(μ_view_min[2]; stop=μ_view_max[2], length=99*2)
    z = zeros(length(θ2), length(θ1))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            z[i,j] = pdf(d, [θ1[j], θ2[i]])
        end
    end
    # levels = sort( range(maximum(z); stop=0.05, length=10) )
    # cs = ax.contour(θ1, θ2, z, levels=levels, cmap=PyPlot.cm.coolwarm)
    # ax.clabel(cs, cs.levels, inline=true, fontsize=8)
    cs = ax.contourf(θ1, θ2, z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.set_aspect("equal")


    PyPlot.PyObject(PyPlot.axes3D)      # PyPlot.pyimport("mpl_toolkits.mplot3d.axes3d")
    ax = fig.add_subplot(1,2,2, projection="3d")
    ax.cla()

    X = θ1' .* ones(length(θ2))
    Y = ones(length(θ1))' .* θ2
    Z = zeros(size(X))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            Z[i,j] = pdf(d, [X[i,j], Y[i,j]])
        end
    end
    ax.plot_surface(X, Y, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.view_init(elev=38, azim=-15)

    if record
        fig.savefig("../TeX/figures/posterior_cond_theta3.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
end


function plot_full_posterior(r::ThreeLink; x::Vector=[-1.5, -0.4], record::Bool=false)
    # Plot full posterior P(θ | x) by marginalizing θ3
    π_θ_tilde, μ_θ_tilde, Σ_θθ_tilde = conditionalize(r, x)

    d = Array{MvNormal, 1}()

    for j = 1:r.M
        push!(d, MvNormal(μ_θ_tilde[1:2,j], Σ_θθ_tilde[j][1:2,1:2]))
    end

    fig = figure(5)
    fig.clf()
    fig.suptitle(L"$P(\theta_1, \theta_2 \mid x = {%$(round.(x; digits=2))})$ (marginalized over $\theta_3$)", fontsize=16)

    ax = fig.add_subplot(1,2,1)
    ax.cla()

    # θ1 = range(-3.4; stop=-1.5, length=201)
    # θ2 = range(-2.5; stop=1.33, length=99*2)
    θ1 = range(-π; stop=π, length=201)
    θ2 = range(-π; stop=π, length=99*2)
    Z = zeros(length(θ2), length(θ1))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            Z[i,j] = sum( π_θ_tilde[k]*pdf(d[k], [θ1[j], θ2[i]]) for k = 1:r.M )
        end
    end
    # levels = sort( range(maximum(Z); stop=0.05, length=10) )
    # cs = ax.contour(θ1, θ2, Z, levels=levels, cmap=PyPlot.cm.coolwarm)
    # ax.clabel(cs, cs.levels, inline=true, fontsize=8)
    cs = ax.contourf(θ1, θ2, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.set_aspect("equal")


    PyPlot.PyObject(PyPlot.axes3D)      #equivalently: PyPlot.pyimport("mpl_toolkits.mplot3d.axes3d")
    ax = fig.add_subplot(1,2,2, projection="3d")
    ax.cla()

    X = θ1' .* ones(length(θ2))
    Y = ones(length(θ1))' .* θ2
    Z = zeros(size(X))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            Z[i,j] = sum( π_θ_tilde[k]*pdf(d[k], [θ1[j], θ2[i]]) for k = 1:r.M )
        end
    end
    ax.plot_surface(X, Y, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.view_init(elev=34, azim=-74)

    if record
        fig.savefig("../TeX/figures/full_posterior_marginal.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
end


function plot_full_posterior(r::ThreeLink, θ3::Float64; x::Vector=[-1.5, -0.4], record::Bool=false)
    # Plot the full posterior P(θ | x) by conditioning θ3
    π_θ_tilde, μ_θ_tilde, Σ_θθ_tilde = conditionalize(r, x)

    d = Array{MvNormal, 1}()

    for j = 1:r.M
        μ_cond_θ3 = μ_θ_tilde[1:2,j] + Σ_θθ_tilde[j][1:2,3] * 1/Σ_θθ_tilde[j][3,3] * (θ3 - μ_θ_tilde[3,j])
        temp = Σ_θθ_tilde[j][1:2,1:2] - Σ_θθ_tilde[j][3,1:2]*1/Σ_θθ_tilde[j][3,3]*Σ_θθ_tilde[j][1:2,3]'
        Σ_cond_θ3 = Matrix(Hermitian(temp))

        push!(d, MvNormal(μ_cond_θ3, Σ_cond_θ3))
    end

    fig = figure(5)
    fig.clf()
    fig.suptitle(L"$P(\theta_1, \theta_2 \mid x = {%$(round.(x; digits=2))})$ (conditioned at $\theta_3 = {%$(round(θ3; digits=2))}$)", fontsize=16)

    ax = fig.add_subplot(1,2,1)
    ax.cla()

    θ1 = range(-π; stop=π, length=201)
    θ2 = range(-π; stop=π, length=99*2)
    Z = zeros(length(θ2), length(θ1))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            Z[i,j] = sum( π_θ_tilde[k]*pdf(d[k], [θ1[j], θ2[i]]) for k = 1:r.M )
        end
    end
    # levels = sort( range(maximum(Z); stop=0.05, length=10) )
    # cs = ax.contour(θ1, θ2, Z, levels=levels, cmap=PyPlot.cm.coolwarm)
    # ax.clabel(cs, cs.levels, inline=true, fontsize=8)
    cs = ax.contourf(θ1, θ2, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.set_aspect("equal")


    PyPlot.PyObject(PyPlot.axes3D)      #equivalently: PyPlot.pyimport("mpl_toolkits.mplot3d.axes3d")
    ax = fig.add_subplot(1,2,2, projection="3d")
    ax.cla()

    X = θ1' .* ones(length(θ2))
    Y = ones(length(θ1))' .* θ2
    Z = zeros(size(X))
    for i = 1:length(θ2)
        for j = 1:length(θ1)
            Z[i,j] = sum( π_θ_tilde[k]*pdf(d[k], [θ1[j], θ2[i]]) for k = 1:r.M )
        end
    end
    ax.plot_surface(X, Y, Z, cmap=PyPlot.cm.coolwarm)
    ax.set_xlabel(L"θ_1", fontsize=16)
    ax.set_ylabel(L"θ_2", fontsize=16)
    ax.view_init(elev=34, azim=-74)

    if record
        fig.savefig("../TeX/figures/full_posterior_marginal_cond.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
end


function plot_posteriors_sequentially_3D(x::Vector; maxiter::Int=9)
    r = ThreeLink(N=2001, M=101)
    fig = figure(3)
    fig.clf()
    PyPlot.PyObject(PyPlot.axes3D) 

    θ1 = range(-3; stop=3, length=101)
    θ2 = range(-3; stop=3, length=99)

    X = θ1' .* ones(length(θ2))
    Y = ones(length(θ1))' .* θ2
    Z = zeros(size(X))

    for n = 1:maxiter+1
        μ, Σ = predict(r, x)
        μ12 = μ[1:2]
        Σ12 = Σ[1:2,1:2]
        d = MvNormal(μ12, Σ12)

        for i = 1:length(θ2)
            for j = 1:length(θ1)
                Z[i,j] = pdf(d, [X[i,j], Y[i,j]])
            end
        end

        ax = fig.add_subplot(2,maxiter÷2+1,n, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=PyPlot.cm.coolwarm)
        ax.view_init(elev=30, azim=60)
        ax.set_xlabel(L"θ_1", fontsize=16)
        ax.set_ylabel(L"θ_2", fontsize=16)
        ax.tick_params(labelsize=14)

        execute_em!(r, maxiter=1)
    end

    # fig.suptitle("Evolution of the posterior distribution", fontsize=16)
    fig.savefig("../TeX/figures/belief_evolution.eps", dpi=600, 
        bbox_inches="tight", format="eps")
end



function plot_posteriors_sequentially(x::Vector; niter::Int=9, record::Bool=false)
    # N*(θ \mid x) marginalized over θ3

    r = ThreeLink(N=1001, M=61)
    fig = figure(3)
    fig.clf()
    fig.suptitle("Evolution of the P(θ | x) as EM iterates", fontsize=16)

    for n = 1:niter+1
        μ, Σ = predict(r, x)
        μ12 = μ[1:2]
        Σ12 = Σ[1:2,1:2]
        d = MvNormal(μ12, Σ12)

        res = svd(Σ12)
        μ_view_min = μ12 - 5*sum(res.S[i] * res.U[:,i] for i=1:2)
        μ_view_max = μ12 + 5*sum(res.S[i] * res.U[:,i] for i=1:2)
        
        θ1 = range(μ_view_min[1]; stop=μ_view_max[1], length=101)
        θ2 = range(μ_view_min[2]; stop=μ_view_max[2], length=99)
        Z = zeros(length(θ2), length(θ1))

        for i = 1:length(θ2)
            for j = 1:length(θ1)
                Z[i,j] = pdf(d, [θ1[j], θ2[i]])
            end
        end
        ax = fig.add_subplot(2,niter÷2+1,n)
        
        cs = ax.contourf(θ1, θ2, Z, cmap=PyPlot.cm.coolwarm)
        ax.set_xlabel(L"θ_1", fontsize=16)
        ax.set_ylabel(L"θ_2", fontsize=16)
        ax.set_xticks(round.(range(μ_view_min[1]; stop=μ_view_max[1], length=4); digits=2))
        ax.set_yticks(round.(range(μ_view_min[2]; stop=μ_view_max[2], length=4); digits=2))
        ax.tick_params(labelsize="large")
        ax.set_aspect("auto")

        execute_em!(r, maxiter=3)
        ax.set_title(LaTeXString("EM Iteration: $(3*(n-1))"), fontsize=14)
    end
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if record   
        fig.savefig("../TeX/figures/posterior_evolution.eps", dpi=600, 
            bbox_inches="tight", format="eps")
    end
end

function set_scene()
    tl = TLVisualizer()

    a = [1,1,1/2]
    t = 0.05

    links = Array{Rect3D, 1}()
    joints = Array{Sphere, 1}()
    body_vis = Array{Visualizer, 1}()
    groups = Array{Visualizer, 1}()

    for i = 1:3
        push!(links, HyperRectangle(Vec(0., 0, 0), Vec(a[i], t, t)))
        push!(joints, Sphere{Float64}(Vec(0,0,0), 2*t))
        push!(groups, tl.vis["group"  * string(i)])

        if i == 1
            linkmaterial = MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5))
        elseif i == 2
            linkmaterial = MeshPhongMaterial(color=RGBA(0, 1, 0, 0.5))
        else
            linkmaterial = MeshPhongMaterial(color=RGBA(0, 0, 1, 0.75))
        end
        push!(body_vis, 
            setobject!(groups[i]["link"  * string(i)], links[i], linkmaterial)
        )
        push!(body_vis, 
            setobject!(groups[i]["joint" * string(i)], joints[i], linkmaterial)
        )
        settransform!(groups[i]["link" * string(i)], Translation(0, -t/2, -t/2))
        settransform!(groups[i]["joint" * string(i)], Translation(a[i],0,0))
    end

    return tl, groups
end


function move_joints_cs(r::ThreeLink; θ0::Vector=zeros(3), θf::Vector=[π/2, -π/3, π/4])
    a = [1,1,1/2]

    # OPTIONAL: Set camera pose
    # settransform!(vis["/Cameras/default"], 
    #     Translation(0, 0, 1) ∘ LinearMap(RotX(deg2rad(-71.5))) ∘ LinearMap(RotZ(-π/2)))
    tl, groups = set_scene()

    # Initialize the position vectors and rotation matrices
    R = Array{RotZ, 1}()            # Variable
    q = Array{Vec3, 1}()            # Location of the joints
    push!(q, Vec3(0., 0., 0))
    p = Array{Vec3, 1}()            # Constant
    for i = 1:3
        push!(R, RotZ(θ0[i]))
        push!(p, Vec3(a[i], 0, 0) )
        if i > 1
            push!(q, q[i-1] + R[i-1]*p[i-1])
        end
        # settransform!(groups[i], Translation(q[i]) ∘ LinearMap(R[i]))
    end

    # Set initial pose in animation
    anim = Animation()
    atframe(anim, 0) do
        for i = 1:3
            settransform!(groups[i], Translation(q[i]) ∘ LinearMap(R[i]))
        end
    end

    # Set animation steps
    nSteps = 120
    for i = 1:nSteps
        θ = i/nSteps*θf

        atframe(anim, i) do
            for k = 1:3
                R[k] = RotZ(sum(θ[1:k]))
                if k > 1
                    q[k] = q[k-1] + R[k-1]*p[k-1]
                end
                settransform!(groups[k], Translation(q[k]) ∘ LinearMap(R[k]))
            end
        end
    end
    setanimation!(tl.vis, anim)

    # delete!(tl.vis)
    # close(tl.win)
    return tl
end


function move_ee_cs(r::ThreeLink; x0::Vector=[-1.5, -0.4], xf::Vector=[1.5, 0.4])
    a = [1,1,1/2]
    tl, groups = set_scene()

    # OPTIONAL: Set camera pose
    # settransform!(tl.vis["/Cameras/default"], 
    #     LinearMap(RotX(deg2rad(-70))) ∘ 
    #     LinearMap(RotZ(-π/2)) ∘ 
    #     Translation(0, 0, 1*0) 
    # )
        

    # # Set obstacles
    # obstacles = Array{Rect3D, 1}()
    # push!(obstacles, HyperRectangle(Vec(0., 0, 0), Vec(1/4, 1/4, 1/4)))
    # push!(tl.vis, 
    #     setobject!(groups[i]["obstacle"  * string(1)], obstacles[i])
    # )
    # settransform!(groups[i]["obstacle" * string(i)], Translation(0, -t/2, -t/2))
    
    # Show goal
    s = 1/4
    goal = HyperRectangle(Vec(0., 0, 0), Vec(s, s, s))
    setobject!(tl.vis["goal"], goal, 
        MeshPhongMaterial(wireframe=true, wireframeLinewidth=2.0, color=RGBA(1, 1, 1, 0.5)))
    settransform!(tl.vis["goal"], Translation(xf[1]-s/2, xf[2]-s/2, 0.0-s/2))


    # Solve inverse kinematics
    # θ0 = solve_elbow_up_optimization(x0; start=predict_elbow_up(r, x0))
    θ0 = solve_elbow_down_optimization(x0, deg2rad(-90); start=[2.6, 1.96, -π/2])
    @info [fk(θ0); θ0]
    # θf = solve_elbow_up_optimization(xf; start=predict_elbow_up(r, xf))
    θf = solve_elbow_up_optimization(xf, -0.42; start=[1.42, -1.69, -0.42])
    # θf = solve_optimization(xf; start=θ0)
    @info [fk(θf); θf]

    # Initialize the position vectors and rotation matrices
    R = Array{RotZ, 1}()            # Variable
    q = Array{Vec3, 1}()            # Location of the joints
    push!(q, Vec3(0., 0., 0))
    p = Array{Vec3, 1}()            # Constant
    for i = 1:3
        push!(R, RotZ(θ0[i]))
        push!(p, Vec3(a[i], 0, 0) )
        if i > 1
            push!(q, q[i-1] + R[i-1]*p[i-1])
        end
        # settransform!(groups[i], Translation(q[i]) ∘ LinearMap(R[i]))
    end

    # Set initial pose in animation
    anim = Animation()
    atframe(anim, 0) do
        for i = 1:3
            settransform!(groups[i], Translation(q[i]) ∘ LinearMap(R[i]))
        end
    end

    # Set animation steps
    nSteps = 120
    for i = 1:nSteps
        θ = (1 - i/nSteps)*θ0 + i/nSteps*θf

        atframe(anim, i) do
            for k = 1:3
                R[k] = RotZ(sum(θ[1:k]))
                if k > 1
                    q[k] = q[k-1] + R[k-1]*p[k-1]
                end
                settransform!(groups[k], Translation(q[k]) ∘ LinearMap(R[k]))
            end
        end
    end
    setanimation!(tl.vis, anim)

    # delete!(tl.vis)
    # close(tl.win)
    return tl
end



"""
Data generation for assignment
"""


function jacobian!(J::Matrix, θ::AbstractArray)
    a = [1., 1., 1/2]

    J[1,1] = -a[1]*sin(θ[1]) - a[2]*sin(θ[1]+θ[2]) - a[3]*sin(θ[1]+θ[2]+θ[3])
    J[1,2] = -a[2]*sin(θ[1]+θ[2]) - a[3]*sin(θ[1]+θ[2]+θ[3])
    J[1,3] = -a[3]*sin(θ[1]+θ[2]+θ[3])
    J[2,1] = a[1]*cos(θ[1]) + a[2]*cos(θ[1]+θ[2]) + a[3]*cos(θ[1]+θ[2]+θ[3])
    J[2,2] = a[2]*cos(θ[1]+θ[2]) + a[3]*cos(θ[1]+θ[2]+θ[3])
    J[2,3] = a[3]*cos(θ[1]+θ[2]+θ[3])
end


function construct_solution(;N::Int=11,seed::Int=0)
    r = ThreeLink(N=N,M=1)
    rng = MersenneTwister(seed)

    d = Uniform(-1*π/180, 1*π/180)
    θdot = rand(d, 3, N)
    xdot = zeros(2, N)
    J = zeros(2,3)
    for i = 1:N
        jacobian!(J, r.θ[:,i])
        xdot[:,i] = J*θdot[:,i]
    end


    solution = DataFrame(
        :x=>r.ξ[1,:],
        :y=>r.ξ[2,:],
        :th1=>r.ξ[3,:],
        :th2=>r.ξ[4,:],
        :th3=>r.ξ[5,:],
        :xdot=>xdot[1,:],
        :ydot=>xdot[2,:],
        :th1dot=>θdot[1,:],
        :th2dot=>θdot[2,:],
        :th3dot=>θdot[3,:]
    )
    new_indices = randperm(rng, size(solution)[1])
    for column in eachcol(solution)
        column = column[new_indices]
    end

    return solution
end

function construct_forward_question(;N::Int=11, seed::Int=0)
    question = construct_solution(N=N, seed=seed)
    solution = deepcopy(question)
    select!(question, Not([:x, :y, :xdot, :ydot]))
    return question, solution
end

function construct_inverse_question(;N::Int=11, seed::Int=0)
    question = construct_solution(N=N, seed=seed)
    solution = deepcopy(question)
    select!(question, Not([:th1, :th2, :th3, :th1dot, :th2dot, :th3dot]))
    return question, solution
end

function generate_data(;nGroups=10)
    str = "./data/group" 
    
    for i = 1:nGroups
        if !isdir(str * string(i)) 
            Base.Filesystem.mkdir(str * string(i))
        else
            Base.Filesystem.rm(str * string(i); recursive=true, force=true)
            Base.Filesystem.mkdir(str * string(i))
        end

        question, solution = construct_forward_question(N=11, seed=i)
        CSV.write(str * string(i) * "/fk_question.csv", question)
        CSV.write(str * string(i) * "/fk_solution.csv", solution)

        question, solution = construct_inverse_question(N=11, seed=i)
        CSV.write(str * string(i) * "/ik_question.csv", question)
        CSV.write(str * string(i) * "/ik_solution.csv", solution)
    end
    for n = nGroups+1:nGroups+50
        if isdir(str * string(n))
            Base.Filesystem.rm(str * string(n); recursive=true, force=true)
        end
    end
end


function sim_student()
    str = "./submissions/group1"
    data = CSV.File(str * "/ik_question.csv") |> DataFrame
    N = size(data)[1]
    th = zeros(3,size(data)[1])
    thdot = zeros(3,size(data)[1])
    J = zeros(2,3)

    success::Int = 0
    i = 1
    for row in eachrow(data)
        θ, model = solve_optimization(vcat(row[:x], row[:y]))
        jacobian!(J, θ)
        θdot = J \ vcat(row[:xdot], row[:ydot])
        
        if isapprox(fk(θ), vcat(row[:x], row[:y]); atol=1e-3) && isapprox(J*θdot, vcat(row[:xdot], row[:ydot]); atol=1e-3)
            success += 1
        end
        th[:,i] = θ
        thdot[:,i] = θdot
        i += 1
    end
    @info "My success rate = $(round(success/N; digits=3)*100)%" 
    my_solution = deepcopy(data)
    for k = 1:3
        insertcols!(my_solution, k+2, "th$(k)"=>th[k,:])
        insertcols!(my_solution, size(my_solution)[2]+1, "th$(k)dot"=>thdot[k,:])
    end

    CSV.write(str * "/ik_solution.csv", my_solution)
end

function check_student()
    str = "./submissions/group1"
    str_sol = "./data/group1"
    data = CSV.File(str_sol * "/ik_solution.csv") |> DataFrame
    student_solution = CSV.File(str * "/ik_solution.csv") |> DataFrame
    J = zeros(2,3)

    success::Int = 0
    i = 1
    for row in eachrow(student_solution)
        # θ_real = vcat(data[i,:][:th1], data[i,:][:th2], data[i,:][:th3])
        θ_student = vcat(row[:th1], row[:th2], row[:th3])
        θdot_student = vcat(row[:th1dot], row[:th2dot], row[:th3dot])
        jacobian!(J, θ_student)

        if isapprox(fk(θ_student), vcat(data[i,:][:x], data[i,:][:y]); atol=1e-3) && 
            isapprox(J*θdot_student, vcat(row[:xdot], row[:ydot]); atol=1e-3)
            success += 1
        end
        i += 1
    end
    @info "Student success rate = $(round(success/N; digits=3)*100)%" 
end