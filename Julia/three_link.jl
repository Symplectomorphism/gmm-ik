using Random, Distributions
using Clustering
using LinearAlgebra
using GaussianMixtures
using PyCall, PyPlot
using JuMP, Ipopt
using BSON: @save, @load     # enable if you want to load one of the .bson files

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
        
        if verbose
            println("Iteration: $k, |Δμ| = $(round(μ_error, digits=6)), |ΔΣ| = $(round(Σ_error, digits=6))")
        end

        k += 1
        k > maxiter ? break : nothing
    end
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
    while θ[3] >= θ3 + 0.1 || θ[3] <= θ3 - 0.1
        θ = rand(d)
        counter += 1
        if counter > 1000 
            @warn "Satisfactory approximate solution not found." 
            break
        end
    end
    return θ
end

function predict_elbow_down(r::ThreeLink, x::Vector; mode::Symbol=:slse)
    μ, Σ = predict(r, x; mode=mode)
    d = MvNormal(μ, Σ)
    θ = rand(d)
    while θ[2] <= 0
        θ = rand(d)
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


function solve_optimization(x::Vector)
    model = Model(Ipopt.Optimizer)
    @variable(model, θ[1:3])
    @constraint(model, -π .<= θ .<= π)
    @NLobjective(model, Min, 
        (cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(θ[1]+θ[2]+θ[3]) - x[1])^2 + 
        (sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(θ[1]+θ[2]+θ[3]) - x[2])^2
    )
    optimize!(model)
    return value.(θ)
end

function solve_optimization(x::Vector, θ3::Float64)
    model = Model(Ipopt.Optimizer)
    @variable(model, θ[1:3])
    @constraint(model, -π .<= θ .<= π)
    @constraint(model, θ[3] == θ3)
    @NLobjective(model, Min, 
        (cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(θ[1]+θ[2]+θ[3]) - x[1])^2 + 
        (sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(θ[1]+θ[2]+θ[3]) - x[2])^2
    )
    optimize!(model)
    return value.(θ)
end

function solve_optimization(x::Vector; start::Vector)
    model = Model(Ipopt.Optimizer)
    @variable(model, θ[1:3])
    @constraint(model, -π .<= θ .<= π)
    @NLobjective(model, Min, 
        (cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(θ[1]+θ[2]+θ[3]) - x[1])^2 + 
        (sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(θ[1]+θ[2]+θ[3]) - x[2])^2
    )
    set_start_value.(θ, start)
    optimize!(model)
    return value.(θ)
end

function solve_optimization(x::Vector, θ3::Float64; start::Vector)
    model = Model(Ipopt.Optimizer)
    @variable(model, θ[1:3])
    @constraint(model, -π .<= θ .<= π)
    @constraint(model, θ[3] == θ3)
    @NLobjective(model, Min, 
        (cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(θ[1]+θ[2]+θ[3]) - x[1])^2 + 
        (sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(θ[1]+θ[2]+θ[3]) - x[2])^2
    )
    set_start_value.(θ, start)
    optimize!(model)
    return value.(θ)
end

function solve_elbow_down_optimization(x::Vector; start::Vector)
    model = Model(Ipopt.Optimizer)
    @variable(model, θ[1:3])
    @constraint(model, -π .<= θ .<= π)
    @constraint(model, θ[2] >= 0)
    @NLobjective(model, Min, 
        (cos(θ[1]) + cos(θ[1]+θ[2]) + 1/2*cos(θ[1]+θ[2]+θ[3]) - x[1])^2 + 
        (sin(θ[1]) + sin(θ[1]+θ[2]) + 1/2*sin(θ[1]+θ[2]+θ[3]) - x[2])^2
    )
    set_start_value.(θ, start)
    optimize!(model)
    return value.(θ)
end


function hypertrain_M(;N=2001)
    for M = 181:10:201
        r = ThreeLink(N=N, M=M)
        try execute_em!(r; maxiter=100, verbose=true) catch end
        avg_cost = test_training(r; nPoints=200)
        println("Average Cost(N=$M) = $(avg_cost)")
    end
end

function hypertrain_N(;M=101)
    for N = 1001:1000:7001
        r = ThreeLink(N=N, M=M)
        try execute_em!(r; maxiter=100, verbose=true) catch end
        avg_cost = test_training(r; nPoints=200)
        println("Average Cost(N=$N) = $(avg_cost)\n")
    end
end


function generate_cartesian_distribution(r::ThreeLink; nPoints::Int=100)
    xmin, xmax = (minimum(r.ξ[1,:]), maximum(r.ξ[1,:]))
    ymin, ymax = (minimum(r.ξ[2,:]), maximum(r.ξ[2,:]))
    dx = Uniform(xmin, xmax)
    dy = Uniform(ymin, ymax)
    x = [rand(dx), rand(dy)]


    μ, Σ = predict(r, x)
    d = MvNormal(μ, Σ)
    θ_dist = rand(d, nPoints)
    x_dist = hcat([fk(θ_dist[:,i]) for i = 1:nPoints]...)
    
    fig = figure(1);
    fig.clf()
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


function generate_cartesian_distribution(r::ThreeLink, x::Vector; nPoints::Int=100, record::Bool=false)
    μ, Σ = predict(r, x; mode=:slse)
    d = MvNormal(μ, Σ)
    θ_dist = rand(d, nPoints)
    x_dist = hcat([fk(θ_dist[:,i]) for i = 1:nPoints]...)
    
    fig = figure(1);
    fig.clf()
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

    if record
        fig.savefig("../TeX/figures/sample_solution-v1.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
end


function plot_posterior(r::ThreeLink, x::Vector=[-1.5, -0.4]; record::Bool=false)
    # Plot most likely posterior N*(θ | x) by marginalizing θ3
    μ, Σ = predict(r, x)

    μ12 = μ[1:2]
    Σ12 = Σ[1:2,1:2]
    d = MvNormal(μ12, Σ12)


    fig = figure(2)
    fig.clf()
    fig.suptitle(L"$\mathcal{N}(\theta_1, \theta_2 \mid x = [-1.5, -0.4])$ (marginalized over $\theta_3$)", fontsize=16)
    
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
    fig.suptitle(L"$\mathcal{N}(\theta_1, \theta_2 \mid x = [-1.5, -0.4])$ (conditioned at $\theta_3 = {%$(round(θ3; digits=2))}$)", fontsize=16)
    
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


function plot_full_posterior(r::ThreeLink, x::Vector=[-1.5, -0.4]; record::Bool=false)
    # Plot full posterior P(θ | x) by marginalizing θ3
    π_θ_tilde, μ_θ_tilde, Σ_θθ_tilde = conditionalize(r, x)

    d = Array{MvNormal, 1}()

    for j = 1:r.M
        push!(d, MvNormal(μ_θ_tilde[1:2,j], Σ_θθ_tilde[j][1:2,1:2]))
    end

    fig = figure(5)
    fig.clf()
    fig.suptitle(L"$P(\theta_1, \theta_2 \mid x = [-1.5, -0.4])$ (marginalized over $\theta_3$)", fontsize=16)

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


function plot_full_posterior(r::ThreeLink, θ3::Float64, x::Vector=[-1.5, -0.4]; record::Bool=false)
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
    fig.suptitle(L"$P(\theta_1, \theta_2 \mid x = [-1.5, -0.4])$ (conditioned at $\theta_3 = {%$(round(θ3; digits=2))}$)", fontsize=16)

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
        fig.savefig("../TeX/figures/full_posterior_marginal.png", dpi=600, 
            bbox_inches="tight", format="png")
    end
end


function plot_marginals_sequentially(x::Vector; maxiter::Int=9)
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