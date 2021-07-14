using LaTeXStrings
using PyCall, PyPlot

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