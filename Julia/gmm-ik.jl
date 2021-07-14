using LaTeXStrings
using GaussianMixtures

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
    # Conditions p(x, θ) on x = value to find p(θ ∣ x=value).
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