using CSV, DataFrames

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
        θ, model = ik_optimization(vcat(row[:x], row[:y]))
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