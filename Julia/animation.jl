using MeshCat, CoordinateTransformations, Rotations
using GeometryBasics
using Colors: RGBA, RGB
using Blink
# Blink.AtomShell.install()

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
    # θ0 = ik_elbow_up_optimization(x0; start=predict_elbow_up(r, x0))
    θ0 = ik_elbow_down_optimization(x0, deg2rad(-90); start=[2.6, 1.96, -π/2])
    @info [fk(θ0); θ0]
    # θf = ik_elbow_up_optimization(xf; start=predict_elbow_up(r, xf))
    θf = ik_elbow_up_optimization(xf, -0.42; start=[1.42, -1.69, -0.42])
    # θf = ik_optimization(xf; start=θ0)
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