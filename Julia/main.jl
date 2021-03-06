# using BenchmarkTools
using BSON: @save, @load     # enable if you want to load one of the .bson files

include("three_link.jl")
include("animation.jl")
include("gmm-ik.jl")
include("plotting.jl")
include("assignment.jl")

@load "long_and_big_training.bson" r
tl, th, thdot, thddot = move_ee_cs(r);