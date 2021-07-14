# using BenchmarkTools
using BSON: @save, @load     # enable if you want to load one of the .bson files

include("animation.jl")
include("three_link.jl")
include("gmm-ik.jl")
include("plotting.jl")
include("assignment.jl")

@load "long_and_big_training.bson" r
move_ee_cs(r)