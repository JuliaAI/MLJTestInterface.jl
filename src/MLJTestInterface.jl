module MLJTestInterface

const N_MODELS_FOR_REPEATABILITY_TEST = 20

using MLJBase
using Pkg
using Test

include("attemptors.jl")
include("test.jl")
include("datasets.jl")


end # module
