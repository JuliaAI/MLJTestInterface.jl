# MLJTestInterface.jl

Package for testing an implementation of the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) model interface.

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/bcgov/repomountie/blob/master/doc/lifecycle-badges.md) [![Build Status](https://github.com/JuliaAI/MLJTestInterface.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJTestInterface.jl/actions) [![Coverage](https://codecov.io/gh/JuliaAI/MLJTestInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJTestInterface.jl?branch=master) 

# Installation

```julia
using Pkg
Pkg.add("MLJTestInterface")
```

# Usage

To test that a collection of model types, `models`, satisfy the [MLJ model interface
requirements](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/),
use the `MLJTestInterface.test` function:

```julia
MLJTestInterface.test(models, data...; mod=Main, level=2, throw=false, verbosity=1) 
    -> failures, summary
```

Here `data` is training data acceptable to all the specified `models`, as would appear in
a call `MLJModelInterface.fit(model_instance, verbosity, data...)`.

For detailed documentation, run `using MLJTestInterface; @doc MLJTestInterface.test`.


# Example

The following tests the model interface implemented by the `DecisionTreeClassifier` model
implemented in the package MLJDecisionTreeInterface.jl.

```julia
import MLJDecisionTreeInterface
import MLJTestInterface
using Test
X, y = MLJTestInterface.make_multiclass()
failures, summary = MLJTestInterface.test(
    [MLJDecisionTreeInterface.DecisionTreeClassifier, ],
    X, y,
    verbosity=0, # set to 2 when debugging
    throw=false, # set to `true` when debugging
    mod=@__MODULE__,
)
@test isempty(failures)
```

# Datasets

The following commands generate small datasets of the form `(X, y)` suitable for interface
tests:

- `MLJTestInterface.make_binary` 

- `MLJTestInterface.make_multiclass` 

- `MLJTestInterface.make_regression` 

- `MLJTestInterface.make_count` 

