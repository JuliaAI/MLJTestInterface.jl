const DOC_LIST_OF_TESTS1 =
    """
    Tests are applied in sequence. When a test fails, subsequent tests for
    that model are skipped. The following are applied to all models:
    """

const DOC_LIST_OF_TESTS2 =
    """
    - `:model_instance`: Create a default instance.

    - `:fitted_machine`: Bind instance to data in a machine and `fit!`. Call `report` and
      `fitted_params` on the machine.

    - `:operations`: Call implemented operations, such as `predict` and `transform`
    """

WARN_LEVEL(level) = "Only `level=1` and `level=2` tests supported. Using `level=$level`. "

_package_name(model) = MLJBase.package_name(model)
_name(model) = MLJBase.name(model)
_package_name(model::NamedTuple) = model.package_name
_name(model::NamedTuple) = model.name

# to update progress meter:
function next!(p)
    p.counter +=1
    MLJBase.ProgressMeter.updateProgress!(p)
end

const WARN_FAILURES_ENCOUNTERED =
    "Some errors were encountered. To isolate specific errors you may want "*
    "test again, specifiying `throw=true` to get a full stack trace. You may also want "*
    "limit tests to problem model(s). "

# for updating `failures` and `summary` tables output by `test(...)` below; returns the
# updated row, as added to `summary`:
function update!(summary, failures, row, i, test, value_or_exception, outcome)
    outcome_nt = NamedTuple{(test,)}((outcome,))
    updated_row = merge(row, outcome_nt)
    summary[i] = updated_row
    if outcome == "×"
        failures_row = (
            ; name=row.name,
            package_name=row.package_name,
            test=string(test),
            exception=value_or_exception
        )
        push!(failures, failures_row)
    end
    return updated_row
end

"""
    test(models, data...; mod=Main, level=2, throw=false, verbosity=1)

Apply a battery of MLJ integration tests to a collection of models, using `data` for
training. Here `mod` should be the module from which `test` is called (generally,
`mod=@__MODULE__` will work). Here `models` is a collection of model types implementing
the [MLJ model
interface](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/).

Code defining the model types to be tested must already be
loaded into the module `mod`.

The extent of testing is controlled by `level`:

|`level`          | description                    | tests (full list below) |
|:----------------|:-------------------------------|:------------------------|
| 1               | test code loading              | `:model_type`           |
| 2 (default)     | basic test of model interface  | all four tests          |

For extensive MLJ integration tests, instead use `MLJTestIntegration.test`, from
MLJTestIntegration.jl.

By default, exceptions caught in tests are not thrown. If `throw=true`, testing will
terminate at the first execption encountered, after throwing that exception (useful to
obtain stack traces).

# Return value

Returns `(failures, summary)` where:

- `failures`: table of exceptions thrown

- `summary`: table summarizing the outcomes of each test, where
  outcomes are indicated as below:

| entry | interpretation                     |
|:------|:-----------------------------------|
| ✓     | test succesful                     |
| ×     | test unsuccessful                  |
| n/a   | skipped because not applicable     |
| -     | test skipped for some other reason |

In the special case of `operations`, an empty entry, `""`, indicates that there don't
appear to be any operations implemented.

# Example

The following tests the model interface implemented by the `DecisionTreeClassifier` model
implemented in the package MLJDecisionTreeInterface.jl.

```julia
import MLJDecisionTreeInterface
import MLJTestInterface
using Test
X, y = MLJTestInterface.make_binary()
failures, summary = MLJTestInterface.test(
    [MLJDecisionTreeInterface.DecisionTreeClassifier, ],
    X, y,
    verbosity=0, # set to 2 when debugging
    throw=false, # set to `true` when debugging
    mod=@__MODULE__,
)
@test isempty(failures)
```

# List of tests

$DOC_LIST_OF_TESTS1

- `:model_type`: Check `load_path` trait is correctly overloaded by attempting to
  re-import the type based on that trait's value.

$DOC_LIST_OF_TESTS2

See also [`MLJTestInterface.make_binary`](@ref),
[`MLJTestInterface.make_multiclass`](@ref), [`MLJTestInterface.make_regression`](@ref),
[`MLJTestInterface.make_count`](@ref).

"""
function test(model_types, data...; mod=Main, level=2, throw=false, verbosity=1,)

    if level < 1 || level > 2
        level = level < 1 ? 1 : 2
        @warn WARN(level)
    end

    nmodels = length(model_types)

    # initiate return objects:
    failures = NamedTuple{(:name, :package_name, :test, :exception), NTuple{4, Any}}[]
    summary = Vector{NamedTuple{(
        :name,
        :package_name,
        :model_type,
        :model_instance,
        :fitted_machine,
        :operations,
    ), NTuple{6, String}}}(undef, nmodels)

    # summary table row corresponding to all tests skipped:
    row0 = (
        ; name="undefined",
        package_name= "undefined",
        model_type = "-",
        model_instance = "-",
        fitted_machine = "-",
        operations = "-",
    )

    if verbosity == 1
        meter = MLJBase.ProgressMeter.Progress(
            nmodels,
            desc = "Testing $nmodels models: ",
            barglyphs = MLJBase.ProgressMeter.BarGlyphs("[=> ]"),
            barlen = 25,
            color = :yellow
        )
    end

    for (i, model_type_candidate) in enumerate(model_types)

        verbosity == 1 && next!(meter)

        package_name = _package_name(model_type_candidate)
        name = _name(model_type_candidate)

        verbosity > 1 && @info "\nTesting $name from $package_name"

        row = merge(row0, (; name, package_name))

        # [model_type]:
        model_type, outcome = MLJTestInterface.model_type(
            model_type_candidate,
            mod;
            throw,
            verbosity,
        )
        row = update!(summary, failures, row, i, :model_type, model_type, outcome)
        outcome == "×" && continue

        level > 1 || continue

        # [model_instance]:
        model_instance, outcome =
            MLJTestInterface.model_instance(model_type; throw, verbosity)
        row = update!(summary, failures, row, i, :model_instance, model_instance, outcome)
        outcome == "×" && continue

        # [fitted_machine]:
        fitted_machine, outcome =
            MLJTestInterface.fitted_machine(model_instance, data...; throw, verbosity)
        row = update!(summary, failures, row, i, :fitted_machine, fitted_machine, outcome)
        outcome == "×" && continue

        # [operations]:
        operations, outcome =
            MLJTestInterface.operations(fitted_machine, data...; throw, verbosity)
        # special treatment to get list of operations in `summary`:
        if outcome == "×"
            row = update!(summary, failures, row, i, :operations, operations, outcome)
            continue
        else
            row = update!(summary, failures, row, i, :operations, operations, operations)
        end
    end

    isempty(failures) || verbosity > -1 && @warn WARN_FAILURES_ENCOUNTERED

    return failures, summary
end
