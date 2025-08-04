"""
    attempt(f, message; throw=false)

Return `(f(), "✓") if `f()` executes without throwing an exception. Otherwise, return
`(ex, "×"), where `ex` is the exception caught. Only truly throw the exception if
`throw=true`.

If `message` is not empty, then it is logged to `Info`, together with
the second return value ("✓" or "×").


"""
function attempt(f, message; throw=false)
    ret = try
        (f(), "✓")
    catch ex
        throw && Base.throw(ex)
        (ex, "×")
    end
    isempty(message) || @info message*last(ret)
    return ret
end

finalize(message, verbosity) = verbosity < 2 ? "" : message


# # ATTEMPTORS

root(load_path) = split(load_path, '.') |> first

function model_type(T, mod; throw=false, verbosity=1)
    # check interface package really is in current environment:
    message = "[:model_type] Loading model type "
    model_type, outcome = attempt(finalize(message, verbosity); throw) do
        load_path = MLJBase.load_path(T)
        load_path_ex = load_path |> Meta.parse
        api_pkg_ex = root(load_path) |> Symbol
        import_ex = :(import $api_pkg_ex)
        quote
            $import_ex
            $load_path_ex
        end |>  mod.eval
    end

    # catch case of interface package not in current environment:
    if outcome == "×" && model_type isa ArgumentError
        # try to get the name of interface package; if this fails we
        # catch the exception thrown but take no further
        # action. Otherwise, we test if in the original exception caught
        # above, `model_type`, was triggered because API package is
        # missing from in environment.
        api_pkg = try
            load_path = MLJBase.load_path(T)
            api_pkg = root(load_path)
        catch
            nothing
        end
        if !isnothing(api_pkg) &&
               api_pkg != "unknown" &&
               contains(model_type.msg, "$api_pkg not found in")
            Base.throw(model_type)
        end
    end

    return model_type, outcome
end

# helpers:
ismissing_or_isa(x, T) = ismissing(x) || x isa T
bad_trait(model_type) = "$model_type has a bad trait declaration.\n"

const err_is_pure_julia(model_type) = ErrorException(
    bad_trait(model_type)*"`is_pure_julia` must return `true` or `false`. "
)
const err_supports_weights(model_type) = ErrorException(
    bad_trait(model_type)*"`supports_weights` must return `true`, `false` or `missing`. "
)
const err_supports_class_weights(model_type) = ErrorException(
    bad_trait(model_type)*"`supports__class_weights` must return `true`, `false` or `missing`. "
)
const err_is_wrapper(model_type) = ErrorException(
    bad_trait(model_type)*"`is_wrapper` must return `true` or `false`. "
)
const err_package_name(model_type) = ErrorException(
    bad_trait(model_type)*"`package_name` must return a `String`. "
)
const err_packge_license(model_type) = ErrorException(
    bad_trait(model_type)*"`package_license` must return a `String`. "
)
const err_iteration_parameter(model_type) = ErrorException(
    bad_trait(model_type)*"`iteration_parameter` must return a `Symbol` or `nothing`. "
)

function traits(model_type; throw=false, verbosity=1)
    message = "[:traits] Apply smoke test to some model traits"
    attempt(finalize(message, verbosity); throw)  do
        ismissing_or_isa(MLJBase.is_pure_julia(model_type), Bool) ||
            throw(err_is_pure_julia(model_type))
        ismissing_or_isa(MLJBase.supports_weights(model_type), Bool) ||
            throw(err_supports_(model_type))
        ismissing_or_isa(MLJBase.supports_class_weights(model_type), Bool) ||
            throw(err_supports_class_weights(model_type))
        MLJBase.package_name(model_type) isa String ||
            throw(err_package_name(model_type))
        MLJBase.package_license(model_type) isa String ||
            throw(err_package_license(model_type))
        MLJBase.iteration_parameter(model_type) isa Union{Nothing,Symbol} ||
            throw(err_iteration_parameter(model_type))
        nothing
    end
end

function model_instance(model_type; throw=false, verbosity=1)
    message = "[:model_instance] Instantiating default model "
    attempt(finalize(message, verbosity); throw)  do
        model_type()
    end
end

function fitted_machine(model, data...; throw=false, verbosity=1)
    message = "[:fitted_machine] Fitting machine "
    attempt(finalize(message, verbosity); throw)  do
        mach = model isa Static ? machine(model) :
                                  machine(model, data...)
        fit!(mach, verbosity=-1)
        train = 1:MLJBase.nrows(first(data))
        model isa Static || fit!(mach, rows=train, verbosity=-1)
        model isa Static || fit!(mach, rows=:, verbosity=-1)
        MLJBase.report(mach)
        MLJBase.fitted_params(mach)
        mach
    end
end

function operations(fitted_machine, data...; throw=false, verbosity=1)
    message = "[:operations] Calling `predict`, `transform` and/or `inverse_transform` "
    attempt(finalize(message, verbosity); throw)  do
        model = fitted_machine.model
        operations = String[]
        methods = MLJBase.implemented_methods(fitted_machine.model)
        _, test = MLJBase.partition(1:MLJBase.nrows(first(data)), 0.01)
        if :predict in methods
            yhat = predict(fitted_machine, first(data))
            model isa Static || predict(fitted_machine, rows=test)
            model isa Static || predict(fitted_machine, rows=:)
            push!(operations, "predict")

            # check for double wrapped CategoricalValues in predict output for
            # classifiers:
            if target_scitype(model) <: AbstractVector{<:Finite} &&
                model isa Union{Deterministic,Probabilistic}
                η = model isa Deterministic ? first(yhat) : rand(first(yhat))
                unwrap(η) isa MLJBase.CategoricalArrays.CategoricalValue &&
                    error("Doubly wrapped CategoricalValue encountered. Check use of "*
                    "CategoricalArrays methods `levels` and `unique`, which changed in "*
                    "version 1.0. ")
            end
        end
        if :transform in methods
            W = if model isa Static
                transform(fitted_machine, data...)
            else
                transform(fitted_machine, first(data))
            end
            model isa Static || transform(fitted_machine, rows=test)
            model isa Static || transform(fitted_machine, rows=:)
            push!(operations, "transform")
            if :inverse_transform in methods
                inverse_transform(fitted_machine, W)
                push!(operations, "inverse_transform")
            end
        end
        join(operations, ", ")
    end
end
