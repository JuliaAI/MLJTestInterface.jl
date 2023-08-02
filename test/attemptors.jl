@testset "attempt()" begin
    e = ArgumentError("elephant")
    bad() = throw(e)
    good() = 42

    @test (@test_logs MLJTestInterface.attempt(bad, "")) == (e, "×")
    @test(@test_logs(
        (:info, "look ×"),
        MLJTestInterface.attempt(bad, "look "),
    )  == (e, "×"))
    @test (@test_logs MLJTestInterface.attempt(good, "")) == (42, "✓")
    @test (@test_logs(
        (:info, "look ✓"),
        MLJTestInterface.attempt(good, "look "),
    )  == (42, "✓"))
    @test_throws e MLJTestInterface.attempt(bad, ""; throw=true)
end

struct DummyModel <: Deterministic end
MLJBase.package_name(::Type{<:DummyModel}) = "DummyPackage"
MLJBase.load_path(::Type{<:DummyModel}) = "DummyPackage.Some.Where.Over.The.Rainbow"

@testset "model_type" begin

    # test error thrown (not caught) if pkg missing from environment:
    @test_throws ArgumentError MLJTestInterface.model_type(
        DummyModel,
        @__MODULE__
    )

    M, outcome = MLJTestInterface.model_type(
        MLJDecisionTreeInterface.DecisionTreeClassifier,
        @__MODULE__;
        verbosity=0
    )
    @test M == MLJDecisionTreeInterface.DecisionTreeClassifier
    @test outcome == "✓"
end

struct DummyStatic <: Static end
MLJBase.transform(::DummyStatic, _, x, y) = hcat(x, y)
MLJBase.package_name(::Type{<:DummyStatic}) = "DummyPackage"
MLJBase.load_path(::Type{<:DummyStatic}) = "DummyPackage.Some.Thing.Different"

struct SupervisedTransformer <: Deterministic end
MLJBase.fit(::SupervisedTransformer, verbosity, X, y) = (42, nothing, nothing)
MLJBase.predict(::SupervisedTransformer, _, Xnew) = fill(4.5, length(Xnew))
MLJBase.transform(model::SupervisedTransformer, Θ, Xnew) =
    predict(model, Θ, Xnew)
MLJBase.package_name(::Type{<:SupervisedTransformer}) = "DummyPackage"
MLJBase.load_path(::Type{<:SupervisedTransformer}) =
    "DummyPackage.Some.Thing.Else"

@testset "operations" begin
    X = fill(1.2, 10)
    y = X
    mach = machine(SupervisedTransformer(), X, y) |> fit!
    operations, outcome = MLJTestInterface.operations(mach, X, y, throw=true)
    @test operations == "predict, transform"
    @test outcome == "✓"

    smach = machine(DummyStatic())
    operations, outcome = MLJTestInterface.operations(smach, X, y)
    @test operations == "transform"
    @test outcome == "✓"
end

true
