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

true
