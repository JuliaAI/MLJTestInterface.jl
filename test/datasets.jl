@testset "loading of datasets" begin
    X, y = MTI.make_binary()
    @test X isa NamedTuple
    @test first(X) isa AbstractVector{Float64}
    @test MLJBase.scitype(y) == AbstractVector{MLJBase.OrderedFactor{2}}
    Xr, yr = MTI.make_binary(row_table=true)
    @test Xr isa AbstractVector
    @test MLJBase.Tables.rowtable(X) == Xr
    @test yr == y

    X, y = MTI.make_multiclass()
    @test X isa NamedTuple
    @test first(X) isa AbstractVector{Float64}
    @test MLJBase.scitype(y) == AbstractVector{MLJBase.Multiclass{3}}
    Xr, yr = MTI.make_multiclass(row_table=true)
    @test Xr isa AbstractVector
    @test MLJBase.Tables.rowtable(X) == Xr
    @test yr == y

    X, y = MTI.make_regression()
    @test X isa NamedTuple
    @test first(X) isa AbstractVector{Float64}
    @test MLJBase.scitype(y) == AbstractVector{MLJBase.Continuous}
    Xr, yr = MTI.make_regression(row_table=true)
    @test Xr isa AbstractVector
    @test MLJBase.Tables.rowtable(X) == Xr
    @test yr == y

    X, y = MTI.make_count()
    @test X isa NamedTuple
    @test first(X) isa AbstractVector{Float64}
    @test MLJBase.scitype(y) == AbstractVector{MLJBase.Count}
    Xr, yr = MTI.make_count(row_table=true)
    @test Xr isa AbstractVector
    @test MLJBase.Tables.rowtable(X) == Xr
    @test yr == y
end

true
