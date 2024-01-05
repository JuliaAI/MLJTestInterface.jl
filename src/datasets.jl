"""
    make_binary(; row_table=false)

Return data `(X, y)` for the crabs dataset, restricted to the two features `:FL`,
`:RW`. Target is `Multiclass{2}`.

The table `X` is a named tuple of vectors. For a vector of named tuples, set
`row_table=true`.

"""
function make_binary(; row_table=false)
    data = MLJBase.load_crabs()
    y_, X = unpack(data, ==(:sp), col->col in [:FL, :RW])
    y = coerce(y_, MLJBase.OrderedFactor)
    row_table ? (MLJBase.Tables.rowtable(X), y) : (X, y)
end

"""
    make_multiclass(; row_table=false)

Return data `(X, y)` for the unshuffled iris dataset. Target is `Multiclass{3}`.

"""
function make_multiclass(; row_table=false)
    X, y = MLJBase.@load_iris
    row_table ? (MLJBase.Tables.rowtable(X), y) : (X, y)
end

"""
    make_regression(; row_table=false)

Return data `(X, y)` for the Boston dataset, restricted to the two features `:LStat`,
`:Rm`. Target is `Continuous`.

The table `X` is a named tuple of vectors. For a vector of named tuples, set
`row_table=true`.

"""
function make_regression(; row_table=false)
    data = MLJBase.load_boston()
    y, X = unpack(data, ==(:MedV), col->col in [:LStat, :Rm])
    row_table ? (MLJBase.Tables.rowtable(X), y) : (X, y)
end

"""
    make_count(; row_table=false)

Return data `(X, y)` for the Boston dataset, restricted to the two features `:LStat`,
`:Rm`, with the `Continuous` target converted to `Count` (integer).

The table `X` is a named tuple of vectors. For a vector of named tuples, set
`row_table=true`.

"""
function make_count(; row_table=false)
    X, y_ = make_regression()
    y = map(η -> round(Int, η), y_)
    row_table ? (MLJBase.Tables.rowtable(X), y) : (X, y)
end
