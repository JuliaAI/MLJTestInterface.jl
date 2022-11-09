"""
    make_binary()

Return data `(X, y)` for the crabs dataset, restricted to the two features `:FL`,
`:RW`. Target is `Multiclass{2}`.

"""
function make_binary()
    data = MLJBase.load_crabs()
    y_, X = unpack(data, ==(:sp), col->col in [:FL, :RW])
    y = coerce(y_, MLJBase.OrderedFactor)
    return X, y
end

"""
    make_multiclass()

Return data `(X, y)` for the unshuffled iris dataset. Target is `Multiclass{3}`.

"""
make_multiclass() = MLJBase.@load_iris

"""
    make_regression()

Return data `(X, y)` for the Boston dataset, restricted to the two features `:LStat`,
`:Rm`. Target is `Continuous`.

"""
function make_regression()
    data = MLJBase.load_boston()
    y, X = unpack(data, ==(:MedV), col->col in [:LStat, :Rm])
    return X, y
end

"""
    make_count()

Return data `(X, y)` for the Boston dataset, restricted to the two features `:LStat`,
`:Rm`, with the `Continuous` target converted to `Count` (integer).

"""
function make_count()
    X, y_ = make_regression()
    y = map(η -> round(Int, η), y_)
    return X, y
end
