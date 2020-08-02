abstract type AbstractDataContainer end

mutable struct DataContainer{
    Tx<:Real,
    TX<:AbstractVector,
    Ty<:Real,
    TY<:AbstractVector,
} <: AbstractDataContainer
    X::TX # Feature vectors
    y::TY # Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int # Number of samples
    nDim::Int # Number of features per sample
end

function wrap_data(X::TX, y::TY) where {TX, TY}
    @assert size(y, 1) == length(X) "There is not the same number of samples in X and y"
    Tx = eltype(first(X))
    Ty = eltype(first(y))
    return DataContainer{Tx, TX, Ty, TY}(X, y, length(X), length(first(X)))
end

function wrap_X(X::AbstractMatrix{T}, obsdim = 2) where {T<:Real}
    return KernelFunctions.vec_of_vecs(X, obsdim = obsdim), T
end

function wrap_X(X::AbstractVector{T}) where {T<:Real}
    return X, T
end

function wrap_X(X::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    return X, T
end

nSamples(d::AbstractDataContainer) = d.nSamples
nDim(d::AbstractDataContainer) = d.nDim

input(d::AbstractDataContainer) = d.X
output(d::AbstractDataContainer) = d.y

mutable struct MODataContainer <: AbstractDataContainer

end