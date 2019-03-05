module Discretization

abstract type AbstractField{T, D} <: Number end

@inline (f::AbstractField{T, D})(args::Tuple{Vararg{T, D}}) where {T} = f(args...)

struct ConstantField{T, D} <: AbstractField{T, D}
    value::T
end

Base.promote_type(::Type{<:ConstantField{T}}, ::Type{S<:Real}) where {T, S} = promote_type(T, S)

Base.:+(a::ConstantField, b::ConstantField) = ConstantField(a.value + b.value)

Base.:*(a::ConstantField, b::ConstantField) = ConstantField(a.value * b.value)



abstract type AbstractGrid{D} end

Base.ndims(::AbstractGrid{D}) where {D} = D

function Base.similar(g::AbstractGrid, element_type=eltype(g), dims=size(g))
    return similar(CartesianIndices(dims), element_type=element_type)
end

struct UniformCartesianGrid{D, T} <: AbstractGrid{D}
    X0::SVector{D, T} #Origin
    L::SVector{D, T}  #Length
    N::NTuple{D, Int} #size
end

Base.eltype(::UniformCartesianGrid{D, T}) where {D, T} = T
Base.size(grid::UniformCartesianGrid) = grid.N
Base.axes(grid::UniformCartesianGrid) = map(Base.OneTo, grid.N)

function findindex(grid::UniformCartesianGrid, X...)
    offset = X .- grid.X0
    I = Int.(fld.(offset .* grid.N, grid.L))
    return CartesianIndex(I)
end

struct GriddedField{T, D, F<:AbstractArray{<:AbstractField{T, D}}, G<:AbstractGrid} <: AbstractField{T, D}
    cells::F
    grid::G
end

#It's probabluy the case that we should apply some "remapped field" wrapper around the cells to encode
#which part of the grid they correspond to now
@inline (f::GriddedField{T, D})(args...) where {T} = f.cells[findindex(f.grid, args...)](args...)



end # module
