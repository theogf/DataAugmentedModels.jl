###
# Distribution only based on its laplace transform function `f` and exponential tilting `c²`.
# The sampling is from Ridout, M. S. (2009). Generating random numbers from a distribution specified by its laplace transform. Statistics and Computing, 19(4):439.
# And the inverse laplace operation is done via Grassmann, W. K. (Ed.). (2000). Computational Probability. International Series in #Operations Research & Management Science. doi:10.1007/978-1-4757-4828-4
struct LaplaceTransformDistribution{T,Tφ,TAlg} <:
       Distributions.ContinuousUnivariateDistribution
    φ::Tφ # Laplace transform of the pdf
    c²::T # Exponential tilting parameter
    alg::TAlg # Algorithm to compute the inverse Laplace transform
    function LaplaceTransformDistribution{T,TAlg}(
        f::Tφ, c²::T, alg::TAlg
    ) where {T<:Real,Tφ,TAlg}
        _check_f(f) || error("The function passed is not valid") # Do a series of checks on f
        c² >= 0 || error("c² has to a be non-negative real")
        return new{T,Tφ,TAlg}(f, c², alg)
    end
end

function LaplaceTransformDistribution(
    f, c²::T=0.0, alg::TAlg=BromwichInverseLaplace()
) where {T,TAlg}
    return LaplaceTransformDistribution{T,TAlg}(f, c², alg)
end

function _check_f(f)
    return true # TODO Add tests for complete monotonicity / PDR
end
_gradf(d::LaplaceTransformDistribution, x::Real) = first(ForwardDiff.gradient(d.φ ∘ first, [x]))
function _gradlogf(d::LaplaceTransformDistribution, x::Real)
    return first(ForwardDiff.gradient(log ∘ d.φ ∘ first, [x]))
end
function _hessianlogf(d::LaplaceTransformDistribution, x::Real)
    return first(ForwardDiff.hessian(log ∘ d.φ ∘ first, [x]))
end

Distributions.pdf(dist::LaplaceTransformDistribution, x::Real) = apply_f(dist, x)
function Distributions.mean(dist::LaplaceTransformDistribution)
    return -_gradlogf(dist, dist.c²)# _gradf(dist, dist.c²) / dist.f(dist.c²)
end
function Distributions.var(dist::LaplaceTransformDistribution)
    return _hessianlogf(dist, dist.c²) / dist.φ(dist.c²) - mean(dist)^2
end

function Random.rand(dist::LaplaceTransformDistribution)
    return first(rand(dist, 1))
end

# Sampling from Ridout (09)
function Random.rand(dist::LaplaceTransformDistribution, n::Int)
    nTries = 100
    i = 0
    while i < nTries
        try
            if i > 10
                @warn "This is taking more than 10 tries! Trie $i/$nTries!"
            end
            return laptrans(dist; n=n)
        catch e
            if e isa AssertionError
                @warn "$(e.msg), Trying again"
                i += 1
                continue
            else
                rethrow(e)
            end
        end
    end
    @error "Sampler failed to converge"
end

struct BromwichInverseLaplace{T}
    A::Int
    l::Int
    m::Int
    n::Int
    expA2l::T
    ijπl::Vector{Complex{T}}
    expijπl::Vector{Complex{T}}
    coefs::Vector{T}
    altern::Vector{Int}
    b::Vector{T}
    s::Vector{T}
    function BromwichInverseLaplace(A::Int=19, l::Int=1, m::Int=11, n::Int=28)
        A > 0 || error("A should be a positive integer")
        l > 0 || error("l should be a positive integer")
        m > 0 || error("m should be a positive integer")
        n > 0 || error("n should be a positive integer")
        T = typeof(exp(A / (2 * l)))
        ijπl = 1im .* (1:l) .* π / l
        return new{T}(
            A,
            l,
            m,
            n,
            exp(A / (2 * l)),
            ijπl,
            exp.(ijπl),
            binomial.(m, 0:m) ./ (2^m),
            (-1) .^ (0:(n + m)),
            zeros(n + m + 1),
            zeros(n + m + 1),
        )
    end
end

@inline function apply_f(dist::LaplaceTransformDistribution, t)
    if t == 0
        return zero(t)
    else
        invlaplace(dist.φ, t, dist.alg)
    end
end

@inline function apply_F(dist::LaplaceTransformDistribution, t)
    if t == 0
        return zero(t)
    else
        invlaplace(s -> dist.φ(s) / s, t, dist.alg)
    end
end

function invlaplace(f::Function, t::Real, A::Int=19, l::Int=1, m::Int=11, n::Int=28)
    # n is the burnin, m is the number of components, A and l aare discretization parameters
    alg = BromwichInverseLaplace(A, l, m, n)
    return invlaplace(f, t, alg)
end

function invlaplace(f::Function, t::Real, alg::BromwichInverseLaplace)
    scaled_t = 1 / (2 * alg.l * t)
    alg.b[1] =
        f(alg.A * scaled_t) +
        2 * sum(real.(f.(alg.A * scaled_t .+ alg.ijπl ./ t) .* alg.expijπl))
    @inbounds for k in 1:(alg.n + alg.m)
        alg.b[k + 1] =
            2 * sum(
                real.(
                    f.(alg.A * scaled_t .+ alg.ijπl ./ t .+ 1im * π * k / t) .* alg.expijπl
                ),
            )
    end
    acoeff = alg.expA2l * scaled_t
    cumsum!(alg.s, acoeff .* alg.b .* alg.altern)
    return dot(view(alg.s, (alg.n + 1):length(alg.s)), alg.coefs)
end

# Laplace transform implemented from "Computational Probability Grassmann 2000"
function laptrans(
    dist::LaplaceTransformDistribution;
    n::Int=10,
    jmax::Int=500,
    kmax::Int=1000,
    b::Real=2.0,
    τ::Real=1e-7,
)
    # Step 1
    u = sort!(rand(n))
    # Step 2
    xmax = 0.1
    j = 0
    while apply_F(dist, xmax) < u[end] && j < jmax
        xmax = b * xmax
        j += 1
    end
    @assert j != jmax "Xmax search failed to converge"
    x = zeros(n)
    # Step 3
    for i in 1:n
        x_L = i == 1 ? 0.0 : x[i - 1]
        x_U = xmax
        k = 0
        t = x_L
        while abs(apply_F(dist, t) - u[i]) > τ && k < kmax
            k += 1
            t = t - (apply_F(dist, t) - u[i]) / apply_f(dist, t)
            if t < x_L || t > x_U
                t = 0.5 * (x_L + x_U)
            end
            if apply_F(dist, t) <= u[i]
                x_L = t
            else
                x_U = t
            end
        end
        @assert k != kmax "Modified Newton-Raphson procedure failed to converge for i=$i, x[$i]=$t"
        x[i] = t
    end
    # Step 4
    return shuffle!(x)
end
