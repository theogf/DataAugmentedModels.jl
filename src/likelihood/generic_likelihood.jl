"""
Template file for automatically generated likelihood creation
"""

macro augmodel()
    @error "You need to provide a list of arguments, please check the documentation on the macro"
end

const InputSymbol = Union{Symbol,Expr,Real}

check_model_name(name::Symbol) = !isnothing(match(r"[^\w]*", string(name)))

function treat_likelihood(likelihood::Expr) end
using MacroTools
function treat_functions(expr, args)
    for arg in args
        expr = MacroTools.postwalk(expr) do x
            @capture(x, $(arg)) || return x
            return :(l.$(arg))
        end
    end
    if expr == :(0)
        return :(zero(T))
    elseif expr == :(1)
        return :(one(T))
    else
        return expr
    end
end
function correct_input(expr, input)
    return expr = MacroTools.postwalk(expr) do x
        @capture(x, x | r | f) || return x
        return input
    end
end


function correct_parenthesis(text::AbstractString)
    return replace(text, r"(?=[()*+-.])" => "\\")
end

function check_likelihoodtype(ltype)
    return ltype == :Regression || ltype == :Classification || ltype == :Event
end

const AGP = AugmentedGaussianProcesses

macro augmodel(name::Symbol, likelihoodtype::Symbol, likelihood::Expr) end

"""
    @augmodel(name, ltype, C, g, α, β, γ, φ, ∇φ; θ...)

Macro to create an augmentable likelihood, following the theory of 
"Automated Augmented Conjugate Inference for Non-conjugate Gaussian Process Models", Galy-Fajou et al.
You should try to write the likelihood as 
```math
p(f|y,θ) = C(y;θ)e^{g(y;θ) f}φ(α(y;θ) - β(y;θ)f + γ(y;θ)f^2; θ))
```

## Arguments
- `name` : The name of the likelihood
- `ltype` : The type of likelihood. Options are `Regression`, `Event` or `Classification`
- `C`, `g`, `α`, `β`, `γ`: Functions expressions depending uniquely on `θ` and `y`. For example `y / θ`
- `φ` : Positive definite radial function, input should be `x` and `θ`
- `∇φ` : Derivative of `φ` given `x`
- `θ` : Here you can set up all your hyperparameters by defining them as `β=2.0, c=1.0` etc. A default value is required!

The parameters you define in `θ` can then be used in your other functions.

## Example (Laplace)
```julia
@augmodel(AugLaplace, Regression, 0.5 / β, y^2, 2*y, 1, exp(-sqrt(x))/β), -exp(-sqrt(x)/β) / (2β * sqrt(x)), β=2.0)
d = AugLaplace(;β=3.0)
pdf(Laplace(0.0, 3.0), 2.0) == d(0.0, 2.0)
```

There are a few additional functions that you can define to override the existing generic implementations
```

- `Statistics.var(l::MyLikelihood)` for adding the variance correctly in `Regression` types of likelihood
- ``


"""
macro augmodel(
    name,
    ltype,
    C::InputSymbol,
    g::InputSymbol,
    α::InputSymbol,
    β::InputSymbol,
    γ::InputSymbol,
    φ::InputSymbol,
    # ∇φ::InputSymbol,
    addargs...,
)
    ## Check args here
    check_model_name(name) || error("Please only use alphabetic characters for the name of the likelihood")
    check_likelihoodtype(ltype) || error("Please use a correct likelihood type : Regression, Classification or Event")

    functions = Dict(
        :C=>C,
        :g=>g,
        :α=>α,
        :β=>β,
        :γ=>γ,
        :φ=>φ,
    )
    ## Create the needed fields and co
    fielddefs = quote end
    fielddefs.args = [:(c²::A), :(θ::A)]
    args = Any[]
    kwargs = Expr(:parameters)
    kwargsvar = Expr(:parameters)
    for input in addargs
        input.head == :(=) || error("Additional variables should be given a default value, for example `b=1`")
        var = input.args[1]
        val = input.args[2]
        push!(fielddefs.args, :($(var)::T))
        push!(args, var)
        push!(kwargs.args, Expr(:kw, var, val))
        push!(kwargsvar.args, Expr(:kw, var, var))
    end
    # Create the outer default constructor
    if length(args) > 0
        outerc = :($(name)($kwargs) = $(name){Float64}($kwargsvar))
    else
        outerc = :($(name)() = $(name){Float64}())
    end
    # Create inner constructor using kwargs only
    if length(args) > 0
        innerc1 = :( $(name){T}($(kwargs)) where {T<:Real} = new{T,Vector{T}}($(vcat(:([]), :([]), args)...)))
    else
        innerc1 = :( $(name){T}() where {T<:Real} = new{T,Vector{T}}([], []))
    end
    # Create inner constructor with additional parameters
    if length(args) > 0
        all_args = [kwargs, :(c²::A), :(θ::A)]
        innerc2 = :( $(name){T}($(all_args...)) where {T<:Real,A<:AbstractVector{T}}= new{T,A}(c², θ, $(args...)))
        # push!(innerc2.args[1].args, kwargs)
    else
        innerc2 = :( $(name){T}(c²::A, θ::A) where {T<:Real,A<:AbstractVector{T}} = new{T,A}(c², θ))
    end

    # Replace occurences of the given parameters in the functions
    for f in keys(functions)
        functions[f] = treat_functions(functions[f], args)
    end
    functions[:φ] = correct_input(functions[:φ], :r)
    # functions[:∇φ] = correct_input(functions[:∇φ], :r)

    if length(args) > 0
        l_args = [:(zeros(T, nSamplesUsed)), :(zeros(T, nSamplesUsed))]
        l_kwargs = Expr(:parameters)
        # all_args = [:(c²::A), :(θ::A), kwargs.args...]
        for arg in args
            push!(l_kwargs.args, Expr(:kw, arg, :(l.$(arg))))
        end
        l_args = vcat(l_kwargs, l_args)
        init_like = quote
            function AGP.init_likelihood(
                l::$(name){T},
                i::AGP.AbstractInference{T},
                nLatent::Int,
                nSamplesUsed::Int,
            ) where {T}
                if i isa AnalyticVI || i isa GibbsSampling
                    $(name){T}($(l_args...))
                else
                    $(name){T}($(l_kwargs))
                end
            end
        end
    else
        init_like = quote
            function AGP.init_likelihood(
                l::$(name){T},
                i::AGP.AbstractInference{T},
                nLatent::Int,
                nSamplesUsed::Int,
            ) where {T}
                if inference isa AnalyticVI || inference isa GibbsSampling
                    $(name){T}(zeros(T, nSamplesUsed), zeros(T, nSamplesUsed))
                else
                    $(name){T}()
                end
            end
        end
    end

    @show __module__
    return esc(generate_likelihood(
            Symbol(name), Symbol(ltype, "Likelihood"), functions, fielddefs, outerc, innerc1, innerc2, init_like
        ))
end
function generate_likelihood(lname, ltype, functions, fielddefs, outerc, innerc1, innerc2, init_like)
    quote
        begin
            using Statistics
            using Distributions
            struct $(lname){T<:Real,A<:AbstractVector{T}} <: AGP.$(ltype){T}
                $(fielddefs)
                $(innerc1)
                $(innerc2)
            end

            $(outerc)

            function AGP.implemented(
                ::$(lname), ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
            )
                return true
            end

            $(init_like)

            function (l::$(lname))(y::Real, f::Real)
                return _gen_C(l) *
                       exp(_gen_g(l, y) * f) *
                       _gen_φ(l, _gen_α(l, y) - _gen_β(l, y) * f + _gen_γ(l, y) * f^2)
            end

            function Distributions.loglikelihood(l::$(lname), y::Real, f::Real)
                return log(_gen_C(l)) +
                       _gen_g(l, y) * f +
                       log(_gen_φ(l, _gen_α(l, y) - _gen_β(l, y) * f + _gen_γ(l, y) * f^2))
            end

            function _gen_C(l::$(lname){T}) where {T<:Real}
                return $(functions[:C])
            end

            function _gen_g(l::$(lname), y::T) where {T<:Real}
                return $(functions[:g])
            end

            function _gen_α(l::$(lname), y::T) where {T<:Real}
                return $(functions[:α])
            end

            function _gen_β(l::$(lname), y::T) where {T<:Real}
                return $(functions[:β])
            end

            function _gen_γ(l::$(lname), y::T) where {T<:Real}
                return $(functions[:γ])
            end

            function _gen_φ(l::$(lname), r::T) where {T<:Real}
                return $(functions[:φ])
            end

            function _gen_∇φ(l::$(lname), r::T) where {T<:Real}
                return first(Zygote.gradient(Base.Fix1(_gen_φ, l), r))
            end

            function _gen_∇²φ(l::$(lname), r::T) where {T}
                return first(ForwardDiff.gradient(x->_gen_∇φ(l, first(x)), [r]))
            end

            function Base.show(io::IO, model::$(lname))
                return print(io, string($lname), " Likelihood")
            end

            function Statistics.var(l::$(lname){T}) where {T<:Real}
                @warn "The variance of the likelihood is not implemented : returnin 0.0"
                return zero(T)
            end

            function AGP.compute_proba(
                l::$(lname){T}, μ::AbstractVector{T}, σ²::AbstractVector{T}
            ) where {T<:Real}
                if typeof(l) <: RegressionLikelihood
                    return μ, max.(σ², zero(T)) .+ var(l)
                else
                    N = length(μ)
                    pred = zeros(T, N)
                    sig_pred = zeros(T, N)
                    for i in 1:N
                        x = AGP.pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
                        pred[i] = dot(AGP.pred_weights, AGP.pdf.(l, 1, x))
                        sig_pred[i] =
                            dot(AGP.pred_weights, AGP.pdf.(l, 1, x) .^ 2) - pred[i]^2
                    end
                    return pred, sig_pred
                end
            end

            ### Local Updates Section ###

            function AGP.local_updates!(
                l::$(lname), y::AbstractVector, μ::AbstractVector, diag_cov::AbstractVector
            ) where {T}
                l.c² .=
                    _gen_α.(l, y) - _gen_β.(l, y) .* μ +
                    _gen_γ.(l, y) .* (abs2.(μ) + diag_cov)
                return l.θ .= -_gen_∇φ.(l, l.c²) ./ _gen_φ.(l, l.c²)
            end

            function pω(l::$(lname), c²)
               AGP.LaplaceTransformDistribution(Base.Fix1(_gen_φ, l), c²)
            end

            function AGP.sample_local!(
                l::$(lname), y::AbstractVector, f::AbstractVector
            ) where {T}
                return set_ω!(
                    l,
                    rand.(
                        pω.(
                            l,
                            sqrt.(
                                0.5 * (
                                    l,
                                    _gen_α.(l, y) - _gen_β.(l, y) .* f +
                                    _gen_γ.(l, y) .* (abs2.(f)),
                                ),
                            ),
                        )
                    ),
                )
            end

            ### Natural Gradient Section ###

            @inline function AGP.∇E_μ(
                l::$(lname), ::AugmentedGaussianProcesses.AOptimizer, y::AbstractVector
            ) where {T}
                return (_gen_g.(l, y) + l.θ .* _gen_β.(l, y),)
            end
            @inline function AGP.∇E_Σ(
                l::$(lname), ::AugmentedGaussianProcesses.AOptimizer, y::AbstractVector
            ) where {T}
                return (l.θ .* _gen_γ.(l, y),)
            end

            ### ELBO Section ###
            function AGP.expec_loglikelihood(
                l::$(lname),
                i::AnalyticVI,
                y::AbstractVector,
                μ::AbstractVector,
                diag_cov::AbstractVector,
            ) where {T}
                tot = length(y) * log(_gen_C(l))
                tot += dot(_gen_g.(l, y), μ)
                tot +=
                    -(
                        dot(l.θ, _gen_α.(l, y)) - dot(l.θ, _gen_β.(l, y) .* μ) +
                        dot(l.θ, _gen_γ.(l, y) .* (abs2.(μ) + diag_cov))
                    )
                return tot
            end

            function AGP.AugmentedKL(l::$(lname), ::AbstractVector) where {T}
                return -dot(l.c², l.θ) - sum(log, _gen_φ.(l, l.c²))
            end

            ### Gradient Section ###

            @inline function AGP.∇loglikehood(
                l::$(lname){T}, y::Real, f::Real
            ) where {T<:Real}
                h² = _gen_α(l, y) - _gen_β(l, y) * f + _gen_γ(l, y) * f^2
                return _gen_g(l, y) +
                       (-_gen_β(l, y) + 2 * _gen_γ(l, y) * f) * _gen_∇φ(l, h²) / _gen_φ(l, h²)
            end

            @inline function AGP.hessloglikehood(
                l::$(lname){T}, y::Real, f::Real
            ) where {T<:Real}
                h² = _gen_α(l, y) - _gen_β(l, y) * f + _gen_γ(l, y) * f^2
                φ = _gen_φ(l, h²)
                ∇φ = _gen_∇φ(l, h²)
                ∇²φ = _gen_∇²φ(l, h²)
                return (
                    2 * _gen_γ(l, y) * ∇φ / φ - ((-_gen_β(l, y) + 2 * _gen_γ(l, y) * f) * ∇φ / φ)^2 +
                    (-_gen_β(l, y) + 2 * _gen_γ(l, y) * f)^2 * ∇²φ / φ
                )
            end
        end
    end
end
