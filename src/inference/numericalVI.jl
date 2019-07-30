"""
Solve any non-conjugate likelihood using Variational Inference
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood ad its gradients
"""
abstract type NumericalVI{T<:Real} <: Inference{T} end

include("quadratureVI.jl")
include("MCVI.jl")


""" `NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Variational Inference via numerical approximation.

**Argument**

    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,false)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,0,optimizer,false)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

""" `NumericalSVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Stochastic Variational Inference via numerical approximation.

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches
    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalSVI(nMinibatch::Integer,integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,true,nMinibatch)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,0,optimizer,true,nMinibatch)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mc"
    end
end

function Base.show(io::IO,inference::NumericalVI{T}) where T
    print(io,"$(inference.Stochastic ? "Stochastic numerical" : "Numerical") inference by $(isa(inference,MCIntegrationVI) ? "Monte Carlo Integration" : "Quadrature")")
end

function init_inference(inference::NumericalVI{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    inference.ρ = nSamples/nSamplesUsed
    inference.HyperParametersUpdated = true
    inference.optimizer_η₁ = [copy(inference.optimizer_η₁[1]) for _ in 1:nLatent]
    inference.optimizer_η₂ = [copy(inference.optimizer_η₂[1]) for _ in 1:nLatent]
    inference.∇η₁ = [zeros(T,nFeatures) for _ in 1:nLatent];
    inference.∇η₂ = [Symmetric(Diagonal(ones(T,nFeatures))) for _ in 1:nLatent]
    inference.∇μE = [zeros(T,nSamplesUsed) for _ in 1:nLatent];
    inference.∇ΣE = [zeros(T,nSamplesUsed) for _ in 1:nLatent]
    return inference
end

function variational_updates!(model::VGP{T,L,<:NumericalVI}) where {T,L}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates!(model::SVGP{T,L,<:NumericalVI}) where {T,L}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function natural_gradient!(model::VGP{T,L,<:NumericalVI}) where {T,L}
    model.inference.∇η₂ .= Symmetric.(Diagonal.(model.inference.∇ΣE) .- 0.5.*model.invKnn .- model.η₂)
    model.inference.∇η₁ .= model.inference.∇μE .+ model.invKnn.*(model.μ₀.-model.μ) - 2 .*model.inference.∇η₂.*model.μ
end

function natural_gradient!(model::SVGP{T,L,<:NumericalVI}) where {T,L}
    model.inference.∇η₁ .= model.Σ.*(model.inference.ρ.*transpose.(model.κ).*model.inference.∇μE .- model.invKmm.*model.μ)
    model.inference.∇η₂ .= Symmetric.(model.inference.ρ.*transpose.(model.κ).*Diagonal.(model.inference.∇ΣE).*model.κ.-0.5.*model.invKmm .- model.η₂)
end

function global_update!(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    model.η₁ .= model.η₁ .+ update.(model.inference.optimizer_η₁,model.inference.∇η₁)
    for k in 1:model.nLatent
        Δ = update(model.inference.optimizer_η₂[k],vcat(model.inference.∇η₁[1],model.inference.∇η₂[k][:]))
        Δ₁ = Δ[1:model.nFeatures]
        Δ₂ = reshape(Δ[model.nFeatures+1:end],model.nFeatures,model.nFeatures)
        # Δ = update(model.inference.optimizer_η₂[k],model.inference.∇η₂[k])
        α=1.0
        while !isposdef(-(model.η₂[k] + α*Δ₂)) &&  α > 1e-6
            α *= 0.1
        end
        if α <= 1e-6
            @error "α too small, postive definiteness could not be achieved"
        end
        model.η₂[k] = Symmetric(model.η₂[k] + α*Δ₂)
        model.η₁[k] = model.η₁[k] + α*Δ₁
        if isa(model.inference.optimizer_η₂[k],Adam)
            model.inference.optimizer_η₂[k].α = min(model.inference.optimizer_η₂[k].α * α*2.0,1.0)
            # model.inference.optimizer_η₁[k].α = min(model.inference.optimizer_η₁[k].α*α*2.0,1.0)
        elseif isa(model.inference.optimizer_η₂[k],VanillaGradDescent)
            # model.inference.optimizer_η₂[k].η = min(model.inference.optimizer_η₂[k].η*α*2.0,1.0)
            # model.inference.optimizer_η₁[k].η = min(model.inference.optimizer_η₁[k].η*α*2.0,1.0)
        elseif isa(model.inference.optimizer_η₂[k],ALRSVI)
        elseif isa(model.inference.optimizer_η₂[k],InverseDecay)
        end
    end
    model.Σ .= -0.5.*inv.(model.η₂)
    # model.μ .= model.η₁
    model.μ .= model.Σ.*model.η₁
end

function ELBO(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function η_ξ(μ::AbstractVector{<:Real},Σ)

end

function ξ_η(η₁::AbstractVector{<:Real},η₂)

end

function θ_ξ(μ::AbstractVector{<:Real},Σ::AbstractMatrix{<:Real})
    μ,Σ+μ*transpose(μ)
end

function ξ_θ(θ₁::AbstractVector{<:Real},θ₂::AbstractMatrix{<:Real})
    θ₁,θ₂-θ₁*transpose(θ₁)
end

function dL_dξxdξ_dθ(θ₁::AbstractVector{<:Real},θ₂::AbstractMatrix{<:Real},∇μL::AbstractVector{<:Real},∇Σ::AbstractVector{<:Real})

end

function expec_μ(model::AbstractGP{T,L,<:NumericalVI},index::Integer) where {T,L}
    return model.inference.∇μE[index]
end

function expec_μ(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    return model.inference.∇μE
end


function expec_Σ(model::AbstractGP{T,L,<:NumericalVI},index::Integer) where {T,L}
    return model.inference.∇ΣE[index]
end

function expec_Σ(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    return model.inference.∇ΣE
end

function global_update!(model::SVGP{T,L,NumericalVI}) where {T,L}
    if model.inference.Stochastic
    else
        model.η₁ .= model.inference.∇η₁ .+ model.η₁
        model.η₂ .= Symmetric.(model.inference.∇η₂ .+ model.η₂)
    end
    model.Σ .= -0.5*inv.(model.η₂)
    model.μ .= model.Σ.*model.η₁
end

function convert(::Type{T1},x::T2) where {T1<:VGP{<:Likelihood,T3} where {T3<:NumericalVI},T2<:VGP{<:Real,<:Likelihood,<:AnalyticVI}}
    #TODO Check if likelihood is compatible
    inference = T3(x.inference.ϵ,x.inference.nIter,x.inference.optimizer,defaultn(T3),x.inference.Stochastic,x.inference.nSamples,x.inference.nSamplesUsed,x.inference.MBIndices,x.inference.ρ,x.inference.HyperParametersUpdated,x.inference.∇η₁,x.inference.∇η₂,copy(expec_μ(x)),copy(expec_Σ(x)))
    likelihood =isaugmented(x.likelihood) ? remove_augmentation(x.likelihood) : likelihood
    return T1(x.X,x.y,x.nSample,x.nDim,x.nFeatures,x.nLatent,x.IndependentPriors,x.nPrior,x.μ,x.Σ,x.η₁,x.η₂,x.Knn,x.invKnn,x.kernel,likelihood,inference,x.verbose,x.optimizer,x.atfrequency,x.Trained)
end