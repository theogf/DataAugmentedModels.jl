"""
Bayesian SVM (hinge loss function) : ``p(y|f) =  ``

"""
abstract type AbstractBayesianSVM{T<:Real} <: ClassificationLikelihood{T} end

function pdf(l::AbstractBayesianSVM,y::Real,f::Real)
    svmlikelihood(y*f)
end

function Base.show(io::IO,model::AbstractBayesianSVM{T}) where T
    print(io,"Bayesian SVM")
end

"""Return likelihood equivalent to SVM hinge loss"""
function svmlikelihood(f::Real)
    pos = svmpseudolikelihood(f)
    return pos./(pos.+svmpseudolikelihood(-f))
end

"""Return the pseudo likelihood of the SVM hinge loss"""
function svmpseudolikelihood(f::Real)
    return exp.(-2.0*max.(1.0.-f,0))
end



function compute_proba(l::AbstractBayesianSVM{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = svmlikelihood(μ[i])
        else
            pred[i] =  expectation(svmlikelihood,Normal(μ[i],sqrt(σ²[i])))
        end
    end
    return pred
end

###############################################################################

struct BayesianSVM{T<:Real} <: AbstractBayesianSVM{T}
    α::AbstractVector{AbstractVector{T}}
    θ::AbstractVector{AbstractVector{T}}
    function BayesianSVM{T}() where {T<:Real}
        new{T}()
    end
    function BayesianSVM{T}(α::AbstractVector{<:AbstractVector{<:Real}},θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
        new{T}(α,θ)
    end
end

function BayesianSVM()
    BayesianSVM{Float64}()
end

isaugmented(::BayesianSVM{T}) where T = true

function init_likelihood(likelihood::BayesianSVM{T},nLatent::Integer,nSamplesUsed) where T
    BayesianSVM{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
end


function local_updates!(model::VGP{<:BayesianSVM,<:AnalyticInference})
    model.likelihood.α .= broadcast((μ,Σ,y)->abs2.(1.0 .- y.*μ) + Σ ,model.μ,diag.(model.Σ),model.y)
    model.likelihood.θ .= broadcast(α->1.0./sqrt.(α),model.likelihood.α)
end

function local_updates!(model::SVGP{<:BayesianSVM,<:AnalyticInference})
    model.likelihood.α .= broadcast((κ,μ,Σ,y,K̃)->abs2.(1.0 .- y[model.inference.MBIndices].*(κ*μ)) + opt_diag(κ*Σ,κ) + K̃,model.κ,model.μ,diag.(model.Σ),model.y,model.K̃)
    model.likelihood.θ .= broadcast(α->1.0./sqrt.(α),model.likelihood.α)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:BayesianSVM},index::Integer)
    return model.y[index].*(model.likelihood.θ[index].+1.0)
end

function ∇μ(model::VGP{<:BayesianSVM})
    return broadcast((y,θ)->y.*(θ.+1.0),model.y,model.likelihood.θ)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:BayesianSVM},index::Integer)
    return model.y[index][model.inference.MBIndices].*(model.likelihood.θ[index].+1.0)
end

function ∇μ(model::SVGP{<:BayesianSVM})
    return broadcast((y,θ)->y[model.inference.MBIndices].*(θ.+1.0),model.y,model.likelihood.θ)
end

function expec_Σ(model::AbstractGP{<:BayesianSVM},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{<:BayesianSVM})
    return 0.5*model.likelihood.θ
end

function ELBO(model::AbstractGP{<:BayesianSVM})
    return expecLogLikelihood(model) - GaussianKL(model) - GIGKL(model)
end

function expecLogLikelihood(model::VGP{BayesianSVM{T}}) where T
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((μ,y,θ,Σ)->(sum(μ.*y)-0.5*dot(θ,Σ+abs2.(1.0.-y.*μ))),
                        model.μ,model.y,model.likelihood.θ,diag.(model.Σ)))
    return tot
end

function expecLogLikelihood(model::SVGP{BayesianSVM{T}}) where T
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->(sum(κμ.*y)-0.5*dot(θ,K̃+κΣκ+abs2.(1.0.-y.*κμ))),
                        model.κ.*model.μ,model.y,model.likelihood.θ,opt_diag(model.κ.*model.Σ,model.κ'),model.K̃))
    return model.inference.ρ*tot
end