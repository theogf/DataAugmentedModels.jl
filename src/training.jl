""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,convergence=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `convergence::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(
    model::AbstractGP{T,TLike,TInf},
    iterations::Int = 100;
    callback::Union{Nothing,Function} = nothing,
    convergence::Union{Nothing,Function} = nothing,
) where {T,TLike<:Likelihood,TInf<:Inference}
    if model.verbose > 0
        println(
            "Starting training $model with $(model.nSamples) samples, $(size(model.X,2)) features and $(model.nLatent) latent GP" *
            (model.nLatent > 1 ? "s" : ""),
        )
    end

    @assert iterations > 0 "Number of iterations should be positive"
    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter = 1
    conv = Inf
    p = Progress(iterations, dt = 0.2, desc = "Training Progress: ")
    prev_elbo = -Inf
    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            update_parameters!(model) #Update all the variational parameters
            model.Trained = true
            if !isnothing(callback)
                callback(model, model.inference.nIter) #Use a callback method if set by user
            end
            if (model.inference.nIter % model.atfrequency == 0) &&
               model.inference.nIter >= 3
                update_hyperparameters!(model) #Update the hyperparameters
            end
            # Print out informations about the convergence
            if model.verbose > 2 || (model.verbose > 1 && local_iter % 10 == 0)
                if isa(TInf, GibbsSampling)
                    next!(p; showvalues = [(:samples, local_iter)])
                else
                    if (model.verbose > 2 || local_iter % 10 == 0)
                        elbo = objective(model)
                        prev_elbo = elbo
                        next!(
                            p;
                            showvalues = [(:iter, local_iter), (:ELBO, elbo)],
                        )
                    else
                        next!(
                            p;
                            showvalues = [
                                (:iter, local_iter),
                                (:ELBO, prev_elbo),
                            ],
                        )
                    end
                end
            end
            local_iter += 1
            model.inference.nIter += 1
            (local_iter <= iterations) || break #Verify if the number of maximum iterations has been reached
        # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        catch e
            if isa(e, InterruptException)
                println("Training interrupted by user at iteration $local_iter")
                break
            else
                rethrow(e)
            end
        end
    end
    if model.verbose > 0
        println("Training ended after $(local_iter-1) iterations. Total number of iterations $(model.inference.nIter)")
    end
    computeMatrices!(model) #Compute final version of the matrices for prediction
    return model.Trained = true
end

function sample(model::MCGP{T,TLike,TInf},nSamples::Int=1000;callback::Union{Nothing,Function}=nothing,cat_samples::Bool=false) where {T,TLike<:Likelihood,TInf<:Inference}
    if model.verbose > 0
      println("Starting sampling $model with $(model.nSamples) samples with $(size(model.X,2)) features and $(model.nLatent) latent GP"*(model.nLatent > 1 ? "s" : ""))
    end
    @assert nSamples > 0  "Number of samples should be positive"
    return sample_parameters(model,nSamples,callback,cat_samples)
end

function update_parameters!(model::GP)
    analytic_updates!(model)
    computeMatrices!(model); #Recompute the matrices if necessary (when hyperparameters have been updated)
end

function update_parameters!(model::VGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end

"""Update all variational parameters of the sparse variational GP Model"""
function update_parameters!(m::SVGP)
    if isStochastic(m.inference)
        setMBIndices!(m.inference, StatsBase.sample(1:nSamples(m), nMinibatch(m.inference), replace = false))
        setxview!(m.inference, view(m.X, MBIndices(m.inference), :))
        setyview!(m.inference, view_y(m.likelihood, m.y, MBIndices(m.inference)))
    end
    computeMatrices!(m); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(m);
end

function update_parameters!(m::MOVGP)
    computeMatrices!(m); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    update_A!(m)
    variational_updates!(m);
end

function update_parameters!(m::MOSVGP)
    if isStochastic(m.inference)
        MBIndices = StatsBase.sample(1:nSamples(m.inference), nMinibatch(m.inference), replace = false)
        for i in 1:nX(m)
            setMBIndices!(m.inference, i, MBIndices)
            setxview!(m.inference, i, view(m.X[i], MBIndices(m.inference, i), :))
            setyview!(m.inference, i, view_y(m.likelihood,m.y[i], MBIndices(m.inference, i)))
        end
    end
    computeMatrices!(m); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    update_A!(m)
    variational_updates!(m);
end

function update_parameters!(m::VStP)
    computeMatrices!(m); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    local_prior_updates!(m);
    variational_updates!(m);
end

function computeMatrices!(m::GP{T}) where {T}
    compute_K!.(m.f, m.inference.xview, T(jitt) + first(m.likelihood.σ²))
    setHPupdated!(m.inference, false)
end

@traitfn function computeMatrices!(m::TGP) where {T,TGP<:AbstractGP{T};IsFull{TGP}}
    if isHPupdated(m.inference)
        compute_K!.(m.f, m.inference.xview, T(jitt))
    end
    setHPupdated!(m.inference, false)
end

@traitfn function computeMatrices!(m::TGP) where {T,TGP<:AbstractGP{T};!IsFull{TGP}}
    if isHPupdated(m.inference)
        compute_K!.(m.f, T(jitt))
    end
    #If change of hyperparameters or if stochatic
    if isHPupdated(m.inference) || isStochastic(m.inference)
        compute_κ!.(m.f, m.inference.xview, T(jitt))
    end
    setHPupdated!(m.inference, false)
end

# function computeMatrices!(model::VStP{T,<:Likelihood,<:Inference}) where {T}
#     if model.inference.HyperParametersUpdated
#         compute_K!.(model.f,[],T(jitt))
#         model.invL .= inv.(getproperty.(cholesky.(model.Knn),:L))
#         model.invKnn .= Symmetric.(inv.(cholesky.(model.Knn)))
#         # model.invKnn .= Symmetric.(model.invL.*transpose.(model.invL))
#     end
# end
