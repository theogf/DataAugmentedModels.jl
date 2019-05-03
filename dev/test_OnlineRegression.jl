using Distributions, Random
using Plots
using MLDataPattern
using Clustering, LinearAlgebra
using ValueHistories
pyplot()
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses


function generate_random_walk_data(N,dim,lambda)
    d = Poisson(lambda)
    i = Vector()
    # if dim == 1
        push!(i,1)
        i_t = 1
        while i_t < N
            i_t = min(N,i_t+rand(d)+1)
            push!(i,i_t)
        end
    # else
        # push!(i,CartesianIndex(1,1))
        # i_t1 = 1; i_t2 = 1
        # while i_t1 < N && i_t2 < N
            # i_t1 = min(N,i_t1+rand(d))
            # i_t2 = min(N,i_t2+rand(d))
            # push!(i,CartesianIndex(i_t1,i_t2))
        # end
    # end
    return i
end

function generate_uniform_data(N,dim,box_width)
    X = rand(N,dim)*box_width
end


function generate_grid_data(N,dim,box_width)
    if dim == 1
        return reshape(collect(range(0,box_width,length=N)),N,1)
    else
        x1 = range(0,box_width,length=N)
        x2 = range(0,box_width,length=N)
        return hcat([i for i in x1, j in x2][:],[j for i in x1, j in x2][:])
    end
end

function generate_gaussian_data(N,dim,variance=1.0)
    if dim == 1
        d = Normal(0,1)
    else
        d = MvNormal(zeros(dim),variance*Diagonal{Float64}(I,dim))
    end
    X = rand(d,N)
end

function plotting1D(X,f,ind_points,pred_ind,X_test,pred,sig_pred,title;full=false)
    p = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
    if !full
        p = plot!(ind_points[:,1],pred_ind,t=:scatter,lab="",color=:red)
    end
    n_sig = 3
    p = plot!(X_test,pred+n_sig*sqrt.(sig_pred),fill=(pred-n_sig*sqrt.(sig_pred)),alpha=0.3,lab="")
    p =  plot!(X_test,pred,lab="",title=title)
    return p
end

function plotting2D(X,f,ind_points,pred_ind,x1_test,x2_test,pred,minf,maxf,title;full=false,ρ=0.0)
    N_test = size(x1_test,1)
    p = plot(x1_test,x2_test,reshape(pred,length(x1_test),length(x2_test))',t=:contour,clim=(minf,maxf),fill=true,lab="",title=title,xlims=(0,1),ylims=(0,1))
    # p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
    if !full
        θ = range(0.0,2π,length=100)
        for j in 1:size(ind_points,1)
            p = plot!(ind_points[j,1].+ρ*cos.(θ),ind_points[j,2].+ρ*sin.(θ),lab="",color=:black)
        end
        p = plot!(ind_points[:,1],ind_points[:,2],t=:scatter,lab="",color=1)
    end
    return p
end

function plottingtruth(X,f,X_test,x1_test,x2_test)
    N_test = size(x1_test,1)
    true_f = randomf(X_test)
    p = plot(x1_test,x2_test,reshape(true_f,N_test,N_test),t=:contour,fill=true,lab="",title="Truth")
    p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
end

function RMSE(y_pred,y_test)
    return norm(y_pred-y_test)/sqrt(length(y_test))
end

function LogLikelihood(y_pred,y_test)
    return -0.5*norm(y_pred-y_test)/noise^2-0.5*length(y_test)*log(2*π*noise^2)
end

function KLGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -0.5*N
    tot += 0.5*sum(log.(sig)-log.(sig_f)+(sig_f+(mu-f).^2)./sig)
    return tot
end

function sigpartKLGP(mu,sig,f,sig_f)
    return 0.5*sum(log.(sig))
end

function sigpart2KLGP(mu,sig,f,sig_f)
    return 0.5*sum(sig_f./(sig))
end

function mupartKLGP(mu,sig,f,sig_f)
    return sum(0.5*(mu-f).^2 ./sig)
end

function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1.0./(sig_f)+1.0./(sig)).*((mu-f).^2))
end

function randomf(X)
    return X[:,1].^2+cos.(X[:,1]).*sin.(X[:,2])-sin.(X[:,1]).*tanh.(X[:,2])-X[:,1]
    # return X[:,1].*sin.(X[:,2])
end


function sample_gaussian_process(X,noise)
    N = size(X,1)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end
rmse_list = []
m_list = []
ll_list = []
function callbackplot(model,iter)
    global rmse_list, m_list,ll_list
    y_ind = predict_y(model,model.Zalg.centers)
    y_pred,sig_test = proba_y(model,X_test)
    if iter == 1
        rmse_list = []
        m_list = []
        ll_list = []
    end
    if dim == 1
        p = plotting1D(X,y,model.Zalg.centers,y_ind,X_test,y_pred,sig_test,"$(typeof(model.Zalg)) (m=$(model.Zalg.k))")
        scatter!(model.X,model.y[1],color="black",lab="")
        display(p)
    elseif dim == 2
        # p = plotting2D(X,y,model.Zalg.centers,y_ind,x1_test,x2_test,y_pred,minf,maxf,"$(typeof(model.Zalg)) (m=$(model.Zalg.k))",ρ=getlengthscales(model.kernel[1])*sqrt(-2*log(model.Zalg.lim)))
        # p = scatter!(model.X[:,1],model.X[:,2],color="black",lab="")
        push!(rmse_list,RMSE(y_pred,y_test))
        push!(m_list,model.nFeature)
        push!(ll_list,-LogLikelihood(y_pred,y_test))
        # prmse = plot(1:iter,rmse_list,title="RMSE",lab="")
        # pm = plot(1:iter,m_list,title="M",lab="")
        # pll = plot(1:iter,ll_list,title="Log Likelihood",lab="")
        # lay = @layout [ a [a;b;c ]]
        # fullp = plot(p,prmse,pm,pll,layout=lay)
        # display(fullp)
        # display(p)
        # frame(anim)
    end
end

function callbacksave(metrics)
    function(model,iter)
        push!(metrics,:k_l,getlengthscales(model.kernel[1]))
        push!(metrics,:k_v,getvariance(model.kernel[1]))
        push!(metrics,:ELBO,-ELBO(model))
        y_ind = predict_y(model,model.Zalg.centers)
        y_pred,sig_test = proba_y(model,X_test)
        push!(metrics,:rmse,RMSE(y_pred,y_test))
        push!(metrics,:m,size(model.Z[1],1))
    end
end

kernel = AugmentedGaussianProcesses.RBFKernel(0.1)
dim = 2
monotone = true
sequential = true
n = 3000
N_test= 100
noise=0.001
# X = generate_uniform_data(n,dim,1.0)
X = generate_grid_data(floor(Int64,n^(1/dim)),dim,1.0)
N_test = floor(Int64,n^(1/dim))
indices = generate_random_walk_data(size(X,1),dim,3)

# X = generate_uniform_data(n,dim,5)
# X = generate_gaussian_data(n,dim)'
# X = (X.-mean(X))/sqrt(var(X))
if dim == 1
    y = sample_gaussian_process(X,noise)
    X_test = copy(X); y_test = copy(y)
    if monotone
        s = sortperm(vec(X))
        X = (X[s,:])[indices,:]; y = (y[s])[indices]
    else
        s = randperm(size(X,1))
        X = (X[s,:])[indices,:]; y = (y[s])[indices]
    end
    X_grid = range(minimum(X[:,1]),maximum(X[:,1]),length=N_test)
    x1_test= X_test; x2_test =X_test
elseif dim == 2
    y = sample_gaussian_process(X,noise)
    minf=minimum(y); maxf=maximum(y)
    X_test = copy(X); y_test = copy(y)
    if monotone
        s = sortperm(norm.(eachrow(X)))
        X = (X[s,:])[indices,:]; y = (y[s])[indices]
    else
        s = randperm(size(X,1))
        X = (X[s,:])[indices,:]; y = (y[s])[indices]
    end
    x1_test = range(minimum(X_test[:,1]),maximum(X_test[:,1]),length=floor(Int64,n^(1/dim)))
    x2_test = range(minimum(X_test[:,2]),maximum(X_test[:,2]),length=floor(Int64,n^(1/dim)))
end
if dim == 1
elseif dim == 2
end
k = 50
b = 20
dorand = false
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    f = f[randord]
end

### Non sparse GP :
t_full = @elapsed fullgp = GP(X,y,kernel,noise=noise)
train!(fullgp,iterations=10)
y_full,sig_full = proba_y(fullgp,X_test)
y_train,sig_train = proba_y(fullgp,X)
println("Full GP ($t_full s)\n\tRMSE (train) : $(RMSE(predict_y(fullgp,X),y))\n\tRMSE (test) : $(RMSE(y_full,y_test))")
if dim == 1
    pfull = plotting1D(X,y,[0],[0],X_test,y_full,sig_full,"Full batch GP",full=true)
elseif dim == 2
    pfull = plotting2D(X,y,[0 0],0,x1_test,x2_test,y_full,minf,maxf,"Full batch GP",full=true)
end


### Sparse Offline GP
t_sparse = @elapsed sparsegp = SVGP(X,y,kernel,GaussianLikelihood(noise),AnalyticVI(),20,Autotuning=true)
train!(sparsegp,iterations=1000)
y_sparse,sig_sparse = proba_y(sparsegp,X_test)
y_train,sig_train = proba_y(sparsegp,X)
println("Sparse GP ($t_sparse s)\n\tRMSE (train) : $(RMSE(predict_y(sparsegp,X),y))\n\tRMSE (test) : $(RMSE(y_sparse,y_test))")
if dim == 1
    psparse = plotting1D(X,y,[0],[0],X_test,y_sparse,sig_sparse,"Full batch GP",full=false)
elseif dim == 2
    psparse = plotting2D(X,y,sparsegp.Z[1],0,x1_test,x2_test,y_sparse,minf,maxf,"Sparse GP (m=$(sparsegp.nFeature))",full=false)
end


##### DeterminantalPointProcess for selecting points

# setfixed!(kernel.fields.variance)
anim = Animation()
metdpp = MVHistory()
t_dpp = @elapsed dppgp = OnlineVGP(kernel,GaussianLikelihood(noise),AnalyticSVI(24),DPPAlg(0.8,kernel),verbose=3,Autotuning=true,IndependentPriors=false)
# for (X_batch,y_batch) in eachbatch((X,y),size=10,obsdim=1)
for (X_batch,y_batch) in RandomBatches((X,y),size=10,count=30,obsdim=1)
    train!(dppgp,X_batch,y_batch,iterations=1,callback=callbacksave(metdpp))
    # train!(dppgp,X_batch,y_batch,iterations=10,callback=callbackplot)
end
# t_dpp = @elapsed train!(dppgp,iterations=100)
y_dpp,sig_dpp = proba_y(dppgp,X_test)
y_inddpp = predict_y(dppgp,dppgp.Zalg.centers)
y_traindpp, sig_traindpp = proba_y(dppgp,X)

println("DPP ($t_dpp s)\n\tRMSE (train) : $(RMSE(predict_y(dppgp,X),y))\n\tRMSE (test) : $(RMSE(y_dpp,y_test))")
kl_dpp = KLGP.(y_traindpp,sig_traindpp,y_train,sig_train)
kl_simple = KLGP.(y_traindpp,sig_traindpp,y_train,noise)
js_dpp = JSGP.(y_traindpp,sig_traindpp,y_train,sig_train)
js_simple = JSGP.(y_traindpp,sig_traindpp,y_train,noise)
if dim == 1
    pdpp = plotting1D(X,y,dppgp.Zalg.centers,y_inddpp,X_test,y_dpp,sig_dpp,"DPP (m=$(dppgp.Zalg.k))")
elseif dim == 2
    pdpp = plotting2D(X,y,dppgp.Zalg.centers,y_inddpp,x1_test,x2_test,y_dpp,minf,maxf,"DPP (m=$(dppgp.Zalg.k))")
end


#### Circle K finding method with constant limit
anim=Animation()
metcircle = MVHistory()
# setfixed!(kernel.fields.variance)
# setfree!(kernel.fields.variance)
newkernel = kernel
newkernel = RBFKernel(0.06)

t_circle = @elapsed circlegp = OnlineVGP(newkernel,GaussianLikelihood(noise),AnalyticVI(),CircleKMeans(0.8),verbose=3,Autotuning=true)
# t_circle = @elapsed train!(circlegp,iterations=15,callback=callbackplot)
# for (X_batch,y_batch) in eachbatch((X,y),size=10,obsdim=1)
for (X_batch,y_batch) in RandomBatches((X,y),size=10,count=200,obsdim=1)
    t_dpp = @elapsed train!(circlegp,X_batch,y_batch,iterations=1,callback=callbackplot)
    # t_dpp = @elapsed train!(circlegp,X_batch,y_batch,iterations=1,callback=callbacksave(metcircle))
end
y_circle,sig_circle = proba_y(circlegp,X_test)
y_indcircle = predict_y(circlegp,circlegp.Zalg.centers)
y_traincircle, sig_traincircle = proba_y(circlegp,X)

println("Circle KMeans ($t_circle s)\n\tRMSE (train) : $(RMSE(predict_y(circlegp,X),y))\n\tRMSE (test) : $(RMSE(y_circle,y_test))")
kl_circle = KLGP.(y_traincircle,sig_traincircle,y_train,sig_train)
kl_simple = KLGP.(y_traincircle,sig_traincircle,y_train,noise)
js_circle = JSGP.(y_traincircle,sig_traincircle,y_train,sig_train)
js_simple = JSGP.(y_traincircle,sig_traincircle,y_train,noise)
if dim == 1
    pcircle = plotting1D(X,y,circlegp.Zalg.centers,y_indcircle,X_test,y_circle,sig_circle,"Circle KMeans (m=$(circlegp.Zalg.k))")
elseif dim == 2
    pcircle = plotting2D(X,y,circlegp.Zalg.centers,y_indcircle,x1_test,x2_test,y_circle,minf,maxf,"Circle KMeans (m=$(circlegp.Zalg.k))")
end

##

# plot!(twinx(),X,[kl_const kl_simple js_const js_simple],lab=["KL" "KL_S" "JS" "JS_S"])
# plot!(twinx(),X,[kl_circle js_circle],lab=["KL" "JS"])
#plot!(X,y_trainconst+js_const,fill=(y_trainconst-js_const),alpha=0.3,lab="")

# pdiv_const = plot(X,kl_const,lab="KL")
# pdiv_const = plot!(twinx(),X,js_const,lab="JS",color=:red)
# pdiv_rand = plot(X,kl_rand,lab="KL")
# pdiv_rand = plot!(twinx(),X,js_rand,lab="JS",color=:red)


if dim == 2
    p = contourf(x1_test,x2_test,reshape(y_test,length(x1_test),length(x2_test))')
    display(plot(p,pdpp,psparse,pcircle,pfull));
else
    display(plot(pfull,psparse,pdpp,pcircle)); gui()
end

display(bar(["Full","Sparse","DPP","Circle"],RMSE.([y_full,y_sparse,y_dpp,y_circle],[y_test]),title="RMS_test",lab=""))
