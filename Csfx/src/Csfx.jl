# module Csfx

using Metalhead, Flux, Statistics
using MLUtils
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA

# Config stuff
@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false ## use gpu (if cuda available)
end

## Data Prep

function getdata(args)
    # ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # ## Load dataset
    # xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    # xtest, ytest = MLDatasets.MNIST(:test)[:]

    # ## Reshape input data to flatten each image into a linear array
    # xtrain = Flux.flatten(xtrain)
    # xtest = Flux.flatten(xtest)

    # ## One-hot-encode the labels
    # ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # ## Create two DataLoader objects (mini-batch iterators)
    # train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    # test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    data = rand(Float32, 384, 384, 3, 100), rand(Bool, 7, 100)
    Xs, Ys = shuffleobs((data))
    cv_data, test_data = splitobs((Xs, Ys); at=0.85)

    train_loader = DataLoader((cv_data), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((test_data), batchsize=args.batchsize)

    return train_loader, test_loader
end

## Model building
# effnet = Metalhead.efficientnetv2(:small).layers[1][1:end-1]
# model = Chain(effnet,
#               AdaptiveMeanPool((1, 1)), # These next three are a rewrite of the last layer
#               Flux.flatten,
#               Dense(1280 => 7))

## I think the OG uses torch feature extractor because he's loading the pretrained weights. I can just build
## a model like this
function build_model(; imgaize=(384, 384, 3))
    return Metalhead.efficientnetv2(:small; nclasses=7)
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        yhat = model(x)
        ls += Flux.logitbinarycrossentropy(yhat, y)
        acc += sum((sigmoid(yhat) .> 0.5) .== y)
        num +=  size(x)[end]
    end
    return ls / num, acc / num
end

function train(; kws...)
    args = Args(; kws...) ## Collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    ## Create test and train dataloaders
    train_loader, test_loader = getdata(args)

    ## Construct model
    model = build_model() |> device
    ps = Flux.params(model) ## model's trainable parameters

    ## Optimizer
    opt = Adam(args.η)

    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y) ## transfer data to device
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) ## compute gradient
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end

        ## Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end
end


# end # module

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
