// TrainerCNN.fs
module TrainerCNN

type CNNTrainingConfig = {
    Epochs: int
    BatchSize: int
    LearningRate: float
    Optimizer: Optimizers.Optimizer option
    Loss: Losses.LossFunction
    Verbose: bool
    WeightDecay: float
    RandomSeed: int option
}

type CNNTrainingMetrics = {
    TrainLoss: float list
    EpochsCompleted: int
}

let defaultConfig = {
    Epochs = 5
    BatchSize = 32
    LearningRate = 0.01
    Optimizer = None
    Loss = Losses.CrossEntropy
    Verbose = true
    WeightDecay = 0.0
    RandomSeed = None
}

let train config (network: ConvLayers.CNNNetwork) (dataset: Data.Dataset) channels height width =
    printfn "Starting CNN training..."
    printfn "Network layers: %d" (List.length network)
    printfn "Training samples: %d" (dataset.Features.GetLength(0))
    printfn "Input shape: %dx%dx%d" channels height width
    printfn ""

    let mutable currentNetwork = network
    let optimizer =
        match config.Optimizer with
        | Some opt -> opt
        | None -> Optimizers.SGD config.LearningRate

    let mutable optimizerState = ConvLayers.initOptimizerState currentNetwork optimizer
    let mutable trainLosses = []

    for epoch in 1 .. config.Epochs do
        let shuffled =
            match config.RandomSeed with
            | Some seed -> Data.shuffleWithSeed (seed + epoch - 1) dataset
            | None -> Data.shuffle dataset
        let batches = Data.batch config.BatchSize shuffled

        let mutable epochLoss = 0.0
        let mutable batchCount = 0

        for batch in batches do
            batchCount <- batchCount + 1

            let inputTensor = ConvLayers.matrixToTensor batch.Features channels height width
            let finalOutput, caches = ConvLayers.forwardNetwork currentNetwork (ConvLayers.Tensor4D inputTensor)

            let predictions =
                match finalOutput with
                | ConvLayers.Matrix matrix -> matrix
                | ConvLayers.Tensor4D _ -> failwith "Final CNN layer must output matrix (usually Dense Softmax)"

            let loss = Losses.compute config.Loss predictions batch.Labels
            epochLoss <- epochLoss + loss

            let lossGrad = Losses.gradient config.Loss predictions batch.Labels
            let _, gradients = ConvLayers.backwardNetwork currentNetwork caches (ConvLayers.Matrix lossGrad)

            let gradientsWithDecay =
                if config.WeightDecay <= 0.0 then
                    gradients
                else
                    List.map2 (fun layer grad ->
                        match layer, grad with
                        | ConvLayers.Conv2D conv, ConvLayers.ConvGrad (gradFilters, gradBias) ->
                            let decayedFilters =
                                Array4D.init
                                    (gradFilters.GetLength(0))
                                    (gradFilters.GetLength(1))
                                    (gradFilters.GetLength(2))
                                    (gradFilters.GetLength(3))
                                    (fun i j k l -> gradFilters[i, j, k, l] + config.WeightDecay * conv.Filters[i, j, k, l])
                            ConvLayers.ConvGrad (decayedFilters, gradBias)
                        | ConvLayers.Dense dense, ConvLayers.DenseGrad (gradW, gradB) ->
                            let decayedW =
                                Array2D.init
                                    (gradW.GetLength(0))
                                    (gradW.GetLength(1))
                                    (fun i j -> gradW[i, j] + config.WeightDecay * dense.Weights[i, j])
                            ConvLayers.DenseGrad (decayedW, gradB)
                        | _ -> grad
                    ) currentNetwork gradients

            let newState, updated = ConvLayers.updateNetwork optimizer optimizerState gradientsWithDecay currentNetwork
            optimizerState <- newState
            currentNetwork <- updated

        let avgLoss = epochLoss / float batchCount
        trainLosses <- avgLoss :: trainLosses

        if config.Verbose && (epoch = 1 || epoch % 5 = 0 || epoch = config.Epochs) then
            printfn "Epoch %d/%d - Loss: %.6f" epoch config.Epochs avgLoss

    printfn "CNN training completed!"

    {
        TrainLoss = List.rev trainLosses
        EpochsCompleted = config.Epochs
    }, currentNetwork

let predict (network: ConvLayers.CNNNetwork) (features: float[,]) channels height width =
    let tensor = ConvLayers.matrixToTensor features channels height width
    let output, _ = ConvLayers.forwardNetwork network (ConvLayers.Tensor4D tensor)

    match output with
    | ConvLayers.Matrix matrix -> matrix
    | ConvLayers.Tensor4D _ -> failwith "Final CNN layer must output matrix"

let accuracy (network: ConvLayers.CNNNetwork) (dataset: Data.Dataset) channels height width =
    let predictions = predict network dataset.Features channels height width
    let mutable correct = 0
    let samples = predictions.GetLength(0)

    for i in 0 .. samples - 1 do
        let predClass =
            [| 0 .. (predictions.GetLength(1) - 1) |]
            |> Array.maxBy (fun j -> predictions[i, j])

        let trueClass =
            [| 0 .. (dataset.Labels.GetLength(1) - 1) |]
            |> Array.maxBy (fun j -> dataset.Labels[i, j])

        if predClass = trueClass then
            correct <- correct + 1

    float correct / float samples
