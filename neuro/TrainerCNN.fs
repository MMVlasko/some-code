// TrainerCNN.fs
module TrainerCNN

type CNNTrainingConfig = {
    Epochs: int
    BatchSize: int
    LearningRate: float
    Loss: Losses.LossFunction
    Verbose: bool
}

type CNNTrainingMetrics = {
    TrainLoss: float list
    EpochsCompleted: int
}

let defaultConfig = {
    Epochs = 5
    BatchSize = 32
    LearningRate = 0.01
    Loss = Losses.CrossEntropy
    Verbose = true
}

let train config (network: ConvLayers.CNNNetwork) (dataset: Data.Dataset) channels height width =
    printfn "Starting CNN training..."
    printfn "Network layers: %d" (List.length network)
    printfn "Training samples: %d" (dataset.Features.GetLength(0))
    printfn "Input shape: %dx%dx%d" channels height width
    printfn ""

    let mutable currentNetwork = network
    let mutable trainLosses = []

    for epoch in 1 .. config.Epochs do
        let shuffled = Data.shuffle dataset
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

            currentNetwork <- ConvLayers.updateNetworkSGD config.LearningRate currentNetwork gradients

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

