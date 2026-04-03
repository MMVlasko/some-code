// Trainer.fs
module Trainer

open System

type TrainingConfig = {
    Epochs: int
    BatchSize: int
    Optimizer: Optimizers.Optimizer
    Loss: Losses.LossFunction
    Verbose: bool
    OnEpochEnd: (EpochProgress -> unit) option
    ValidationSplit: float option
    WeightDecay: float
    RandomSeed: int option
}

and EpochProgress = {
    Epoch: int
    TotalEpochs: int
    TrainLoss: float
    ValLoss: float option
}

type TrainingMetrics = {
    TrainLoss: float list
    ValLoss: float list
    EpochsCompleted: int
}

let defaultConfig = {
    Epochs = 100
    BatchSize = 32
    Optimizer = Optimizers.SGD 0.01
    Loss = Losses.MSE
    Verbose = true
    OnEpochEnd = None
    ValidationSplit = None
    WeightDecay = 0.0
    RandomSeed = None
}

let train config (network: Layers.NeuralNetwork) (dataset: Data.Dataset) =
    let log message =
        if config.Verbose then
            printfn "%s" message

    log "Starting training..."
    log (sprintf "Network layers: %d" (List.length network))
    log (sprintf "Training samples: %d" (dataset.Features.GetLength(0)))
    log (sprintf "Features per sample: %d" (dataset.Features.GetLength(1)))
    log (sprintf "Classes: %d" (dataset.Labels.GetLength(1)))
    log ""
    
    let trainSet, valSet =
        match config.ValidationSplit with
        | Some ratio -> Data.split ratio dataset
        | None -> 
            let empty = Data.createDataset (Array2D.zeroCreate 0 0) (Array2D.zeroCreate 0 0)
            dataset, empty
    
    let mutable currentNetwork = network
    let mutable optimizerState = Optimizers.initState network config.Optimizer
    let mutable trainLosses = []
    let mutable valLosses = []
    
    for epoch in 1 .. config.Epochs do
        let shuffledTrain =
            match config.RandomSeed with
            | Some seed -> Data.shuffleWithSeed (seed + epoch - 1) trainSet
            | None -> Data.shuffle trainSet
        let batches = Data.batch config.BatchSize shuffledTrain
        
        let mutable epochLoss = 0.0
        let mutable batchCount = 0
        
        for batch in batches do
            batchCount <- batchCount + 1
            
            let rec forwardWithInputs layers currentInput cache =
                match layers with
                | [] -> (currentInput, List.rev cache)
                | layer :: rest ->
                    let z, output = Layers.forwardTraining layer currentInput
                    forwardWithInputs rest output ((currentInput, z, output) :: cache)
            
            let finalOutput, layerCache = forwardWithInputs currentNetwork batch.Features []
            
            let loss = Losses.compute config.Loss finalOutput batch.Labels
            epochLoss <- epochLoss + loss
            
            let lossGrad = Losses.gradient config.Loss finalOutput batch.Labels
            
            let rec backwardPass layers cache grad grads =
                match layers, cache with
                | [], [] ->
                    List.rev grads
                | layer :: restLayers, (input, z, output) :: restCache ->
                    let gradInput, gradWeights, gradBias = 
                        Layers.backward layer grad input z output
                    let newGrads = (gradWeights, gradBias) :: grads
                    backwardPass restLayers restCache gradInput newGrads
                | _ -> failwith "Mismatch between layers and cache"
            
            let reversedNetwork = List.rev currentNetwork
            let reversedCache = List.rev layerCache
            
            let gradientsReversed = backwardPass reversedNetwork reversedCache lossGrad []
            let gradients = List.rev gradientsReversed

            let gradientsWithDecay =
                if config.WeightDecay <= 0.0 then
                    gradients
                else
                    List.map2 (fun (layer: Layers.DenseLayer) ((gradW: float[,]), (gradB: float[])) ->
                        let decayedW =
                            Array2D.init (gradW.GetLength(0)) (gradW.GetLength(1)) (fun i j ->
                                gradW[i, j] + config.WeightDecay * layer.Weights[i, j])
                        decayedW, gradB
                    ) currentNetwork gradients
            
            let newState, updatedNetwork = 
                Optimizers.update config.Optimizer optimizerState gradientsWithDecay currentNetwork
            optimizerState <- newState
            currentNetwork <- updatedNetwork
        
        let avgTrainLoss = epochLoss / float batchCount
        trainLosses <- avgTrainLoss :: trainLosses
        
        let avgValLoss = 
            if (valSet.Features.GetLength(0) > 0) then
                let valPredictions, _ = Layers.forwardNetwork currentNetwork valSet.Features
                Losses.compute config.Loss valPredictions valSet.Labels
            else 0.0
        
        if (valSet.Features.GetLength(0) > 0) then
            valLosses <- avgValLoss :: valLosses
        
        let progress = {
            Epoch = epoch
            TotalEpochs = config.Epochs
            TrainLoss = avgTrainLoss
            ValLoss =
                if valSet.Features.GetLength(0) > 0 then
                    Some avgValLoss
                else
                    None
        }

        config.OnEpochEnd
        |> Option.iter (fun callback -> callback progress)

        if config.Verbose && (epoch % 10 = 0 || epoch = 1) then
            log (sprintf "Epoch %d/%d - Train Loss: %.6f" epoch config.Epochs avgTrainLoss)
            if (valSet.Features.GetLength(0) > 0) then
                log (sprintf "              Val Loss: %.6f" avgValLoss)
    
    log "Training completed!"
    
    let metrics = {
        TrainLoss = List.rev trainLosses
        ValLoss = List.rev valLosses
        EpochsCompleted = config.Epochs
    }
    
    metrics, currentNetwork

let predict (network: Layers.NeuralNetwork) (features: float[,]) =
    let predictions, _ = Layers.forwardNetwork network features
    predictions

let accuracy (network: Layers.NeuralNetwork) (dataset: Data.Dataset) =
    let predictions = predict network dataset.Features
    let mutable correct = 0
    let nSamples = predictions.GetLength(0)
    
    for i in 0 .. nSamples - 1 do
        let predClass = 
            [| 0 .. (predictions.GetLength(1) - 1) |]
            |> Array.maxBy (fun j -> predictions[i, j])
        
        let trueClass = 
            [| 0 .. (dataset.Labels.GetLength(1) - 1) |]
            |> Array.maxBy (fun j -> dataset.Labels[i, j])
        
        if predClass = trueClass then
            correct <- correct + 1
    
    float correct / float nSamples
