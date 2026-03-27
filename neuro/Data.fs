// Data.fs
module Data

open System

type Dataset = {
    Features: float[,]
    Labels: float[,]
}

let createDataset features labels = 
    { Features = features; Labels = labels }

let map f (dataset: Dataset) =
    let rows = dataset.Features.GetLength(0)
    let cols = dataset.Features.GetLength(1)
    let newFeatures = Array2D.init rows cols (fun i j -> f dataset.Features[i, j])
    { dataset with Features = newFeatures }

let normalize (dataset: Dataset) =
    let rows = dataset.Features.GetLength(0)
    let cols = dataset.Features.GetLength(1)
    
    let mean = Array.zeroCreate cols
    for i in 0 .. rows - 1 do
        for j in 0 .. cols - 1 do
            mean[j] <- mean[j] + dataset.Features[i, j]
    
    for j in 0 .. cols - 1 do
        mean[j] <- mean[j] / float rows
    
    let std = Array.zeroCreate cols
    for i in 0 .. rows - 1 do
        for j in 0 .. cols - 1 do
            let diff = dataset.Features[i, j] - mean[j]
            std[j] <- std[j] + diff * diff
    
    for j in 0 .. cols - 1 do
        std[j] <- sqrt(std[j] / float rows)
    
    let normalized = 
        Array2D.init rows cols (fun i j -> 
            if std[j] > 0.0 then
                (dataset.Features[i, j] - mean[j]) / std[j]
            else 0.0)
    
    { dataset with Features = normalized }

let shuffleWithRandom (rnd: Random) (dataset: Dataset) =
    let rows = dataset.Features.GetLength(0)
    let indices = Array.init rows id
    
    for i in rows - 1 .. -1 .. 1 do
        let j = rnd.Next(i + 1)
        let temp = indices[i]
        indices[i] <- indices[j]
        indices[j] <- temp

    if rows > 1 && (indices |> Array.forall2 (=) (Array.init rows id)) then
        let temp = indices[0]
        indices[0] <- indices[1]
        indices[1] <- temp
    
    let colsFeatures = dataset.Features.GetLength(1)
    let shuffledFeatures = 
        Array2D.init rows colsFeatures (fun i j -> 
            dataset.Features[indices[i], j])
    
    let colsLabels = dataset.Labels.GetLength(1)
    let shuffledLabels = 
        Array2D.init rows colsLabels (fun i j -> 
            dataset.Labels[indices[i], j])
    
    { Features = shuffledFeatures; Labels = shuffledLabels }

let shuffle (dataset: Dataset) =
    shuffleWithRandom (Random()) dataset

let shuffleWithSeed seed (dataset: Dataset) =
    shuffleWithRandom (Random(seed)) dataset

let batch batchSize (dataset: Dataset) =
    let nSamples = dataset.Features.GetLength(0)
    let nFeatures = dataset.Features.GetLength(1)
    let nLabels = dataset.Labels.GetLength(1)
    let nBatches = (nSamples + batchSize - 1) / batchSize
    
    [ for b in 0 .. nBatches - 1 do
        let start = b * batchSize
        let endIdx = min (start + batchSize) nSamples
        let currentBatchSize = endIdx - start
        
        let batchFeatures = 
            Array2D.init currentBatchSize nFeatures (fun i j -> 
                dataset.Features[start + i, j])
        
        let batchLabels = 
            Array2D.init currentBatchSize nLabels (fun i j -> 
                dataset.Labels[start + i, j])
        
        yield { Features = batchFeatures; Labels = batchLabels } ]

let split ratio (dataset: Dataset) =
    let nSamples = dataset.Features.GetLength(0)
    let nTrain = int (float nSamples * ratio)
    let nVal = nSamples - nTrain
    
    let trainFeatures = 
        Array2D.init nTrain (dataset.Features.GetLength(1)) (fun i j -> 
            dataset.Features[i, j])
    
    let trainLabels = 
        Array2D.init nTrain (dataset.Labels.GetLength(1)) (fun i j -> 
            dataset.Labels[i, j])
    
    let valFeatures = 
        Array2D.init nVal (dataset.Features.GetLength(1)) (fun i j -> 
            dataset.Features[nTrain + i, j])
    
    let valLabels = 
        Array2D.init nVal (dataset.Labels.GetLength(1)) (fun i j -> 
            dataset.Labels[nTrain + i, j])
    
    (createDataset trainFeatures trainLabels, 
     createDataset valFeatures valLabels)
