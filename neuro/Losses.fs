// Losses.fs
module Losses

type LossFunction =
    | MSE
    | CrossEntropy
    | BinaryCrossEntropy

let compute loss (yPred: float[,]) (yTrue: float[,]) =
    let rows = yPred.GetLength(0)
    let cols = yPred.GetLength(1)
    
    match loss with
    | MSE ->
        let mutable sumError = 0.0
        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                let diff = yPred[i, j] - yTrue[i, j]
                sumError <- sumError + diff * diff
        sumError / float (rows * cols)
    
    | CrossEntropy ->
        let eps = 1e-8
        let mutable total = 0.0
        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                let pred = max eps (min (1.0 - eps) yPred[i, j])
                total <- total - yTrue[i, j] * log(pred)
        total / float rows
    
    | BinaryCrossEntropy ->
        let eps = 1e-8
        let mutable total = 0.0
        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                let pred = max eps (min (1.0 - eps) yPred[i, j])
                total <- total - (yTrue[i, j] * log(pred) + (1.0 - yTrue[i, j]) * log(1.0 - pred))
        total / float rows

let gradient loss (yPred: float[,]) (yTrue: float[,]) =
    let rows = yPred.GetLength(0)
    let cols = yPred.GetLength(1)
    let result = Array2D.zeroCreate rows cols
    
    match loss with
    | MSE ->
        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                result[i, j] <- (yPred[i, j] - yTrue[i, j]) * 2.0
    
    | CrossEntropy ->
        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                result[i, j] <- yPred[i, j] - yTrue[i, j]
    
    | BinaryCrossEntropy ->
        let eps = 1e-8
        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                let pred = max eps (min (1.0 - eps) yPred[i, j])
                result[i, j] <- (pred - yTrue[i, j]) / (pred * (1.0 - pred) + eps)
    
    result