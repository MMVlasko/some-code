// Layers.fs
module Layers

open System

type DenseLayer = {
    Weights: float[,]
    Bias: float[]
    Activation: Activation.ActivationFunction
    InputSize: int
    OutputSize: int
}

type NeuralNetwork = DenseLayer list

let createLayer inputSize outputSize activation =
    let scale = sqrt(6.0 / float (inputSize + outputSize))
    let rnd = Random()
    
    let weights = Array2D.init outputSize inputSize (fun _ _ -> 
        (rnd.NextDouble() * 2.0 - 1.0) * scale)
    
    let bias = Array.init outputSize (fun _ -> 0.0)
    
    {
        Weights = weights
        Bias = bias
        Activation = activation
        InputSize = inputSize
        OutputSize = outputSize
    }

let forward (layer: DenseLayer) (input: float[,]) =
    let batchSize = input.GetLength(0)
    let inputSize = layer.InputSize
    let outputSize = layer.OutputSize
    
    let z = Array2D.zeroCreate batchSize outputSize
    
    for i in 0 .. batchSize - 1 do
        for j in 0 .. outputSize - 1 do
            let mutable sum = layer.Bias[j]
            for k in 0 .. inputSize - 1 do
                sum <- sum + input[i, k] * layer.Weights[j, k]
            z[i, j] <- sum
    
    let output = Activation.apply layer.Activation z
    (z, output)

let forwardNetwork (network: NeuralNetwork) (input: float[,]) =
    let rec forwardLayer layers currentInput cache =
        match layers with
        | [] -> (currentInput, cache)
        | layer :: rest ->
            let z, output = forward layer currentInput
            forwardLayer rest output ((z, output) :: cache)
    
    forwardLayer network input []

let backward (layer: DenseLayer) (gradOutput: float[,]) (input: float[,]) (z: float[,]) (output: float[,]) =
    let batchSize = gradOutput.GetLength(0)
    let outputSize = layer.OutputSize
    let inputSize = layer.InputSize
    
    let gradActivation = Array2D.zeroCreate batchSize outputSize
    
    match layer.Activation with
    | Activation.Softmax ->
        for i in 0 .. batchSize - 1 do
            for j in 0 .. outputSize - 1 do
                gradActivation[i, j] <- gradOutput[i, j]
    | _ ->
        for i in 0 .. batchSize - 1 do
            for j in 0 .. outputSize - 1 do
                let deriv = Activation.derivative layer.Activation z[i, j]
                gradActivation[i, j] <- gradOutput[i, j] * deriv
    
    let gradWeights = Array2D.zeroCreate outputSize inputSize
    for i in 0 .. batchSize - 1 do
        for j in 0 .. outputSize - 1 do
            let ga = gradActivation[i, j]
            if ga <> 0.0 then
                for k in 0 .. inputSize - 1 do
                    gradWeights[j, k] <- gradWeights[j, k] + ga * input[i, k]
    
    let batchSizeFloat = float batchSize
    for j in 0 .. outputSize - 1 do
        for k in 0 .. inputSize - 1 do
            gradWeights[j, k] <- gradWeights[j, k] / batchSizeFloat
    
    let gradBias = Array.zeroCreate outputSize
    for i in 0 .. batchSize - 1 do
        for j in 0 .. outputSize - 1 do
            gradBias[j] <- gradBias[j] + gradActivation[i, j]
    
    for j in 0 .. outputSize - 1 do
        gradBias[j] <- gradBias[j] / batchSizeFloat
    
    let gradInput = Array2D.zeroCreate batchSize inputSize
    for i in 0 .. batchSize - 1 do
        for j in 0 .. inputSize - 1 do
            let mutable sum = 0.0
            for k in 0 .. outputSize - 1 do
                sum <- sum + gradActivation[i, k] * layer.Weights[k, j]
            gradInput[i, j] <- sum
    
    (gradInput, gradWeights, gradBias)