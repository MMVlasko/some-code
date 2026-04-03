// Layers.fs
module Layers

open System

type LayerKind =
    | Dense
    | Dropout

type DenseLayer = {
    Weights: float[,]
    Bias: float[]
    Activation: Activation.ActivationFunction
    InputSize: int
    OutputSize: int
    Kind: LayerKind
    DropoutRate: float
}

type NeuralNetwork = DenseLayer list

let createLayer inputSize outputSize activation =
    let scale = sqrt(6.0 / float (inputSize + outputSize))
    let rnd = Random()

    let weights =
        Array2D.init outputSize inputSize (fun _ _ ->
            (rnd.NextDouble() * 2.0 - 1.0) * scale)

    let bias = Array.init outputSize (fun _ -> 0.0)

    {
        Weights = weights
        Bias = bias
        Activation = activation
        InputSize = inputSize
        OutputSize = outputSize
        Kind = Dense
        DropoutRate = 0.0
    }

let createDropout size rate =
    if rate < 0.0 || rate >= 1.0 then
        failwith "Dropout rate must be in [0.0, 1.0)"

    let identity = Array2D.init size size (fun i j -> if i = j then 1.0 else 0.0)
    let bias = Array.zeroCreate size

    {
        Weights = identity
        Bias = bias
        Activation = Activation.Linear
        InputSize = size
        OutputSize = size
        Kind = Dropout
        DropoutRate = rate
    }

let forwardWithMode isTraining (layer: DenseLayer) (input: float[,]) =
    match layer.Kind with
    | Dense ->
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

    | Dropout ->
        let rows = input.GetLength(0)
        let cols = input.GetLength(1)

        if cols <> layer.InputSize then
            failwith "Dropout input size mismatch"

        let keepProb = 1.0 - layer.DropoutRate

        if isTraining then
            // Inverted dropout: scale activations during training, keep inference unchanged.
            let rnd = Random()
            let mask =
                Array2D.init rows cols (fun _ _ ->
                    if rnd.NextDouble() < keepProb then 1.0 / keepProb else 0.0)
            let output = Array2D.init rows cols (fun i j -> input[i, j] * mask[i, j])
            (mask, output)
        else
            let mask = Array2D.init rows cols (fun _ _ -> 1.0)
            (mask, input)

let forward (layer: DenseLayer) (input: float[,]) =
    forwardWithMode false layer input

let forwardTraining (layer: DenseLayer) (input: float[,]) =
    forwardWithMode true layer input

let forwardNetwork (network: NeuralNetwork) (input: float[,]) =
    let rec forwardLayer layers currentInput cache =
        match layers with
        | [] -> (currentInput, cache)
        | layer :: rest ->
            let z, output = forwardWithMode false layer currentInput
            forwardLayer rest output ((z, output) :: cache)

    forwardLayer network input []

let backward (layer: DenseLayer) (gradOutput: float[,]) (input: float[,]) (z: float[,]) (output: float[,]) =
    match layer.Kind with
    | Dropout ->
        let rows = gradOutput.GetLength(0)
        let cols = gradOutput.GetLength(1)
        let gradInput = Array2D.init rows cols (fun i j -> gradOutput[i, j] * z[i, j])
        let gradWeights = Array2D.zeroCreate layer.OutputSize layer.InputSize
        let gradBias = Array.zeroCreate layer.OutputSize
        (gradInput, gradWeights, gradBias)

    | Dense ->
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
