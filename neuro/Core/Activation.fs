// Core/Activation.fs
module Activation

type ActivationFunction =
    | Sigmoid
    | ReLU
    | Tanh
    | Softmax
    | Linear

let activate activation value =
    match activation with
    | Sigmoid -> 1.0 / (1.0 + exp(-value))
    | ReLU -> max 0.0 value
    | Tanh -> tanh value
    | Linear -> value
    | Softmax -> failwith "Softmax requires vector input"

let derivative activation value =
    match activation with
    | Sigmoid -> 
        let sigmoidVal = 1.0 / (1.0 + exp(-value))
        sigmoidVal * (1.0 - sigmoidVal)
    | ReLU -> if value > 0.0 then 1.0 else 0.0
    | Tanh -> 
        let tanhVal = tanh value
        1.0 - tanhVal * tanhVal
    | Linear -> 1.0
    | Softmax -> failwith "Softmax derivative is handled in backward pass"

let apply activation (matrix: float[,]) =
    match activation with
    | Softmax ->
        let rows = matrix.GetLength(0)
        let cols = matrix.GetLength(1)
        let result = Array2D.zeroCreate rows cols
        
        for i in 0 .. rows - 1 do
            let row = Array.init cols (fun j -> matrix[i, j])
            let maxVal = Array.max row
            let expVals = row |> Array.map (fun x -> exp(x - maxVal))
            let sumExp = Array.sum expVals
            for j in 0 .. cols - 1 do
                result[i, j] <- expVals[j] / sumExp
        result
    | _ ->
        let rows = matrix.GetLength(0)
        let cols = matrix.GetLength(1)
        Array2D.init rows cols (fun i j -> activate activation matrix[i, j])