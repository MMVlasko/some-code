// Core/Matrix.fs
module Matrix

open System

let zeros rows cols = Array2D.zeroCreate rows cols

let ones rows cols = Array2D.init rows cols (fun _ _ -> 1.0)

let random rows cols =
    let rnd = Random()
    Array2D.init rows cols (fun _ _ -> (rnd.NextDouble() - 0.5) * 0.1)

let multiply (a: float[,]) (b: float[,]) =
    let rowsA = a.GetLength(0)
    let colsA = a.GetLength(1)
    let colsB = b.GetLength(1)
    let result = Array2D.zeroCreate rowsA colsB
    
    for i in 0..rowsA-1 do
        for k in 0..colsA-1 do
            let aik = a[i, k]
            if aik <> 0.0 then
                for j in 0..colsB-1 do
                    result[i, j] <- result[i, j] + aik * b[k, j]
    result

let transpose (matrix: float[,]) =
    let rows = matrix.GetLength(0)
    let cols = matrix.GetLength(1)
    Array2D.init cols rows (fun i j -> matrix[j, i])

let add (a: float[,]) (b: float[,]) =
    let rows = a.GetLength(0)
    let cols = a.GetLength(1)
    Array2D.init rows cols (fun i j -> a[i, j] + b[i, j])

let subtract (a: float[,]) (b: float[,]) =
    let rows = a.GetLength(0)
    let cols = a.GetLength(1)
    Array2D.init rows cols (fun i j -> a[i, j] - b[i, j])

let scale scalar (matrix: float[,]) =
    let rows = matrix.GetLength(0)
    let cols = matrix.GetLength(1)
    Array2D.init rows cols (fun i j -> matrix[i, j] * scalar)

let hadamard (a: float[,]) (b: float[,]) =
    let rows = a.GetLength(0)
    let cols = a.GetLength(1)
    Array2D.init rows cols (fun i j -> a[i, j] * b[i, j])

let sum (matrix: float[,]) =
    let mutable total = 0.0
    for i in 0..matrix.GetLength(0)-1 do
        for j in 0..matrix.GetLength(1)-1 do
            total <- total + matrix[i, j]
    total

let mean (matrix: float[,]) =
    sum matrix / float (matrix.GetLength(0) * matrix.GetLength(1))

let map f (matrix: float[,]) =
    let rows = matrix.GetLength(0)
    let cols = matrix.GetLength(1)
    Array2D.init rows cols (fun i j -> f matrix[i, j])

let init rows cols f = Array2D.init rows cols f

let copy (matrix: float[,]) =
    let rows = matrix.GetLength(0)
    let cols = matrix.GetLength(1)
    Array2D.init rows cols (fun i j -> matrix[i, j])