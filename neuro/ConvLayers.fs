// ConvLayers.fs
module ConvLayers

open System

type LayerData =
    | Tensor4D of float[,,,]
    | Matrix of float[,]

type Conv2DLayer = {
    Filters: float[,,,] // [outChannels, inChannels, kernel, kernel]
    Bias: float[]
    InChannels: int
    OutChannels: int
    KernelSize: int
    Stride: int
    Padding: int
    Activation: Activation.ActivationFunction
}

type MaxPool2DLayer = {
    PoolSize: int
    Stride: int
}

type AvgPool2DLayer = {
    PoolSize: int
    Stride: int
}

type FlattenLayer = {
    InputChannels: int
    InputHeight: int
    InputWidth: int
}

type ReshapeToTensorLayer = {
    OutputChannels: int
    OutputHeight: int
    OutputWidth: int
}

type BatchNorm2DLayer = {
    Channels: int
    Gamma: float[]
    Beta: float[]
    Epsilon: float
}

type CNNLayer =
    | Conv2D of Conv2DLayer
    | MaxPool2D of MaxPool2DLayer
    | AvgPool2D of AvgPool2DLayer
    | Flatten of FlattenLayer
    | ReshapeToMatrix of FlattenLayer
    | ReshapeToTensor of ReshapeToTensorLayer
    | BatchNorm2D of BatchNorm2DLayer
    | Dense of Layers.DenseLayer

type CNNNetwork = CNNLayer list

type LayerCache =
    | ConvCache of float[,,,] * float[,,,] * float[,,,]
    | MaxPoolCache of float[,,,] * float[,,,]
    | AvgPoolCache of int * int * int * int
    | FlattenCache of int * int * int
    | ReshapeToTensorCache of int * int
    | BatchNormCache of float[,,,] * float[] * float[] * float[,,,]
    | DenseCache of float[,] * float[,] * float[,]

type LayerGradient =
    | ConvGrad of float[,,,] * float[]
    | DenseGrad of float[,] * float[]
    | BatchNormGrad of float[] * float[]
    | NoGrad

type OptimizerSlot =
    | ConvSlot of float[,,,] * float[]
    | DenseSlot of float[,] * float[]
    | BatchNormSlot of float[] * float[]
    | EmptySlot

type CNNOptimizerState = {
    Velocities: OptimizerSlot list option
    AdamM: OptimizerSlot list option
    AdamV: OptimizerSlot list option
    Step: int
}

let private zeros4DLike (x: float[,,,]) =
    Array4D.zeroCreate (x.GetLength(0)) (x.GetLength(1)) (x.GetLength(2)) (x.GetLength(3))

let private zeros2DLike (x: float[,]) =
    Array2D.zeroCreate (x.GetLength(0)) (x.GetLength(1))

let private map4D (f: float -> float) (x: float[,,,]) =
    Array4D.init (x.GetLength(0)) (x.GetLength(1)) (x.GetLength(2)) (x.GetLength(3)) (fun a b c d -> f x[a, b, c, d])

let private map2D (f: float -> float) (x: float[,]) =
    Array2D.init (x.GetLength(0)) (x.GetLength(1)) (fun i j -> f x[i, j])

let private map2Arrays4D (f: float -> float -> float) (a: float[,,,]) (b: float[,,,]) =
    Array4D.init (a.GetLength(0)) (a.GetLength(1)) (a.GetLength(2)) (a.GetLength(3)) (fun i j k l -> f a[i, j, k, l] b[i, j, k, l])

let private map2Arrays2D (f: float -> float -> float) (a: float[,]) (b: float[,]) =
    Array2D.init (a.GetLength(0)) (a.GetLength(1)) (fun i j -> f a[i, j] b[i, j])

let createConv2D inChannels outChannels kernelSize stride padding activation =
    if activation = Activation.Softmax then
        failwith "Softmax is not supported for Conv2D layers"

    let fanIn = float (inChannels * kernelSize * kernelSize)
    let fanOut = float (outChannels * kernelSize * kernelSize)
    let scale = sqrt (6.0 / (fanIn + fanOut))
    let rnd = Random()

    let filters =
        Array4D.init outChannels inChannels kernelSize kernelSize (fun _ _ _ _ ->
            (rnd.NextDouble() * 2.0 - 1.0) * scale)

    {
        Filters = filters
        Bias = Array.zeroCreate outChannels
        InChannels = inChannels
        OutChannels = outChannels
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
        Activation = activation
    }

let createMaxPool2D poolSize stride : MaxPool2DLayer =
    { PoolSize = poolSize; Stride = stride }

let createAvgPool2D poolSize stride : AvgPool2DLayer =
    { PoolSize = poolSize; Stride = stride }

let createFlatten channels height width =
    {
        InputChannels = channels
        InputHeight = height
        InputWidth = width
    }

let createReshapeToMatrix channels height width =
    createFlatten channels height width

let createReshapeToTensor channels height width =
    {
        OutputChannels = channels
        OutputHeight = height
        OutputWidth = width
    }

let createBatchNorm2D channels =
    {
        Channels = channels
        Gamma = Array.init channels (fun _ -> 1.0)
        Beta = Array.zeroCreate channels
        Epsilon = 1e-5
    }

let matrixToTensor (features: float[,]) channels height width =
    let batch = features.GetLength(0)
    let expectedCols = channels * height * width
    if features.GetLength(1) <> expectedCols then
        failwith $"Expected {expectedCols} features, got {features.GetLength(1)}"

    let tensor = Array4D.zeroCreate batch channels height width
    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let idx = c * height * width + h * width + w
                    tensor[n, c, h, w] <- features[n, idx]
    tensor

let tensorToMatrix (tensor: float[,,,]) =
    let batch = tensor.GetLength(0)
    let channels = tensor.GetLength(1)
    let height = tensor.GetLength(2)
    let width = tensor.GetLength(3)

    let features = channels * height * width
    let matrix = Array2D.zeroCreate batch features

    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let idx = c * height * width + h * width + w
                    matrix[n, idx] <- tensor[n, c, h, w]

    matrix

let private activate4D activation (tensor: float[,,,]) =
    let n = tensor.GetLength(0)
    let c = tensor.GetLength(1)
    let h = tensor.GetLength(2)
    let w = tensor.GetLength(3)

    let output = Array4D.zeroCreate n c h w
    for i in 0 .. n - 1 do
        for j in 0 .. c - 1 do
            for y in 0 .. h - 1 do
                for x in 0 .. w - 1 do
                    output[i, j, y, x] <- Activation.activate activation tensor[i, j, y, x]
    output

let private convForward (layer: Conv2DLayer) (input: float[,,,]) =
    let batch = input.GetLength(0)
    let inChannels = input.GetLength(1)
    let inH = input.GetLength(2)
    let inW = input.GetLength(3)

    if inChannels <> layer.InChannels then
        failwith "Conv2D input channels mismatch"

    let outH = ((inH + 2 * layer.Padding - layer.KernelSize) / layer.Stride) + 1
    let outW = ((inW + 2 * layer.Padding - layer.KernelSize) / layer.Stride) + 1

    let z = Array4D.zeroCreate batch layer.OutChannels outH outW

    for n in 0 .. batch - 1 do
        for oc in 0 .. layer.OutChannels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let mutable sum = layer.Bias[oc]
                    for ic in 0 .. layer.InChannels - 1 do
                        for kh in 0 .. layer.KernelSize - 1 do
                            for kw in 0 .. layer.KernelSize - 1 do
                                let ih = oh * layer.Stride + kh - layer.Padding
                                let iw = ow * layer.Stride + kw - layer.Padding
                                if ih >= 0 && ih < inH && iw >= 0 && iw < inW then
                                    sum <- sum + input[n, ic, ih, iw] * layer.Filters[oc, ic, kh, kw]
                    z[n, oc, oh, ow] <- sum

    let output = activate4D layer.Activation z
    (z, output)

let private maxPoolForward (layer: MaxPool2DLayer) (input: float[,,,]) =
    let batch = input.GetLength(0)
    let channels = input.GetLength(1)
    let inH = input.GetLength(2)
    let inW = input.GetLength(3)

    let outH = ((inH - layer.PoolSize) / layer.Stride) + 1
    let outW = ((inW - layer.PoolSize) / layer.Stride) + 1

    let output = Array4D.zeroCreate batch channels outH outW
    let mask = Array4D.zeroCreate batch channels inH inW

    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let mutable maxVal = Double.NegativeInfinity
                    let mutable maxH = 0
                    let mutable maxW = 0

                    for ph in 0 .. layer.PoolSize - 1 do
                        for pw in 0 .. layer.PoolSize - 1 do
                            let ih = oh * layer.Stride + ph
                            let iw = ow * layer.Stride + pw
                            let v = input[n, c, ih, iw]
                            if v > maxVal then
                                maxVal <- v
                                maxH <- ih
                                maxW <- iw

                    output[n, c, oh, ow] <- maxVal
                    mask[n, c, maxH, maxW] <- 1.0

    (mask, output)

let private avgPoolForward (layer: AvgPool2DLayer) (input: float[,,,]) =
    let batch = input.GetLength(0)
    let channels = input.GetLength(1)
    let inH = input.GetLength(2)
    let inW = input.GetLength(3)

    let outH = ((inH - layer.PoolSize) / layer.Stride) + 1
    let outW = ((inW - layer.PoolSize) / layer.Stride) + 1
    let area = float (layer.PoolSize * layer.PoolSize)

    let output = Array4D.zeroCreate batch channels outH outW

    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let mutable sum = 0.0
                    for ph in 0 .. layer.PoolSize - 1 do
                        for pw in 0 .. layer.PoolSize - 1 do
                            let ih = oh * layer.Stride + ph
                            let iw = ow * layer.Stride + pw
                            sum <- sum + input[n, c, ih, iw]
                    output[n, c, oh, ow] <- sum / area

    output

let private batchNormForward (layer: BatchNorm2DLayer) (input: float[,,,]) =
    let batch = input.GetLength(0)
    let channels = input.GetLength(1)
    let height = input.GetLength(2)
    let width = input.GetLength(3)

    if channels <> layer.Channels then
        failwith "BatchNorm2D channels mismatch"

    let count = float (batch * height * width)
    let mean = Array.zeroCreate channels
    let variance = Array.zeroCreate channels

    for c in 0 .. channels - 1 do
        let mutable s = 0.0
        for n in 0 .. batch - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    s <- s + input[n, c, h, w]
        mean[c] <- s / count

        let mutable sv = 0.0
        for n in 0 .. batch - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let d = input[n, c, h, w] - mean[c]
                    sv <- sv + d * d
        variance[c] <- sv / count

    let normalized = Array4D.zeroCreate batch channels height width
    let output = Array4D.zeroCreate batch channels height width

    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            let invStd = 1.0 / sqrt (variance[c] + layer.Epsilon)
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let xhat = (input[n, c, h, w] - mean[c]) * invStd
                    normalized[n, c, h, w] <- xhat
                    output[n, c, h, w] <- layer.Gamma[c] * xhat + layer.Beta[c]

    (mean, variance, normalized, output)

let forwardLayer (layer: CNNLayer) (input: LayerData) =
    match layer, input with
    | Conv2D conv, Tensor4D tensor ->
        let z, output = convForward conv tensor
        Tensor4D output, ConvCache (tensor, z, output)

    | MaxPool2D pool, Tensor4D tensor ->
        let mask, output = maxPoolForward pool tensor
        Tensor4D output, MaxPoolCache (tensor, mask)

    | AvgPool2D pool, Tensor4D tensor ->
        let output = avgPoolForward pool tensor
        let inH = tensor.GetLength(2)
        let inW = tensor.GetLength(3)
        Tensor4D output, AvgPoolCache (pool.PoolSize, pool.Stride, inH, inW)

    | Flatten flatten, Tensor4D tensor
    | ReshapeToMatrix flatten, Tensor4D tensor ->
        let channels = tensor.GetLength(1)
        let height = tensor.GetLength(2)
        let width = tensor.GetLength(3)
        if channels <> flatten.InputChannels || height <> flatten.InputHeight || width <> flatten.InputWidth then
            failwith "Flatten/ReshapeToMatrix input shape mismatch"

        let matrix = tensorToMatrix tensor
        Matrix matrix, FlattenCache (channels, height, width)

    | ReshapeToTensor reshape, Matrix matrix ->
        let tensor = matrixToTensor matrix reshape.OutputChannels reshape.OutputHeight reshape.OutputWidth
        Matrix matrix |> ignore
        Tensor4D tensor, ReshapeToTensorCache (matrix.GetLength(0), matrix.GetLength(1))

    | BatchNorm2D bn, Tensor4D tensor ->
        let mean, variance, normalized, output = batchNormForward bn tensor
        Tensor4D output, BatchNormCache (tensor, mean, variance, normalized)

    | Dense dense, Matrix matrix ->
        let z, output = Layers.forward dense matrix
        Matrix output, DenseCache (matrix, z, output)

    | _ -> failwith "Layer/data type mismatch"

let forwardNetwork (network: CNNNetwork) (input: LayerData) =
    let rec loop layers current cachesRev =
        match layers with
        | [] -> current, List.rev cachesRev
        | layer :: rest ->
            let output, cache = forwardLayer layer current
            loop rest output (cache :: cachesRev)

    loop network input []

let private convBackward (layer: Conv2DLayer) (input: float[,,,]) (z: float[,,,]) (gradOutput: float[,,,]) =
    let batch = input.GetLength(0)
    let inChannels = input.GetLength(1)
    let inH = input.GetLength(2)
    let inW = input.GetLength(3)

    let outChannels = gradOutput.GetLength(1)
    let outH = gradOutput.GetLength(2)
    let outW = gradOutput.GetLength(3)

    let gradActivation = Array4D.zeroCreate batch outChannels outH outW
    for n in 0 .. batch - 1 do
        for oc in 0 .. outChannels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let deriv = Activation.derivative layer.Activation z[n, oc, oh, ow]
                    gradActivation[n, oc, oh, ow] <- gradOutput[n, oc, oh, ow] * deriv

    let gradFilters = Array4D.zeroCreate layer.OutChannels layer.InChannels layer.KernelSize layer.KernelSize
    let gradBias = Array.zeroCreate layer.OutChannels
    let gradInput = Array4D.zeroCreate batch inChannels inH inW

    for n in 0 .. batch - 1 do
        for oc in 0 .. layer.OutChannels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let ga = gradActivation[n, oc, oh, ow]
                    gradBias[oc] <- gradBias[oc] + ga

                    for ic in 0 .. layer.InChannels - 1 do
                        for kh in 0 .. layer.KernelSize - 1 do
                            for kw in 0 .. layer.KernelSize - 1 do
                                let ih = oh * layer.Stride + kh - layer.Padding
                                let iw = ow * layer.Stride + kw - layer.Padding
                                if ih >= 0 && ih < inH && iw >= 0 && iw < inW then
                                    gradFilters[oc, ic, kh, kw] <- gradFilters[oc, ic, kh, kw] + ga * input[n, ic, ih, iw]
                                    gradInput[n, ic, ih, iw] <- gradInput[n, ic, ih, iw] + ga * layer.Filters[oc, ic, kh, kw]

    let batchF = float batch
    for oc in 0 .. layer.OutChannels - 1 do
        gradBias[oc] <- gradBias[oc] / batchF
        for ic in 0 .. layer.InChannels - 1 do
            for kh in 0 .. layer.KernelSize - 1 do
                for kw in 0 .. layer.KernelSize - 1 do
                    gradFilters[oc, ic, kh, kw] <- gradFilters[oc, ic, kh, kw] / batchF

    gradInput, ConvGrad (gradFilters, gradBias)

let private maxPoolBackward (layer: MaxPool2DLayer) (input: float[,,,]) (mask: float[,,,]) (gradOutput: float[,,,]) =
    let batch = input.GetLength(0)
    let channels = input.GetLength(1)
    let inH = input.GetLength(2)
    let inW = input.GetLength(3)

    let outH = gradOutput.GetLength(2)
    let outW = gradOutput.GetLength(3)

    let gradInput = Array4D.zeroCreate batch channels inH inW

    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let g = gradOutput[n, c, oh, ow]
                    for ph in 0 .. layer.PoolSize - 1 do
                        for pw in 0 .. layer.PoolSize - 1 do
                            let ih = oh * layer.Stride + ph
                            let iw = ow * layer.Stride + pw
                            if mask[n, c, ih, iw] > 0.0 then
                                gradInput[n, c, ih, iw] <- gradInput[n, c, ih, iw] + g

    gradInput

let private avgPoolBackward (poolSize: int) (stride: int) (inH: int) (inW: int) (gradOutput: float[,,,]) =
    let batch = gradOutput.GetLength(0)
    let channels = gradOutput.GetLength(1)
    let outH = gradOutput.GetLength(2)
    let outW = gradOutput.GetLength(3)
    let area = float (poolSize * poolSize)

    let gradInput = Array4D.zeroCreate batch channels inH inW

    for n in 0 .. batch - 1 do
        for c in 0 .. channels - 1 do
            for oh in 0 .. outH - 1 do
                for ow in 0 .. outW - 1 do
                    let g = gradOutput[n, c, oh, ow] / area
                    for ph in 0 .. poolSize - 1 do
                        for pw in 0 .. poolSize - 1 do
                            let ih = oh * stride + ph
                            let iw = ow * stride + pw
                            gradInput[n, c, ih, iw] <- gradInput[n, c, ih, iw] + g

    gradInput

let private batchNormBackward (layer: BatchNorm2DLayer) (input: float[,,,]) (mean: float[]) (variance: float[]) (normalized: float[,,,]) (gradOutput: float[,,,]) =
    let batch = input.GetLength(0)
    let channels = input.GetLength(1)
    let height = input.GetLength(2)
    let width = input.GetLength(3)
    let m = float (batch * height * width)

    let gradInput = Array4D.zeroCreate batch channels height width
    let gradGamma = Array.zeroCreate channels
    let gradBeta = Array.zeroCreate channels

    for c in 0 .. channels - 1 do
        let mutable sumDy = 0.0
        let mutable sumDyXhat = 0.0

        for n in 0 .. batch - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let dy = gradOutput[n, c, h, w]
                    sumDy <- sumDy + dy
                    sumDyXhat <- sumDyXhat + dy * normalized[n, c, h, w]

        gradBeta[c] <- sumDy / m
        gradGamma[c] <- sumDyXhat / m

        let invStd = 1.0 / sqrt (variance[c] + layer.Epsilon)
        for n in 0 .. batch - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let dy = gradOutput[n, c, h, w]
                    let xhat = normalized[n, c, h, w]
                    let term = m * dy - sumDy - xhat * sumDyXhat
                    gradInput[n, c, h, w] <- (layer.Gamma[c] * invStd / m) * term

    gradInput, BatchNormGrad (gradGamma, gradBeta)

let backwardLayer (layer: CNNLayer) (cache: LayerCache) (gradOutput: LayerData) =
    match layer, cache, gradOutput with
    | Conv2D conv, ConvCache (input, z, _), Tensor4D grad ->
        let gradInput, gradParams = convBackward conv input z grad
        Tensor4D gradInput, gradParams

    | MaxPool2D pool, MaxPoolCache (input, mask), Tensor4D grad ->
        let gradInput = maxPoolBackward pool input mask grad
        Tensor4D gradInput, NoGrad

    | AvgPool2D _, AvgPoolCache (poolSize, stride, inH, inW), Tensor4D grad ->
        let gradInput = avgPoolBackward poolSize stride inH inW grad
        Tensor4D gradInput, NoGrad

    | Flatten flatten, FlattenCache (channels, height, width), Matrix grad
    | ReshapeToMatrix flatten, FlattenCache (channels, height, width), Matrix grad ->
        if channels <> flatten.InputChannels || height <> flatten.InputHeight || width <> flatten.InputWidth then
            failwith "Flatten/ReshapeToMatrix cache mismatch"

        let batch = grad.GetLength(0)
        let expected = channels * height * width
        if grad.GetLength(1) <> expected then
            failwith "Flatten backward gradient shape mismatch"

        let gradInput = Array4D.zeroCreate batch channels height width
        for n in 0 .. batch - 1 do
            for c in 0 .. channels - 1 do
                for h in 0 .. height - 1 do
                    for w in 0 .. width - 1 do
                        let idx = c * height * width + h * width + w
                        gradInput[n, c, h, w] <- grad[n, idx]

        Tensor4D gradInput, NoGrad

    | ReshapeToTensor reshape, ReshapeToTensorCache (_, _), Tensor4D grad ->
        let matrix = tensorToMatrix grad
        let expected = reshape.OutputChannels * reshape.OutputHeight * reshape.OutputWidth
        if matrix.GetLength(1) <> expected then
            failwith "ReshapeToTensor backward shape mismatch"
        Matrix matrix, NoGrad

    | BatchNorm2D bn, BatchNormCache (input, mean, variance, normalized), Tensor4D grad ->
        let gradInput, gradParams = batchNormBackward bn input mean variance normalized grad
        Tensor4D gradInput, gradParams

    | Dense dense, DenseCache (input, z, output), Matrix grad ->
        let gradInput, gradW, gradB = Layers.backward dense grad input z output
        Matrix gradInput, DenseGrad (gradW, gradB)

    | _ -> failwith "Backward layer/cache/gradient mismatch"

let backwardNetwork (network: CNNNetwork) (caches: LayerCache list) (initialGrad: LayerData) =
    let rec loop revLayers revCaches grad gradientsAcc =
        match revLayers, revCaches with
        | [], [] -> grad, gradientsAcc
        | layer :: restLayers, cache :: restCaches ->
            let gradInput, layerGrad = backwardLayer layer cache grad
            loop restLayers restCaches gradInput (layerGrad :: gradientsAcc)
        | _ -> failwith "Mismatch between network layers and caches"

    loop (List.rev network) (List.rev caches) initialGrad []

let private buildEmptySlot layer =
    match layer with
    | Conv2D conv -> ConvSlot (zeros4DLike conv.Filters, Array.zeroCreate conv.Bias.Length)
    | Dense dense -> DenseSlot (zeros2DLike dense.Weights, Array.zeroCreate dense.Bias.Length)
    | BatchNorm2D bn -> BatchNormSlot (Array.zeroCreate bn.Gamma.Length, Array.zeroCreate bn.Beta.Length)
    | _ -> EmptySlot

let initOptimizerState (network: CNNNetwork) (optimizer: Optimizers.Optimizer) =
    let slots = network |> List.map buildEmptySlot

    match optimizer with
    | Optimizers.SGD _
    | Optimizers.GradientClipping _ ->
        { Velocities = None; AdamM = None; AdamV = None; Step = 0 }

    | Optimizers.Momentum _ ->
        { Velocities = Some slots; AdamM = None; AdamV = None; Step = 0 }

    | Optimizers.Adam _ ->
        { Velocities = None; AdamM = Some slots; AdamV = Some slots; Step = 1 }

let private clipGradients threshold grad =
    let clip x =
        if abs x > threshold then
            (if x > 0.0 then threshold else -threshold)
        else x
    match grad with
    | ConvGrad (gw, gb) -> ConvGrad (map4D clip gw, gb |> Array.map clip)
    | DenseGrad (gw, gb) -> DenseGrad (map2D clip gw, gb |> Array.map clip)
    | BatchNormGrad (gg, gb) -> BatchNormGrad (gg |> Array.map clip, gb |> Array.map clip)
    | NoGrad -> NoGrad

let private updateBySGD lr layer grad =
    match layer, grad with
    | Conv2D conv, ConvGrad (gradFilters, gradBias) ->
        let newFilters = map2Arrays4D (fun w g -> w - lr * g) conv.Filters gradFilters
        let newBias = Array.init conv.Bias.Length (fun i -> conv.Bias[i] - lr * gradBias[i])
        Conv2D { conv with Filters = newFilters; Bias = newBias }

    | Dense dense, DenseGrad (gradW, gradB) ->
        let newWeights = map2Arrays2D (fun w g -> w - lr * g) dense.Weights gradW
        let newBias = Array.init dense.Bias.Length (fun i -> dense.Bias[i] - lr * gradB[i])
        Dense { dense with Weights = newWeights; Bias = newBias }

    | BatchNorm2D bn, BatchNormGrad (gradGamma, gradBeta) ->
        let newGamma = Array.init bn.Gamma.Length (fun i -> bn.Gamma[i] - lr * gradGamma[i])
        let newBeta = Array.init bn.Beta.Length (fun i -> bn.Beta[i] - lr * gradBeta[i])
        BatchNorm2D { bn with Gamma = newGamma; Beta = newBeta }

    | _, NoGrad -> layer
    | _ -> failwith "Gradient does not match layer type"

let updateNetworkSGD learningRate (network: CNNNetwork) (gradients: LayerGradient list) =
    List.map2 (fun layer grad -> updateBySGD learningRate layer grad) network gradients

let updateNetwork (optimizer: Optimizers.Optimizer) (state: CNNOptimizerState) (gradients: LayerGradient list) (network: CNNNetwork) =
    match optimizer with
    | Optimizers.SGD lr ->
        let updated = List.map2 (fun layer grad -> updateBySGD lr layer grad) network gradients
        state, updated

    | Optimizers.GradientClipping (lr, threshold) ->
        let clipped = gradients |> List.map (clipGradients threshold)
        let updated = List.map2 (fun layer grad -> updateBySGD lr layer grad) network clipped
        state, updated

    | Optimizers.Momentum (lr, momentum) ->
        match state.Velocities with
        | None -> failwith "Invalid CNN optimizer state for Momentum"
        | Some velocities ->
            let newVelocities, updatedLayers =
                List.map3 (fun layer grad vel ->
                    match layer, grad, vel with
                    | Conv2D conv, ConvGrad (gw, gb), ConvSlot (vw, vb) ->
                        let newVw = map2Arrays4D (fun v g -> momentum * v + lr * g) vw gw
                        let newVb = Array.init vb.Length (fun i -> momentum * vb[i] + lr * gb[i])
                        let newW = map2Arrays4D (fun w v -> w - v) conv.Filters newVw
                        let newB = Array.init conv.Bias.Length (fun i -> conv.Bias[i] - newVb[i])
                        ConvSlot (newVw, newVb), Conv2D { conv with Filters = newW; Bias = newB }

                    | Dense dense, DenseGrad (gw, gb), DenseSlot (vw, vb) ->
                        let newVw = map2Arrays2D (fun v g -> momentum * v + lr * g) vw gw
                        let newVb = Array.init vb.Length (fun i -> momentum * vb[i] + lr * gb[i])
                        let newW = map2Arrays2D (fun w v -> w - v) dense.Weights newVw
                        let newB = Array.init dense.Bias.Length (fun i -> dense.Bias[i] - newVb[i])
                        DenseSlot (newVw, newVb), Dense { dense with Weights = newW; Bias = newB }

                    | BatchNorm2D bn, BatchNormGrad (gg, gb), BatchNormSlot (vg, vb) ->
                        let newVg = Array.init vg.Length (fun i -> momentum * vg[i] + lr * gg[i])
                        let newVb = Array.init vb.Length (fun i -> momentum * vb[i] + lr * gb[i])
                        let newGamma = Array.init bn.Gamma.Length (fun i -> bn.Gamma[i] - newVg[i])
                        let newBeta = Array.init bn.Beta.Length (fun i -> bn.Beta[i] - newVb[i])
                        BatchNormSlot (newVg, newVb), BatchNorm2D { bn with Gamma = newGamma; Beta = newBeta }

                    | _, NoGrad, _ -> vel, layer
                    | _ -> failwith "Momentum state/gradient mismatch"
                ) network gradients velocities
                |> List.unzip

            { state with Velocities = Some newVelocities }, updatedLayers

    | Optimizers.Adam (lr, beta1, beta2, eps) ->
        match state.AdamM, state.AdamV with
        | Some adamM, Some adamV ->
            let t = float state.Step
            let b1t = Math.Pow(beta1, t)
            let b2t = Math.Pow(beta2, t)

            let updatedM, updatedV, updatedLayers =
                List.map3 (fun layer grad (mSlot, vSlot) ->
                    match layer, grad, mSlot, vSlot with
                    | Conv2D conv, ConvGrad (gw, gb), ConvSlot (mw, mb), ConvSlot (vw, vb) ->
                        let newMw = map2Arrays4D (fun m g -> beta1 * m + (1.0 - beta1) * g) mw gw
                        let newVw = map2Arrays4D (fun v g -> beta2 * v + (1.0 - beta2) * g * g) vw gw
                        let newMb = Array.init mb.Length (fun i -> beta1 * mb[i] + (1.0 - beta1) * gb[i])
                        let newVb = Array.init vb.Length (fun i -> beta2 * vb[i] + (1.0 - beta2) * gb[i] * gb[i])

                        let newW =
                            Array4D.init (conv.Filters.GetLength(0)) (conv.Filters.GetLength(1)) (conv.Filters.GetLength(2)) (conv.Filters.GetLength(3)) (fun i j k l ->
                                let mHat = newMw[i, j, k, l] / (1.0 - b1t)
                                let vHat = newVw[i, j, k, l] / (1.0 - b2t)
                                conv.Filters[i, j, k, l] - lr * mHat / (sqrt vHat + eps))

                        let newB =
                            Array.init conv.Bias.Length (fun i ->
                                let mHat = newMb[i] / (1.0 - b1t)
                                let vHat = newVb[i] / (1.0 - b2t)
                                conv.Bias[i] - lr * mHat / (sqrt vHat + eps))

                        ConvSlot (newMw, newMb), ConvSlot (newVw, newVb), Conv2D { conv with Filters = newW; Bias = newB }

                    | Dense dense, DenseGrad (gw, gb), DenseSlot (mw, mb), DenseSlot (vw, vb) ->
                        let newMw = map2Arrays2D (fun m g -> beta1 * m + (1.0 - beta1) * g) mw gw
                        let newVw = map2Arrays2D (fun v g -> beta2 * v + (1.0 - beta2) * g * g) vw gw
                        let newMb = Array.init mb.Length (fun i -> beta1 * mb[i] + (1.0 - beta1) * gb[i])
                        let newVb = Array.init vb.Length (fun i -> beta2 * vb[i] + (1.0 - beta2) * gb[i] * gb[i])

                        let newW =
                            Array2D.init (dense.Weights.GetLength(0)) (dense.Weights.GetLength(1)) (fun i j ->
                                let mHat = newMw[i, j] / (1.0 - b1t)
                                let vHat = newVw[i, j] / (1.0 - b2t)
                                dense.Weights[i, j] - lr * mHat / (sqrt vHat + eps))

                        let newB =
                            Array.init dense.Bias.Length (fun i ->
                                let mHat = newMb[i] / (1.0 - b1t)
                                let vHat = newVb[i] / (1.0 - b2t)
                                dense.Bias[i] - lr * mHat / (sqrt vHat + eps))

                        DenseSlot (newMw, newMb), DenseSlot (newVw, newVb), Dense { dense with Weights = newW; Bias = newB }

                    | BatchNorm2D bn, BatchNormGrad (gg, gb), BatchNormSlot (mg, mb), BatchNormSlot (vg, vb) ->
                        let newMg = Array.init mg.Length (fun i -> beta1 * mg[i] + (1.0 - beta1) * gg[i])
                        let newVg = Array.init vg.Length (fun i -> beta2 * vg[i] + (1.0 - beta2) * gg[i] * gg[i])
                        let newMb = Array.init mb.Length (fun i -> beta1 * mb[i] + (1.0 - beta1) * gb[i])
                        let newVb = Array.init vb.Length (fun i -> beta2 * vb[i] + (1.0 - beta2) * gb[i] * gb[i])

                        let newGamma =
                            Array.init bn.Gamma.Length (fun i ->
                                let mHat = newMg[i] / (1.0 - b1t)
                                let vHat = newVg[i] / (1.0 - b2t)
                                bn.Gamma[i] - lr * mHat / (sqrt vHat + eps))

                        let newBeta =
                            Array.init bn.Beta.Length (fun i ->
                                let mHat = newMb[i] / (1.0 - b1t)
                                let vHat = newVb[i] / (1.0 - b2t)
                                bn.Beta[i] - lr * mHat / (sqrt vHat + eps))

                        BatchNormSlot (newMg, newMb), BatchNormSlot (newVg, newVb), BatchNorm2D { bn with Gamma = newGamma; Beta = newBeta }

                    | _, NoGrad, _, _ -> mSlot, vSlot, layer
                    | _ -> failwith "Adam state/gradient mismatch"
                ) network gradients (List.zip adamM adamV)
                |> List.unzip3

            {
                state with
                    AdamM = Some updatedM
                    AdamV = Some updatedV
                    Step = state.Step + 1
            }, updatedLayers

        | _ -> failwith "Invalid CNN optimizer state for Adam"



