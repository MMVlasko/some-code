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

type FlattenLayer = {
    InputChannels: int
    InputHeight: int
    InputWidth: int
}

type CNNLayer =
    | Conv2D of Conv2DLayer
    | MaxPool2D of MaxPool2DLayer
    | Flatten of FlattenLayer
    | Dense of Layers.DenseLayer

type CNNNetwork = CNNLayer list

type LayerCache =
    | ConvCache of float[,,,] * float[,,,] * float[,,,]
    | PoolCache of float[,,,] * float[,,,]
    | FlattenCache of int * int * int
    | DenseCache of float[,] * float[,] * float[,]

type LayerGradient =
    | ConvGrad of float[,,,] * float[]
    | DenseGrad of float[,] * float[]
    | NoGrad

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

let createMaxPool2D poolSize stride =
    { PoolSize = poolSize; Stride = stride }

let createFlatten channels height width =
    {
        InputChannels = channels
        InputHeight = height
        InputWidth = width
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

let private poolForward (layer: MaxPool2DLayer) (input: float[,,,]) =
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

let private flattenForward (input: float[,,,]) =
    tensorToMatrix input

let forwardLayer (layer: CNNLayer) (input: LayerData) =
    match layer, input with
    | Conv2D conv, Tensor4D tensor ->
        let z, output = convForward conv tensor
        Tensor4D output, ConvCache (tensor, z, output)

    | MaxPool2D pool, Tensor4D tensor ->
        let mask, output = poolForward pool tensor
        Tensor4D output, PoolCache (tensor, mask)

    | Flatten flatten, Tensor4D tensor ->
        let channels = tensor.GetLength(1)
        let height = tensor.GetLength(2)
        let width = tensor.GetLength(3)
        if channels <> flatten.InputChannels || height <> flatten.InputHeight || width <> flatten.InputWidth then
            failwith "Flatten input shape mismatch"
        let matrix = flattenForward tensor
        Matrix matrix, FlattenCache (channels, height, width)

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

let private poolBackward (layer: MaxPool2DLayer) (input: float[,,,]) (mask: float[,,,]) (gradOutput: float[,,,]) =
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

let backwardLayer (layer: CNNLayer) (cache: LayerCache) (gradOutput: LayerData) =
    match layer, cache, gradOutput with
    | Conv2D conv, ConvCache (input, z, _), Tensor4D grad ->
        let gradInput, gradParams = convBackward conv input z grad
        Tensor4D gradInput, gradParams

    | MaxPool2D pool, PoolCache (input, mask), Tensor4D grad ->
        let gradInput = poolBackward pool input mask grad
        Tensor4D gradInput, NoGrad

    | Flatten flatten, FlattenCache (channels, height, width), Matrix grad ->
        if channels <> flatten.InputChannels || height <> flatten.InputHeight || width <> flatten.InputWidth then
            failwith "Flatten cache mismatch"

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

let updateNetworkSGD learningRate (network: CNNNetwork) (gradients: LayerGradient list) =
    let updateConv (layer: Conv2DLayer) (gradFilters: float[,,,]) (gradBias: float[]) =
        let newFilters =
            Array4D.init layer.OutChannels layer.InChannels layer.KernelSize layer.KernelSize (fun oc ic kh kw ->
                layer.Filters[oc, ic, kh, kw] - learningRate * gradFilters[oc, ic, kh, kw])

        let newBias =
            Array.init layer.OutChannels (fun i -> layer.Bias[i] - learningRate * gradBias[i])

        { layer with Filters = newFilters; Bias = newBias }

    let updateDense (layer: Layers.DenseLayer) (gradW: float[,]) (gradB: float[]) =
        let rows = layer.Weights.GetLength(0)
        let cols = layer.Weights.GetLength(1)

        let newWeights =
            Array2D.init rows cols (fun i j -> layer.Weights[i, j] - learningRate * gradW[i, j])

        let newBias =
            Array.init layer.Bias.Length (fun i -> layer.Bias[i] - learningRate * gradB[i])

        { layer with Weights = newWeights; Bias = newBias }

    List.map2 (fun layer grad ->
        match layer, grad with
        | Conv2D conv, ConvGrad (gradFilters, gradBias) -> Conv2D (updateConv conv gradFilters gradBias)
        | Dense dense, DenseGrad (gradW, gradB) -> Dense (updateDense dense gradW gradB)
        | _, NoGrad -> layer
        | _ -> failwith "Gradient does not match layer type"
    ) network gradients

