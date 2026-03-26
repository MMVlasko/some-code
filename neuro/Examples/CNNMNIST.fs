// Examples/CNNMNIST.fs
namespace Examples

open System.IO

module CNNMNIST =

    let private buildBinaryDataset (dataset: Data.Dataset) =
        let rows = dataset.Features.GetLength(0)
        let cols = dataset.Features.GetLength(1)

        let selectedIndices =
            [|
                for i in 0 .. rows - 1 do
                    let label = [|0..9|] |> Array.maxBy (fun j -> dataset.Labels[i, j])
                    if label = 0 || label = 1 then
                        yield i
            |]

        let binaryFeatures =
            Array2D.init selectedIndices.Length cols (fun i j ->
                dataset.Features[selectedIndices[i], j])

        let binaryLabels = Array2D.zeroCreate selectedIndices.Length 2
        for i in 0 .. selectedIndices.Length - 1 do
            let originalIdx = selectedIndices[i]
            let isOne = dataset.Labels[originalIdx, 1] = 1.0
            if isOne then binaryLabels[i, 1] <- 1.0 else binaryLabels[i, 0] <- 1.0

        Data.createDataset binaryFeatures binaryLabels

    let run () =
        printfn "\n========================================"
        printfn "MNIST CNN (Conv2D + Pool + Dense)"
        printfn "========================================\n"

        let trainFilePath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_train.csv")
        let testFilePath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_test.csv")

        if not (File.Exists(trainFilePath)) then
            printfn "ERROR: MNIST training data not found"
            printfn $"  {trainFilePath}"
            ()
        elif not (File.Exists(testFilePath)) then
            printfn "ERROR: MNIST test data not found"
            printfn $"  {testFilePath}"
            ()
        else
            printfn "=== LOADING DATA ==="
            // Naive Conv2D implementation is intentionally compact, so keep sample size moderate.
            let rawTrain = MNIST.loadMnistFromCsv trainFilePath (Some 2000)
            let rawTest = MNIST.loadMnistFromCsv testFilePath (Some 600)

            let trainSet = buildBinaryDataset rawTrain
            let testSet = buildBinaryDataset rawTest

            printfn "Train samples: %d" (trainSet.Features.GetLength(0))
            printfn "Test samples: %d" (testSet.Features.GetLength(0))
            printfn ""

            let network : ConvLayers.CNNNetwork = [
                ConvLayers.Conv2D (ConvLayers.createConv2D 1 8 3 1 1 Activation.ReLU)
                ConvLayers.BatchNorm2D (ConvLayers.createBatchNorm2D 8)
                ConvLayers.MaxPool2D (ConvLayers.createMaxPool2D 2 2)
                ConvLayers.Conv2D (ConvLayers.createConv2D 8 16 3 1 1 Activation.ReLU)
                ConvLayers.BatchNorm2D (ConvLayers.createBatchNorm2D 16)
                ConvLayers.AvgPool2D (ConvLayers.createAvgPool2D 2 2)
                ConvLayers.Flatten (ConvLayers.createFlatten 16 7 7)
                ConvLayers.Dense (Layers.createLayer (16 * 7 * 7) 64 Activation.ReLU)
                ConvLayers.Dense (Layers.createLayer 64 2 Activation.Softmax)
            ]

            let config = {
                TrainerCNN.defaultConfig with
                    Epochs = 8
                    BatchSize = 16
                    LearningRate = 0.01
                    Optimizer = Some (Optimizers.Adam(0.001, 0.9, 0.999, 1e-8))
                    Loss = Losses.CrossEntropy
                    Verbose = true
            }

            printfn "=== TRAINING CNN ==="
            let metrics, trainedNetwork = TrainerCNN.train config network trainSet 1 28 28

            let trainAcc = TrainerCNN.accuracy trainedNetwork trainSet 1 28 28
            let testAcc = TrainerCNN.accuracy trainedNetwork testSet 1 28 28

            printfn "\n=== CNN RESULTS ==="
            printfn "Train accuracy: %.2f%%" (trainAcc * 100.0)
            printfn "Test accuracy: %.2f%%" (testAcc * 100.0)

            if testAcc < 0.85 then
                failwith $"CNN quality gate failed: expected test accuracy >= 85%%, got {testAcc * 100.0:N2}%%"

            let firstLoss = List.head metrics.TrainLoss
            let lastLoss = List.rev metrics.TrainLoss |> List.head
            printfn "First epoch loss: %.6f" firstLoss
            printfn "Final epoch loss: %.6f" lastLoss




