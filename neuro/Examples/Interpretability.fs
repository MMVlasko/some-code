namespace Examples
open System
open System.IO
module Interpretability =
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
            if dataset.Labels[originalIdx, 1] = 1.0 then binaryLabels[i, 1] <- 1.0 else binaryLabels[i, 0] <- 1.0
        Data.createDataset binaryFeatures binaryLabels
    let private runClassificationReport trainPath testPath =
        let trainSet = MNIST.loadMnistFromCsv trainPath (Some 5000)
        let testSet = MNIST.loadMnistFromCsv testPath (Some 1000)
        let network =
            [
                Layers.createLayer 784 128 Activation.ReLU
                Layers.createLayer 128 64 Activation.ReLU
                Layers.createLayer 64 10 Activation.Softmax
            ]
        let cfg =
            {
                Trainer.defaultConfig with
                    Epochs = 10
                    BatchSize = 64
                    Optimizer = Optimizers.Adam(0.001, 0.9, 0.999, 1e-8)
                    Loss = Losses.CrossEntropy
                    Verbose = false
                    ValidationSplit = None
            }
        let _, trained = Trainer.train cfg network trainSet
        let predictions, _ = Layers.forwardNetwork trained testSet.Features
        let predClasses = Experiments.classesFromPredictions predictions
        let trueClasses = Experiments.classesFromOneHot testSet.Labels
        let cm = Experiments.confusionMatrix 10 predClasses trueClasses
        let classNames = [| for i in 0 .. 9 -> string i |]
        let reportPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "mnist_interpretability_report.md")
        Experiments.writeClassificationMarkdown reportPath "MNIST Interpretability Report" classNames cm
        let accuracy = Trainer.accuracy trained testSet
        printfn "Dense model test accuracy: %.2f%%" (accuracy * 100.0)
        printfn "Classification report: %s" reportPath
    let private runCnnSaliency trainPath testPath =
        let rawTrain = MNIST.loadMnistFromCsv trainPath (Some 2000)
        let rawTest = MNIST.loadMnistFromCsv testPath (Some 500)
        let trainSet = buildBinaryDataset rawTrain
        let testSet = buildBinaryDataset rawTest
        let network : ConvLayers.CNNNetwork =
            [
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
        let cfg =
            {
                TrainerCNN.defaultConfig with
                    Epochs = 6
                    BatchSize = 16
                    Optimizer = Some (Optimizers.Adam(0.001, 0.9, 0.999, 1e-8))
                    Loss = Losses.CrossEntropy
                    Verbose = false
            }
        let _, trained = TrainerCNN.train cfg network trainSet 1 28 28
        let testPreds = TrainerCNN.predict trained testSet.Features 1 28 28
        let saliencyDir = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "saliency")
        Directory.CreateDirectory(saliencyDir) |> ignore
        for sampleIdx in 0 .. min 4 (testSet.Features.GetLength(0) - 1) do
            let predictedClass = if testPreds[sampleIdx, 1] > testPreds[sampleIdx, 0] then 1 else 0
            let sample = Array.init 784 (fun j -> testSet.Features[sampleIdx, j])
            let saliency = Experiments.cnnInputSaliency trained sample 1 28 28 predictedClass
            let outputPath = Path.Combine(saliencyDir, $"sample_{sampleIdx}_class_{predictedClass}.pgm")
            Experiments.writeSaliencyAsPgm outputPath 28 28 saliency
        let testAcc = TrainerCNN.accuracy trained testSet 1 28 28
        printfn "CNN binary test accuracy: %.2f%%" (testAcc * 100.0)
        printfn "Saliency maps written to: %s" saliencyDir
    let run () =
        printfn "\n========================================"
        printfn "INTERPRETABILITY TOOLS"
        printfn "========================================"
        let trainPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_train.csv")
        let testPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_test.csv")
        if not (File.Exists(trainPath)) || not (File.Exists(testPath)) then
            printfn "ERROR: MNIST CSV files were not found in Examples/Data"
        else
            printfn "Generating confusion matrix and per-class metrics..."
            runClassificationReport trainPath testPath
            printfn "\nGenerating CNN saliency maps..."
            runCnnSaliency trainPath testPath
