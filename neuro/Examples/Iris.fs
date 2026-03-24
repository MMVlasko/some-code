// Examples/Iris.fs
namespace Examples

open System
open System.IO

module Iris =
    
    let loadDataFromCsv (filePath: string) =
        let lines = File.ReadAllLines(filePath)
        let dataLines = lines[1..]
        
        let nSamples = dataLines.Length
        let nFeatures = 4
        let nClasses = 3
        
        let features = Array2D.zeroCreate nSamples nFeatures
        let labels = Array2D.zeroCreate nSamples nClasses
        
        dataLines
        |> Array.iteri (fun i line ->
            let parts = line.Split(',')
            features[i, 0] <- float parts[1]
            features[i, 1] <- float parts[2]
            features[i, 2] <- float parts[3]
            features[i, 3] <- float parts[4]
            
            match parts[5].Trim() with
            | "Iris-setosa" -> labels[i, 0] <- 1.0
            | "Iris-versicolor" -> labels[i, 1] <- 1.0
            | "Iris-virginica" -> labels[i, 2] <- 1.0
            | _ -> ()
        )
        
        Data.createDataset features labels
    
    let stratifiedSplit (ratio: float) (dataset: Data.Dataset) =
        let nSamples = dataset.Features.GetLength(0)
        let nClasses = dataset.Labels.GetLength(1)
        
        let classIndices = Array.init nClasses (fun _ -> System.Collections.Generic.List<int>())
        for i in 0 .. nSamples - 1 do
            for c in 0 .. nClasses - 1 do
                if dataset.Labels[i, c] = 1.0 then
                    classIndices[c].Add(i)
        
        let rnd = Random(42)
        let trainIndicesList = System.Collections.Generic.List<int>()
        let valIndicesList = System.Collections.Generic.List<int>()
        
        for c in 0 .. nClasses - 1 do
            let indices = classIndices[c].ToArray()
            let shuffled = indices |> Array.sortBy (fun _ -> rnd.NextDouble())
            let nTrain = int (float shuffled.Length * ratio)
            
            for i in 0 .. nTrain - 1 do
                trainIndicesList.Add(shuffled[i])
            for i in nTrain .. shuffled.Length - 1 do
                valIndicesList.Add(shuffled[i])
        
        let nTrain = trainIndicesList.Count
        let nVal = valIndicesList.Count
        let nFeatures = dataset.Features.GetLength(1)
        let nLabels = dataset.Labels.GetLength(1)
        
        let trainFeatures = Array2D.init nTrain nFeatures (fun i j -> 
            dataset.Features[trainIndicesList[i], j])
        let trainLabels = Array2D.init nTrain nLabels (fun i j -> 
            dataset.Labels[trainIndicesList[i], j])
        
        let valFeatures = Array2D.init nVal nFeatures (fun i j -> 
            dataset.Features[valIndicesList[i], j])
        let valLabels = Array2D.init nVal nLabels (fun i j -> 
            dataset.Labels[valIndicesList[i], j])
        
        (Data.createDataset trainFeatures trainLabels,
         Data.createDataset valFeatures valLabels)
    
    let run() =
        printfn "\n========================================"
        printfn "IRIS CLASSIFICATION WITH REAL DATA"
        printfn "========================================\n"
        
        let filePath = __SOURCE_DIRECTORY__ + "\Data\Iris.csv"
        
        if not (File.Exists(filePath)) then
            printfn $"ERROR: Iris dataset not found at {filePath}"
            printfn "Please place Iris.csv in the Data folder"
            ()
        else
            let fullDataset = loadDataFromCsv filePath
            
            printfn "=== DATASET INFO ==="
            printfn $"Total samples: {fullDataset.Features.GetLength(0)}"
            printfn $"Features per sample: {fullDataset.Features.GetLength(1)}"
            printfn "Classes: 3 (Setosa, Versicolor, Virginica)"
            
            printfn "\n=== CLASS DISTRIBUTION (Full Dataset) ==="
            let classCountsFull = Array.zeroCreate 3
            for i in 0 .. (fullDataset.Labels.GetLength(0) - 1) do
                if fullDataset.Labels[i, 0] = 1.0 then classCountsFull[0] <- classCountsFull[0] + 1
                elif fullDataset.Labels[i, 1] = 1.0 then classCountsFull[1] <- classCountsFull[1] + 1
                elif fullDataset.Labels[i, 2] = 1.0 then classCountsFull[2] <- classCountsFull[2] + 1
            printfn $"  Setosa: {classCountsFull[0]}"
            printfn $"  Versicolor: {classCountsFull[1]}"
            printfn $"  Virginica: {classCountsFull[2]}"
            printfn ""
            
            let normalizedDataset = Data.normalize fullDataset
            
            let trainSet, valSet = stratifiedSplit 0.8 normalizedDataset
            
            printfn "=== SPLIT INFO ==="
            printfn $"Train samples: {trainSet.Features.GetLength(0)}"
            printfn $"Validation samples: {valSet.Features.GetLength(0)}"
            
            printfn "\n=== CLASS DISTRIBUTION (Train Set) ==="
            let classCountsTrain = Array.zeroCreate 3
            for i in 0 .. (trainSet.Labels.GetLength(0) - 1) do
                if trainSet.Labels[i, 0] = 1.0 then classCountsTrain[0] <- classCountsTrain[0] + 1
                elif trainSet.Labels[i, 1] = 1.0 then classCountsTrain[1] <- classCountsTrain[1] + 1
                elif trainSet.Labels[i, 2] = 1.0 then classCountsTrain[2] <- classCountsTrain[2] + 1
            printfn $"  Setosa: {classCountsTrain[0]}"
            printfn $"  Versicolor: {classCountsTrain[1]}"
            printfn $"  Virginica: {classCountsTrain[2]}"
            
            printfn "\n=== CLASS DISTRIBUTION (Validation Set) ==="
            let classCountsVal = Array.zeroCreate 3
            for i in 0 .. (valSet.Labels.GetLength(0) - 1) do
                if valSet.Labels[i, 0] = 1.0 then classCountsVal[0] <- classCountsVal[0] + 1
                elif valSet.Labels[i, 1] = 1.0 then classCountsVal[1] <- classCountsVal[1] + 1
                elif valSet.Labels[i, 2] = 1.0 then classCountsVal[2] <- classCountsVal[2] + 1
            printfn $"  Setosa: {classCountsVal[0]}"
            printfn $"  Versicolor: {classCountsVal[1]}"
            printfn $"  Virginica: {classCountsVal[2]}"
            printfn ""
           
            printfn "=== SAMPLE VALIDATION DATA ==="
            for i in 0 .. min 4 (valSet.Features.GetLength(0) - 1) do
                let className = 
                    if valSet.Labels[i, 0] = 1.0 then "Setosa"
                    elif valSet.Labels[i, 1] = 1.0 then "Versicolor"
                    else "Virginica"
                printfn $"  Sample {i}: [{valSet.Features[i, 0]:N2}, {valSet.Features[i, 1]:N2}, {valSet.Features[i, 2]:N2}, {valSet.Features[i, 3]:N2}] -> {className}"
            printfn ""
            
            let network = [
                Layers.createLayer 4 8 Activation.ReLU
                Layers.createLayer 8 3 Activation.Softmax
            ]
            
            printfn "=== NETWORK ARCHITECTURE ==="
            printfn "  Input: 4 features"
            printfn "  Hidden: 8 neurons (ReLU)"
            printfn "  Output: 3 neurons (Softmax)"
            printfn ""
            
            let config = {
                Trainer.defaultConfig with
                    Epochs = 1000
                    BatchSize = 16
                    Optimizer = Optimizers.Adam(0.001, 0.9, 0.999, 1e-8)
                    Loss = Losses.CrossEntropy
                    Verbose = true
                    ValidationSplit = None
            }
            
            printfn "=== TRAINING CONFIGURATION ==="
            printfn "  Optimizer: Adam (lr=0.001)"
            printfn "  Loss: CrossEntropy"
            printfn "  Epochs: 1000"
            printfn "  Batch size: 16"
            printfn ""
            
            let metrics, trainedNetwork = Trainer.train config network trainSet
            
            let trainAccuracy = Trainer.accuracy trainedNetwork trainSet
            let valAccuracy = Trainer.accuracy trainedNetwork valSet
            let trainAccuracyPercent = trainAccuracy * 100.0
            let valAccuracyPercent = valAccuracy * 100.0
            
            printfn "\n=== FINAL RESULTS ==="
            printfn $"Train Accuracy: {trainAccuracyPercent:N2}%%"
            printfn $"Validation Accuracy: {valAccuracyPercent:N2}%%"
            
            let finalLoss = List.rev metrics.TrainLoss |> List.head
            let firstLoss = List.head metrics.TrainLoss
            
            printfn "\n=== LOSS PROGRESSION ==="
            printfn $"  First epoch loss: {firstLoss:N6}"
            printfn $"  Final epoch loss: {finalLoss:N6}"
            
            let valPredictions, _ = Layers.forwardNetwork trainedNetwork valSet.Features
            
            printfn "\n=== VALIDATION PREDICTIONS (first 10 samples) ==="
            let classNames = [|"Setosa"; "Versicolor"; "Virginica"|]
            let mutable correct = 0
            for i in 0 .. min 9 (valSet.Features.GetLength(0) - 1) do
                let predClass = 
                    if valPredictions[i, 0] > valPredictions[i, 1] && 
                       valPredictions[i, 0] > valPredictions[i, 2] then 0
                    elif valPredictions[i, 1] > valPredictions[i, 2] then 1
                    else 2
                
                let trueClass = 
                    if valSet.Labels[i, 0] = 1.0 then 0
                    elif valSet.Labels[i, 1] = 1.0 then 1
                    else 2
                
                let isCorrect = (predClass = trueClass)
                if isCorrect then correct <- correct + 1
                
                let correctMark = if isCorrect then "OK" else "X"
                printfn $"  {i+1,2}: Pred={classNames[predClass],9}, Actual={classNames[trueClass],9}, Probs=[{valPredictions[i, 0]:N4}, {valPredictions[i, 1]:N4}, {valPredictions[i, 2]:N4}] {correctMark}"
            
            let first10CorrectPercent = (float correct / 10.0 * 100.0)
            printfn "\n=== FIRST 10 VALIDATION ACCURACY ==="
            printfn $"Correct: {correct}/10 = {first10CorrectPercent:N0}%%"
            
            let totalCorrect = 
                let mutable cnt = 0
                for i in 0 .. (valSet.Features.GetLength(0) - 1) do
                    let predClass = 
                        if valPredictions[i, 0] > valPredictions[i, 1] && 
                           valPredictions[i, 0] > valPredictions[i, 2] then 0
                        elif valPredictions[i, 1] > valPredictions[i, 2] then 1
                        else 2
                    let trueClass = 
                        if valSet.Labels[i, 0] = 1.0 then 0
                        elif valSet.Labels[i, 1] = 1.0 then 1
                        else 2
                    if predClass = trueClass then cnt <- cnt + 1
                cnt
            
            let overallAccuracyPercent = (float totalCorrect / float (valSet.Features.GetLength(0)) * 100.0)
            
            printfn "\n=== VALIDATION SUMMARY ==="
            printfn $"Total validation samples: {valSet.Features.GetLength(0)}"
            printfn $"Correct predictions: {totalCorrect}"
            printfn $"Overall validation accuracy: {overallAccuracyPercent:N2}%%"