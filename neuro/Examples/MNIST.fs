// Examples/MNIST.fs
namespace Examples

open System.IO

module MNIST =
   
    let loadMnistFromCsv (filePath: string) (maxSamples: int option) =
        printfn $"Loading MNIST data from: {filePath}"
        
        if not (File.Exists(filePath)) then
            failwith $"File not found: {filePath}"
        
        let lines = File.ReadAllLines(filePath)
        let dataLines = 
            match maxSamples with
            | Some limit -> 
                let takeLimit = min limit (lines.Length - 1)
                printfn $"  Limiting to {takeLimit} samples"
                lines[1..takeLimit]
            | None -> lines[1..]
        
        let nSamples = dataLines.Length
        let nFeatures = 784
        let nClasses = 10
        
        printfn $"  Loading {nSamples} samples..."
        
        let images = Array2D.zeroCreate nSamples nFeatures
        let labels = Array2D.zeroCreate nSamples nClasses
        
        dataLines
        |> Array.iteri (fun i line ->
            let parts = line.Split(',')
            let label = int parts[0]
            labels[i, label] <- 1.0
            
            for j in 0 .. nFeatures - 1 do
                let pixelValue = float parts[j + 1] / 255.0
                images[i, j] <- pixelValue
        )
        
        printfn $"  Loaded {nSamples} samples\n"
        Data.createDataset images labels
    
    let printDigit (image: float[]) (width: int) =
        for y in 0 .. width - 1 do
            let row = Array.init width (fun x -> 
                let idx = y * width + x
                let pixel = image[idx]
                if pixel > 0.8 then "██"
                elif pixel > 0.6 then "▓▓"
                elif pixel > 0.4 then "▒▒"
                elif pixel > 0.2 then "░░"
                else "  ")
            let empty = ""
            printfn $"{String.concat empty row}"
    
    let run() =
        printfn "\n========================================"
        printfn "MNIST CLASSIFICATION WITH REAL DATA"
        printfn "========================================\n"
        
        let trainFilePath = __SOURCE_DIRECTORY__ + "\Data\mnist_train.csv"
        let testFilePath = __SOURCE_DIRECTORY__ + "\Data\mnist_test.csv"
        
        if not (File.Exists(trainFilePath)) then
            printfn "ERROR: MNIST training data not found at:"
            printfn $"  {trainFilePath}"
            printfn ""
            printfn "Please place mnist_train.csv and mnist_test.csv in the Data folder"
            printfn "Format: label,pixel1,pixel2,...,pixel784"
            ()
        elif not (File.Exists(testFilePath)) then
            printfn "ERROR: MNIST test data not found at:"
            printfn $"  {testFilePath}"
            ()
        else
            printfn "=== LOADING DATA ===\n"
            let trainSet = loadMnistFromCsv trainFilePath (Some 10000)
            let testSet = loadMnistFromCsv testFilePath (Some 2000)
            
            printfn "=== DATASET INFO ==="
            printfn $"Train samples: {trainSet.Features.GetLength(0)}"
            printfn $"Test samples: {testSet.Features.GetLength(0)}"
            printfn $"Features per sample: {trainSet.Features.GetLength(1)}"
            printfn "Classes: 10 (digits 0-9)"
            printfn ""
            
            printfn "=== CLASS DISTRIBUTION (Train Set) ==="
            let trainClassCounts = Array.zeroCreate 10
            for i in 0 .. (trainSet.Labels.GetLength(0) - 1) do
                for c in 0 .. 9 do
                    if trainSet.Labels[i, c] = 1.0 then trainClassCounts[c] <- trainClassCounts[c] + 1
            for c in 0 .. 9 do
                printfn $"  Digit {c}: {trainClassCounts[c]} samples"
            printfn ""
            
            printfn "=== SAMPLE DIGIT ==="
            let rnd = System.Random()
            let randomIndex = rnd.Next(trainSet.Features.GetLength(0))
            let sampleImage = Array.init 784 (fun j -> trainSet.Features[randomIndex, j])
            let sampleLabel = Array.findIndex (fun x -> x = 1.0) trainSet.Labels[randomIndex, *]
            printfn $"Sample index: {randomIndex}"
            printfn $"Label: {sampleLabel}"
            printfn "Image (28x28):"
            printDigit sampleImage 28
            printfn ""
            
            printfn "=== NETWORK ARCHITECTURE ==="
            let network = [
                Layers.createLayer 784 128 Activation.ReLU
                Layers.createLayer 128 64 Activation.ReLU
                Layers.createLayer 64 10 Activation.Softmax
            ]
            
            printfn "  Input: 784 (28x28 pixels)"
            printfn "  Hidden 1: 128 neurons (ReLU)"
            printfn "  Hidden 2: 64 neurons (ReLU)"
            printfn "  Output: 10 neurons (Softmax)"
            let totalParams = 784*128 + 128 + 128*64 + 64 + 64*10 + 10
            printfn $"  Total parameters: {totalParams}"
            printfn ""
            
            printfn "=== TRAINING CONFIGURATION ==="
            let config = {
                Trainer.defaultConfig with
                    Epochs = 20
                    BatchSize = 64
                    Optimizer = Optimizers.Adam(0.001, 0.9, 0.999, 1e-8)
                    Loss = Losses.CrossEntropy
                    Verbose = true
                    ValidationSplit = None
            }
            
            printfn "  Optimizer: Adam (lr=0.001)"
            printfn "  Loss: CrossEntropy"
            printfn "  Epochs: 20"
            printfn "  Batch size: 64"
            printfn ""
            printfn "Starting training (this may take a few minutes)...\n"
            
            let metrics, trainedNetwork = Trainer.train config network trainSet
            
            let trainAccuracy = Trainer.accuracy trainedNetwork trainSet
            let testAccuracy = Trainer.accuracy trainedNetwork testSet
            let trainAccuracyPercent = trainAccuracy * 100.0
            let testAccuracyPercent = testAccuracy * 100.0
            
            printfn "\n=== FINAL RESULTS ==="
            printfn $"Train Accuracy: {trainAccuracyPercent:N2}%%"
            printfn $"Test Accuracy: {testAccuracyPercent:N2}%%"
            
            let finalLoss = List.rev metrics.TrainLoss |> List.head
            let firstLoss = List.head metrics.TrainLoss
            
            printfn "\n=== LOSS PROGRESSION ==="
            printfn $"  First epoch loss: {firstLoss:N6}"
            printfn $"  Final epoch loss: {finalLoss:N6}"
            
            let testPredictions, _ = Layers.forwardNetwork trainedNetwork testSet.Features
            
            printfn "\n=== SAMPLE PREDICTIONS (first 20 test samples) ==="
            let mutable correct = 0
            for i in 0 .. 19 do
                let predClass = 
                    [|0..9|] |> Array.maxBy (fun j -> testPredictions[i, j])
                
                let trueClass = 
                    [|0..9|] |> Array.find (fun j -> testSet.Labels[i, j] = 1.0)
                
                let isCorrect = (predClass = trueClass)
                if isCorrect then correct <- correct + 1
                
                let maxProb = testPredictions[i, predClass]
                let confidencePercent = maxProb * 100.0
                let correctMark = if isCorrect then "OK" else "X"
                printfn $"  {i+1,2}: Pred={predClass}, Actual={trueClass}, Confidence={confidencePercent:N2}%% {correctMark}"
            
            let first20CorrectPercent = (float correct / 20.0 * 100.0)
            printfn "\n=== FIRST 20 TEST ACCURACY ==="
            printfn $"Correct: {correct}/20 = {first20CorrectPercent:N0}%%"
            
            let totalCorrect = 
                let mutable cnt = 0
                for i in 0 .. (testSet.Features.GetLength(0) - 1) do
                    let predClass = 
                        [|0..9|] |> Array.maxBy (fun j -> testPredictions[i, j])
                    let trueClass = 
                        [|0..9|] |> Array.find (fun j -> testSet.Labels[i, j] = 1.0)
                    if predClass = trueClass then cnt <- cnt + 1
                cnt
            
            let overallAccuracyPercent = (float totalCorrect / float (testSet.Features.GetLength(0)) * 100.0)
            
            printfn "\n=== TEST SUMMARY ==="
            printfn $"Total test samples: {testSet.Features.GetLength(0)}"
            printfn $"Correct predictions: {totalCorrect}"
            printfn $"Overall test accuracy: {overallAccuracyPercent:N2}%%"

            if testAccuracy < 0.95 then
                printfn "\n=== MISCLASSIFIED EXAMPLES (first 5) ==="
                let mutable shown = 0
                for i in 0 .. (testSet.Features.GetLength(0) - 1) do
                    if shown >= 5 then ()
                    else
                        let predClass = 
                            [|0..9|] |> Array.maxBy (fun j -> testPredictions[i, j])
                        let trueClass = 
                            [|0..9|] |> Array.find (fun j -> testSet.Labels[i, j] = 1.0)
                        if predClass <> trueClass then
                            let confidencePercent = testPredictions[i, predClass] * 100.0
                            printfn $"\n  Sample {i}:"
                            printfn $"    Predicted: {predClass}, Actual: {trueClass}"
                            printfn $"    Confidence: {confidencePercent:N2}%%"
                            printfn "    Top 3 probabilities:"
                            let top3 = 
                                [|0..9|] 
                                |> Array.sortByDescending (fun j -> testPredictions[i, j])
                                |> Array.take 3
                            for j in top3 do
                                let probPercent = testPredictions[i, j] * 100.0
                                printfn $"      {j}: {probPercent:N2}%%"
                            shown <- shown + 1