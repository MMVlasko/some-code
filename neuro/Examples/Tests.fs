// Examples/Tests.fs
namespace Examples

open System
open System.Diagnostics
open System.IO

module Tests =
    
    let private assertApproxEqual (actual: float) (expected: float) (tolerance: float) (message: string) =
        if abs (actual - expected) > tolerance then
            failwith $"{message}: Expected {expected}, got {actual} (diff: {abs(actual - expected)})"
    
    let private assertTrue (condition: bool) (message: string) =
        if not condition then
            failwith $"Assertion failed: {message}"
    
    let private assertEqual (actual: 'T) (expected: 'T) (message: string) =
        if actual <> expected then
            failwith $"{message}: Expected {expected}, got {actual}"
    
    
    let testDataFunctions () =
        printfn "\n=== TEST 1: Data Functions ==="
        
        let features = Array2D.init 10 3 (fun i j -> float (i + j))
        let labels = Array2D.init 10 1 (fun i _ -> float (i % 2))
        let dataset = Data.createDataset features labels
        
        printfn "  Testing createDataset..."
        assertEqual (dataset.Features.GetLength(0)) 10 "Dataset rows"
        assertEqual (dataset.Features.GetLength(1)) 3 "Dataset columns"
        
        printfn "  Testing map..."
        let mapped = Data.map (fun x -> x * 2.0) dataset
        assertApproxEqual mapped.Features[0, 0] 0.0 1e-10 "Map first element"
        assertApproxEqual mapped.Features[1, 0] 2.0 1e-10 "Map second element"
        
        printfn "  Testing normalize..."
        let normalized = Data.normalize dataset
        
        let cols = normalized.Features.GetLength(1)
        for j in 0 .. cols - 1 do
            let mutable sum = 0.0
            for i in 0 .. 9 do
                sum <- sum + normalized.Features[i, j]
            let mean = sum / 10.0
            assertApproxEqual mean 0.0 1e-10 $"Column {j} mean should be 0 after normalization"
        
        for j in 0 .. cols - 1 do
            let mutable sumSq = 0.0
            for i in 0 .. 9 do
                sumSq <- sumSq + (normalized.Features[i, j] * normalized.Features[i, j])
            let std = sqrt(sumSq / 10.0)
            assertApproxEqual std 1.0 1e-10 $"Column {j} std should be 1 after normalization"
        
        printfn "  Testing shuffle..."
        let shuffled = Data.shuffle dataset
        let firstRowChanged = 
            shuffled.Features[0, 0] <> dataset.Features[0, 0] ||
            shuffled.Features[0, 1] <> dataset.Features[0, 1] ||
            shuffled.Features[0, 2] <> dataset.Features[0, 2]
        assertTrue firstRowChanged "Shuffle should change order"
        
        printfn "  Testing batch..."
        let batches = Data.batch 3 dataset
        assertEqual (List.length batches) 4 "Number of batches"
        assertEqual (batches[0].Features.GetLength(0)) 3 "First batch size"
        assertEqual (batches[3].Features.GetLength(0)) 1 "Last batch size"
        
        printfn "  Testing split..."
        let train, value = Data.split 0.7 dataset
        assertEqual (train.Features.GetLength(0)) 7 "Train set size"
        assertEqual (value.Features.GetLength(0)) 3 "Validation set size"
        
        printfn "    Data functions passed!\n"
    
    let testMatrixOperations () =
        printfn "=== TEST 2: Matrix Operations ==="
        
        let a = Array2D.init 2 3 (fun i j -> float (i * 3 + j + 1))
        let b = Array2D.init 3 2 (fun i j -> float (i * 2 + j))
        
        printfn "  Testing matrix multiplication..."
        let result = Matrix.multiply a b
        
        assertApproxEqual result[0, 0] 16.0 1e-10 "Multiply element [0,0]"
        assertApproxEqual result[0, 1] 22.0 1e-10 "Multiply element [0,1]"
        assertApproxEqual result[1, 0] 34.0 1e-10 "Multiply element [1,0]"
        assertApproxEqual result[1, 1] 49.0 1e-10 "Multiply element [1,1]"
        
        printfn "  Testing transpose..."
        let transposed = Matrix.transpose a
        assertApproxEqual transposed[0, 0] a[0, 0] 1e-10 "Transpose element"
        assertApproxEqual transposed[1, 0] a[0, 1] 1e-10 "Transpose element"
        assertEqual (transposed.GetLength(0)) 3 "Transpose rows"
        assertEqual (transposed.GetLength(1)) 2 "Transpose columns"
        
        printfn "  Testing addition and subtraction..."
        let sum = Matrix.add a a
        assertApproxEqual sum[0, 0] (a[0, 0] * 2.0) 1e-10 "Addition"
        let diff = Matrix.subtract sum a
        assertApproxEqual diff[0, 0] a[0, 0] 1e-10 "Subtraction"
        
        printfn "  Testing scale..."
        let scaled = Matrix.scale 2.5 a
        assertApproxEqual scaled[0, 0] (a[0, 0] * 2.5) 1e-10 "Scale"
        
        printfn "  Testing hadamard..."
        let hadamard = Matrix.hadamard a a
        assertApproxEqual hadamard[0, 0] (a[0, 0] * a[0, 0]) 1e-10 "Hadamard"
        
        printfn "  Testing sum and mean..."
        let sumAll = Matrix.sum a
        assertApproxEqual sumAll 21.0 1e-10 "Sum all elements"
        let meanAll = Matrix.mean a
        assertApproxEqual meanAll (21.0 / 6.0) 1e-10 "Mean"
        
        printfn "    Matrix operations passed!\n"
    
    let testActivationFunctions () =
        printfn "=== TEST 3: Activation Functions ==="
        
        let testValues = [|-2.0; -1.0; 0.0; 1.0; 2.0|]
        
        printfn "  Testing Sigmoid..."
        let sigmoidResults = testValues |> Array.map (Activation.activate Activation.Sigmoid)
        assertApproxEqual sigmoidResults[0] (1.0 / (1.0 + exp(2.0))) 1e-10 "Sigmoid at -2"
        assertApproxEqual sigmoidResults[2] 0.5 1e-10 "Sigmoid at 0"
        assertApproxEqual sigmoidResults[4] (1.0 / (1.0 + exp(-2.0))) 1e-10 "Sigmoid at 2"
        
        let sigmoidDeriv = testValues |> Array.map (Activation.derivative Activation.Sigmoid)
        assertApproxEqual sigmoidDeriv[0] (sigmoidResults[0] * (1.0 - sigmoidResults[0])) 1e-10 "Sigmoid derivative"
        
        printfn "  Testing ReLU..."
        let reluResults = testValues |> Array.map (Activation.activate Activation.ReLU)
        assertApproxEqual reluResults[0] 0.0 1e-10 "ReLU at -2"
        assertApproxEqual reluResults[2] 0.0 1e-10 "ReLU at 0"
        assertApproxEqual reluResults[4] 2.0 1e-10 "ReLU at 2"
        
        let reluDeriv = testValues |> Array.map (Activation.derivative Activation.ReLU)
        assertApproxEqual reluDeriv[0] 0.0 1e-10 "ReLU derivative at -2"
        assertApproxEqual reluDeriv[2] 0.0 1e-10 "ReLU derivative at 0"
        assertApproxEqual reluDeriv[4] 1.0 1e-10 "ReLU derivative at 2"
        
        printfn "  Testing Tanh..."
        let tanhResults = testValues |> Array.map (Activation.activate Activation.Tanh)
        assertApproxEqual tanhResults[0] (tanh(-2.0)) 1e-10 "Tanh at -2"
        assertApproxEqual tanhResults[2] 0.0 1e-10 "Tanh at 0"
        assertApproxEqual tanhResults[4] (tanh(2.0)) 1e-10 "Tanh at 2"
        
        let tanhDeriv = testValues |> Array.map (Activation.derivative Activation.Tanh)
        assertApproxEqual tanhDeriv[0] (1.0 - tanhResults[0] * tanhResults[0]) 1e-10 "Tanh derivative"
        
        printfn "  Testing Softmax..."
        let softmaxInput = array2D [[1.0; 2.0; 3.0]; [1.0; 1.0; 1.0]]
        let softmaxOutput = Activation.apply Activation.Softmax softmaxInput
        
        let expSum = exp(1.0) + exp(2.0) + exp(3.0)
        assertApproxEqual softmaxOutput[0, 0] (exp(1.0) / expSum) 1e-10 "Softmax first row first col"
        assertApproxEqual softmaxOutput[0, 2] (exp(3.0) / expSum) 1e-10 "Softmax first row third col"
        assertApproxEqual softmaxOutput[1, 0] (1.0 / 3.0) 1e-10 "Softmax second row (equal)"
        
        printfn "    Activation functions passed!\n"
    
    let testLossFunctions () =
        printfn "=== TEST 4: Loss Functions ==="
        
        let predictions = array2D [[0.9; 0.1]; [0.4; 0.6]; [0.7; 0.3]]
        let targets = array2D [[1.0; 0.0]; [0.0; 1.0]; [1.0; 0.0]]
        
        printfn "  Testing MSE..."
        let mseLoss = Losses.compute Losses.MSE predictions targets
        let expectedMSE = ((0.9-1.0)**2.0 + (0.1-0.0)**2.0 + 
                           (0.4-0.0)**2.0 + (0.6-1.0)**2.0 + 
                           (0.7-1.0)**2.0 + (0.3-0.0)**2.0) / 6.0
        assertApproxEqual mseLoss expectedMSE 1e-10 "MSE computation"
        
        let mseGrad = Losses.gradient Losses.MSE predictions targets
        assertApproxEqual mseGrad[0, 0] ((0.9 - 1.0) * 2.0) 1e-10 "MSE gradient"
        
        printfn "  Testing CrossEntropy..."
        let ceLoss = Losses.compute Losses.CrossEntropy predictions targets
        let eps = 1e-8
        let expectedCE = (-log(max eps (min (1.0-eps) 0.9)) - log(max eps (min (1.0-eps) 0.6)) - log(max eps (min (1.0-eps) 0.7))) / 3.0
        assertApproxEqual ceLoss expectedCE 1e-5 "CrossEntropy computation"
        
        let ceGrad = Losses.gradient Losses.CrossEntropy predictions targets
        assertApproxEqual ceGrad[0, 0] (0.9 - 1.0) 1e-10 "CrossEntropy gradient"
        
        printfn "  Testing BinaryCrossEntropy..."
        let bceLoss = Losses.compute Losses.BinaryCrossEntropy predictions targets
       
        let manualBceLoss = 
            let total = 
                [| for i in 0..2 do
                    for j in 0..1 do
                        let pred = max 1e-8 (min (1.0-1e-8) predictions[i, j])
                        let target = targets[i, j]
                        yield - (target * log(pred) + (1.0 - target) * log(1.0 - pred)) |]
                |> Array.sum
            total / 3.0
        
        assertApproxEqual bceLoss manualBceLoss 1e-5 "BinaryCrossEntropy computation"
        
        let bceGrad = Losses.gradient Losses.BinaryCrossEntropy predictions targets
        let hasNaN = 
            [for i in 0..2 do
                for j in 0..1 do
                    yield Double.IsNaN(bceGrad[i, j])]
            |> List.exists id
        assertTrue (not hasNaN) "BinaryCrossEntropy gradient should not contain NaN"
        
        printfn "    Loss functions passed!\n"
    
    let testOptimizers () =
        printfn "=== TEST 5: Optimizers ==="
        
        let createSimpleDataset () =
            let features = Array2D.init 100 1 (fun i _ -> float i / 50.0 - 1.0)
            let labels = Array2D.init 100 1 (fun i _ -> 2.0 * features[i, 0])
            Data.createDataset features labels
        
        let createSimpleNetwork () = [
            Layers.createLayer 1 1 Activation.Linear
        ]
        
        let trainAndEvaluate _ (config: Trainer.TrainingConfig) =
            let dataset = createSimpleDataset ()
            let network = createSimpleNetwork ()
            
            let _, trainedNetwork = Trainer.train config network dataset
            
            let testPoints = array2D [[-0.5]; [0.0]; [0.5]]
            let predictions, _ = Layers.forwardNetwork trainedNetwork testPoints
            
            let errors = 
                [for i in 0..2 -> 
                    let expected = 2.0 * testPoints[i, 0]
                    abs(predictions[i, 0] - expected)]
            
            let maxError = List.max errors
            maxError
        
        printfn "  Testing SGD..."
        let sgdConfig = { 
            Trainer.defaultConfig with 
                Epochs = 200
                BatchSize = 16
                Optimizer = Optimizers.SGD 0.1
                Verbose = false
        }
        let sgdError = trainAndEvaluate "SGD" sgdConfig
        assertTrue (sgdError < 0.05) $"SGD should achieve error < 0.05, got {sgdError}"
        printfn $"    SGD error: {sgdError}"
        
        printfn "  Testing Momentum SGD..."
        let momentumConfig = { 
            Trainer.defaultConfig with 
                Epochs = 200
                BatchSize = 16
                Optimizer = Optimizers.Momentum(0.1, 0.9)
                Verbose = false
        }
        let momentumError = trainAndEvaluate "Momentum" momentumConfig
        assertTrue (momentumError < 0.05) $"Momentum should achieve error < 0.05, got {momentumError}"
        printfn $"    Momentum error: {momentumError}"
        
        printfn "  Testing Adam..."
        let adamConfig = { 
            Trainer.defaultConfig with 
                Epochs = 200
                BatchSize = 16
                Optimizer = Optimizers.Adam(0.1, 0.9, 0.999, 1e-8)
                Verbose = false
        }
        let adamError = trainAndEvaluate "Adam" adamConfig
        assertTrue (adamError < 0.05) $"Adam should achieve error < 0.05, got {adamError}"
        printfn $"    Adam error: {adamError}"
        
        printfn "  Testing Gradient Clipping..."
        let clipConfig = { 
            Trainer.defaultConfig with 
                Epochs = 200
                BatchSize = 16
                Optimizer = Optimizers.GradientClipping(0.1, 0.5)
                Verbose = false
        }
        let clipError = trainAndEvaluate "GradientClipping" clipConfig
        assertTrue (clipError < 0.05) $"Gradient Clipping should achieve error < 0.05, got {clipError}"
        printfn $"    Gradient Clipping error: {clipError}"
        
        printfn "    Optimizers passed!\n"
    
    let testActivationsAndLosses () =
        printfn "=== TEST 6: Different Activations and Losses ==="
        
        let createXORDataset () =
            let features = array2D [[0.0; 0.0]; [0.0; 1.0]; [1.0; 0.0]; [1.0; 1.0]]
            let labels = array2D [[0.0]; [1.0]; [1.0]; [0.0]]
            Data.createDataset features labels
        
        let testCombination name activation loss shouldWork =
            printfn $"  Testing {name}..."
            let dataset = createXORDataset ()
            let network = [
                Layers.createLayer 2 4 activation
                Layers.createLayer 4 1 (if activation = Activation.Softmax then Activation.Sigmoid else activation)
            ]
            
            let config = {
                Trainer.defaultConfig with
                    Epochs = 500
                    BatchSize = 4
                    Optimizer = Optimizers.Adam(0.1, 0.9, 0.999, 1e-8)
                    Loss = loss
                    Verbose = false
            }
            
            try
                let _, trainedNetwork = Trainer.train config network dataset
                let accuracy = Trainer.accuracy trainedNetwork dataset
                let accuracyPercent = accuracy * 100.0
                
                if shouldWork then
                    assertTrue (accuracy > 0.75) $"{name} should achieve >75%% accuracy, got {accuracyPercent:N2}%%"
                    printfn $"    ✓ Accuracy: {accuracyPercent:N2}%%"
                else
                    printfn "    Expected failure (testing error handling)"
            with
            | ex -> 
                if shouldWork then 
                    failwith $"{name} should work but failed with: {ex.Message}"
                else 
                    printfn "    ✓ Failed as expected"
        
        testCombination "ReLU + MSE" Activation.ReLU Losses.MSE true
        testCombination "Tanh + MSE" Activation.Tanh Losses.MSE true
        testCombination "Sigmoid + BinaryCrossEntropy" Activation.Sigmoid Losses.BinaryCrossEntropy true
        testCombination "Linear + MSE (Regression)" Activation.Linear Losses.MSE true
        
        printfn "    Activations and losses combinations passed!\n"
    
    let testFullTrainingCycle () =
        printfn "=== TEST 7: Full Training Cycle ==="
        
        let createSimpleClassificationDataset () =
            let rnd = Random(42)
            let features = Array2D.init 300 2 (fun _ _ -> (rnd.NextDouble() - 0.5) * 4.0)
            let labels = Array2D.zeroCreate 300 3
            
            for i in 0..299 do
                let x = features[i, 0]
                let y = features[i, 1]
                if x < -1.0 && y < -1.0 then labels[i, 0] <- 1.0
                elif x > 1.0 && y > 1.0 then labels[i, 1] <- 1.0
                elif x * x + y * y < 1.0 then labels[i, 2] <- 1.0
                else
                    let classIdx = rnd.Next(3)
                    labels[i, classIdx] <- 1.0
            
            Data.createDataset features labels
        
        printfn "  Creating dataset and network..."
        let dataset = createSimpleClassificationDataset ()
        let trainSet, testSet = Data.split 0.7 dataset
        
        let network = [
            Layers.createLayer 2 8 Activation.ReLU
            Layers.createLayer 8 8 Activation.ReLU
            Layers.createLayer 8 3 Activation.Softmax
        ]
        
        printfn "  Training network..."
        let config = {
            Trainer.defaultConfig with
                Epochs = 200
                BatchSize = 32
                Optimizer = Optimizers.Adam(0.01, 0.9, 0.999, 1e-8)
                Loss = Losses.CrossEntropy
                Verbose = false
        }
        
        let metrics, trainedNetwork = Trainer.train config network trainSet
        
        let firstLoss = List.head metrics.TrainLoss
        let lastLoss = List.rev metrics.TrainLoss |> List.head
        
        printfn $"    Initial loss: {firstLoss:N6}"
        printfn $"    Final loss: {lastLoss:N6}"
        assertTrue (lastLoss < firstLoss) $"Loss should decrease from {firstLoss} to {lastLoss}"
        
        let trainAccuracy = Trainer.accuracy trainedNetwork trainSet
        let testAccuracy = Trainer.accuracy trainedNetwork testSet
        let trainAccuracyPercent = trainAccuracy * 100.0
        let testAccuracyPercent = testAccuracy * 100.0
        
        printfn $"    Train accuracy: {trainAccuracyPercent:N2}%%"
        printfn $"    Test accuracy: {testAccuracyPercent:N2}%%"
        
        assertTrue (trainAccuracy > 0.6) $"Train accuracy should be >60%%, got {trainAccuracyPercent:N2}%%"
        assertTrue (testAccuracy > 0.5) $"Test accuracy should be >50%%, got {testAccuracyPercent:N2}%%"
        
        assertEqual (List.length metrics.TrainLoss) config.Epochs "Number of recorded losses"
        
        printfn "    Full training cycle passed!\n"

    let testGradients () =
        printfn "=== TEST 8: Gradient Computation ==="
        
        let numericalGradient (f: float[,] -> float) (x: float[,]) (epsilon: float) =
            let rows = x.GetLength(0)
            let cols = x.GetLength(1)
            let grad = Array2D.zeroCreate rows cols
            
            for i in 0..rows-1 do
                for j in 0..cols-1 do
                    let original = x[i, j]
                    
                    x[i, j] <- original + epsilon
                    let fPlus = f x
                    
                    x[i, j] <- original - epsilon
                    let fMinus = f x
                    
                    grad[i, j] <- (fPlus - fMinus) / (2.0 * epsilon)
                    x[i, j] <- original
            
            grad
        
        printfn "  Testing gradients for simple linear network..."
        
        let input = array2D [[1.0]]
        let target = array2D [[2.0]]
        let layer = Layers.createLayer 1 1 Activation.Linear
        
        let z, output = Layers.forward layer input
        
        let lossGrad = Losses.gradient Losses.MSE output target
        
        let _, gradWeight, gradBias = Layers.backward layer lossGrad input z output
        
        let fWeights (w: float[,]) =
            let testLayer = { layer with Weights = w }
            let _, testOutput = Layers.forward testLayer input
            Losses.compute Losses.MSE testOutput target
        
        let numericalGradWeight = numericalGradient fWeights layer.Weights 1e-6
        
        let maxWeightDiff = 
            [for i in 0..gradWeight.GetLength(0)-1 do
                for j in 0..gradWeight.GetLength(1)-1 do
                    yield abs(gradWeight[i, j] - numericalGradWeight[i, j])]
            |> List.max
        
        printfn $"    Max weight gradient difference: {maxWeightDiff:e}"
        assertTrue (maxWeightDiff < 1e-5) $"Weight gradient difference too large: {maxWeightDiff:e}"
        
        let fBias (b: float[]) =
            let testLayer = { layer with Bias = b }
            let _, testOutput = Layers.forward testLayer input
            Losses.compute Losses.MSE testOutput target
        
        let numericalGradBias = 
            [| for i in 0..gradBias.Length-1 do
                let original = layer.Bias[i]
                let eps = 1e-6
                
                layer.Bias[i] <- original + eps
                let fPlus = fBias layer.Bias
                
                layer.Bias[i] <- original - eps
                let fMinus = fBias layer.Bias
                
                layer.Bias[i] <- original
                yield (fPlus - fMinus) / (2.0 * eps) |]
        
        let maxBiasDiff = 
            [for i in 0..gradBias.Length-1 do
                yield abs(gradBias[i] - numericalGradBias[i])]
            |> List.max
        
        printfn $"    Max bias gradient difference: {maxBiasDiff:e}"
        assertTrue (maxBiasDiff < 1e-5) $"Bias gradient difference too large: {maxBiasDiff:e}"
        
        printfn "  Testing gradients for sigmoid network..."
        
        let sigmoidLayer = Layers.createLayer 1 1 Activation.Sigmoid
        
        let zSigmoid, outputSigmoid = Layers.forward sigmoidLayer input
        
        let lossGradSigmoid = Losses.gradient Losses.MSE outputSigmoid target
        
        let _, gradWeightSigmoid, _ = Layers.backward sigmoidLayer lossGradSigmoid input zSigmoid outputSigmoid
        
        let fSigmoidWeights (w: float[,]) =
            let testLayer = { sigmoidLayer with Weights = w }
            let _, testOutput = Layers.forward testLayer input
            Losses.compute Losses.MSE testOutput target
        
        let numericalGradSigmoid = numericalGradient fSigmoidWeights sigmoidLayer.Weights 1e-6
        
        let maxSigmoidDiff = 
            [for i in 0..gradWeightSigmoid.GetLength(0)-1 do
                for j in 0..gradWeightSigmoid.GetLength(1)-1 do
                    yield abs(gradWeightSigmoid[i, j] - numericalGradSigmoid[i, j])]
            |> List.max
        
        printfn $"    Max sigmoid gradient difference: {maxSigmoidDiff:e}"
        assertTrue (maxSigmoidDiff < 1e-5) $"Sigmoid gradient difference too large: {maxSigmoidDiff:e}"
        
        printfn "  Testing multi-layer network gradients..."
        
        // Use smooth activation to avoid dead-ReLU false negatives in this gradient-shape test.
        let layer1 = Layers.createLayer 2 3 Activation.Tanh
        let layer2 = Layers.createLayer 3 1 Activation.Linear
        
        let multiInput = array2D [[0.5; -0.5]]
        let multiTarget = array2D [[1.0]]
        
        let z1, output1 = Layers.forward layer1 multiInput
        let z2, output2 = Layers.forward layer2 output1
        
        let lossGradFinal = Losses.gradient Losses.MSE output2 multiTarget
        
        let gradInput2, gradWeight2, _ = Layers.backward layer2 lossGradFinal output1 z2 output2
        let _, gradWeight1, _ = Layers.backward layer1 gradInput2 multiInput z1 output1
        
        let hasNonZeroWeight1 = 
            [for i in 0..gradWeight1.GetLength(0)-1 do
                for j in 0..gradWeight1.GetLength(1)-1 do
                    yield abs(gradWeight1[i, j])]
            |> List.exists (fun x -> x > 1e-10)
        
        let hasNonZeroWeight2 = 
            [for i in 0..gradWeight2.GetLength(0)-1 do
                for j in 0..gradWeight2.GetLength(1)-1 do
                    yield abs(gradWeight2[i, j])]
            |> List.exists (fun x -> x > 1e-10)
        
        assertTrue hasNonZeroWeight1 "First layer weight gradients should be non-zero"
        assertTrue hasNonZeroWeight2 "Second layer weight gradients should be non-zero"
        
        assertEqual (gradWeight1.GetLength(0)) 3 "First layer weight rows"
        assertEqual (gradWeight1.GetLength(1)) 2 "First layer weight cols"
        assertEqual (gradWeight2.GetLength(0)) 1 "Second layer weight rows"
        assertEqual (gradWeight2.GetLength(1)) 3 "Second layer weight cols"
        
        printfn "    Multi-layer gradients computed successfully"
        printfn $"    First layer weight gradient shape: {gradWeight1.GetLength(0)}x{gradWeight1.GetLength(1)}"
        printfn $"    Second layer weight gradient shape: {gradWeight2.GetLength(0)}x{gradWeight2.GetLength(1)}"
        
        printfn "  Testing gradient flow through multiple layers..."

        let net1 = Layers.createLayer 2 4 Activation.Tanh
        // Use smooth activation to avoid dead-ReLU false negatives in this flow test.
        let net2 = Layers.createLayer 4 3 Activation.Tanh
        let net3 = Layers.createLayer 3 1 Activation.Linear
        
        let testInput = array2D [[0.3; -0.2]]
        let testTarget = array2D [[0.5]]
        
        let zz1, out1 = Layers.forward net1 testInput
        let zz2, out2 = Layers.forward net2 out1
        let zz3, out3 = Layers.forward net3 out2
        
        let finalGrad = Losses.gradient Losses.MSE out3 testTarget
        let gradIn3, _, _ = Layers.backward net3 finalGrad out2 zz3 out3
        let gradIn2, _, _ = Layers.backward net2 gradIn3 out1 zz2 out2
        let gradIn1, gradW1, _ = Layers.backward net1 gradIn2 testInput zz1 out1
        
        let allGradientsNonZero = 
            [for i in 0..gradW1.GetLength(0)-1 do
                for j in 0..gradW1.GetLength(1)-1 do
                    yield abs(gradW1[i, j])]
            |> List.exists (fun x -> x > 1e-10)
        
        assertTrue allGradientsNonZero "All layer gradients should propagate through network"
        
        assertEqual (gradIn1.GetLength(0)) 1 "Gradient input rows"
        assertEqual (gradIn1.GetLength(1)) 2 "Gradient input cols"
        
        printfn "    Gradient flow verified through 3 layers"
        
        printfn "    Gradient computation passed!\n"

    let testCNNBlock () =
        printfn "=== TEST 9: CNN Block ==="

        printfn "  Testing tensor/matrix round-trip..."
        let matrix = Array2D.init 2 16 (fun i j -> float (i * 16 + j) / 16.0)
        let tensor = ConvLayers.matrixToTensor matrix 1 4 4
        let matrixBack = ConvLayers.tensorToMatrix tensor

        for i in 0 .. matrix.GetLength(0) - 1 do
            for j in 0 .. matrix.GetLength(1) - 1 do
                assertApproxEqual matrixBack[i, j] matrix[i, j] 1e-10 "Tensor/matrix round-trip"

        printfn "  Testing forward/backward shapes..."
        let testNetwork : ConvLayers.CNNNetwork = [
            ConvLayers.Conv2D (ConvLayers.createConv2D 1 4 3 1 1 Activation.ReLU)
            ConvLayers.BatchNorm2D (ConvLayers.createBatchNorm2D 4)
            ConvLayers.AvgPool2D (ConvLayers.createAvgPool2D 2 2)
            ConvLayers.ReshapeToMatrix (ConvLayers.createReshapeToMatrix 4 2 2)
            ConvLayers.Dense (Layers.createLayer (4 * 2 * 2) 2 Activation.Softmax)
        ]

        let inputBatch = Array2D.init 3 16 (fun i j -> float ((i + j) % 5) / 5.0)
        let inputTensor = ConvLayers.matrixToTensor inputBatch 1 4 4
        let output, caches = ConvLayers.forwardNetwork testNetwork (ConvLayers.Tensor4D inputTensor)

        let predictions =
            match output with
            | ConvLayers.Matrix m -> m
            | _ -> failwith "CNN output should be matrix after Flatten + Dense"

        assertEqual (predictions.GetLength(0)) 3 "CNN forward batch size"
        assertEqual (predictions.GetLength(1)) 2 "CNN forward class count"
        assertEqual (List.length caches) 5 "CNN cache length"

        let labels = array2D [[1.0; 0.0]; [0.0; 1.0]; [1.0; 0.0]]
        let grad = Losses.gradient Losses.CrossEntropy predictions labels
        let _, grads = ConvLayers.backwardNetwork testNetwork caches (ConvLayers.Matrix grad)
        assertEqual (List.length grads) 5 "CNN gradients length"

        printfn "  Testing CNN training quality gate >= 85%% with Adam and Momentum..."
        let createSyntheticImageDataset samplesPerClass =
            let total = samplesPerClass * 2
            let features = Array2D.zeroCreate total 64
            let labels = Array2D.zeroCreate total 2

            let writePixel i row col value =
                let idx = row * 8 + col
                features[i, idx] <- value

            for i in 0 .. samplesPerClass - 1 do
                let idx = i
                labels[idx, 0] <- 1.0
                for r in 0 .. 7 do
                    writePixel idx r 3 1.0

            for i in 0 .. samplesPerClass - 1 do
                let idx = samplesPerClass + i
                labels[idx, 1] <- 1.0
                for c in 0 .. 7 do
                    writePixel idx 4 c 1.0

            Data.createDataset features labels

        let trainSet = createSyntheticImageDataset 120
        let cnn : ConvLayers.CNNNetwork = [
            ConvLayers.Conv2D (ConvLayers.createConv2D 1 4 3 1 1 Activation.ReLU)
            ConvLayers.BatchNorm2D (ConvLayers.createBatchNorm2D 4)
            ConvLayers.MaxPool2D (ConvLayers.createMaxPool2D 2 2)
            ConvLayers.Flatten (ConvLayers.createFlatten 4 4 4)
            ConvLayers.Dense (Layers.createLayer (4 * 4 * 4) 2 Activation.Softmax)
        ]

        let adamConfig = {
            TrainerCNN.defaultConfig with
                Epochs = 6
                BatchSize = 16
                LearningRate = 0.03
                Optimizer = Some (Optimizers.Adam(0.01, 0.9, 0.999, 1e-8))
                Loss = Losses.CrossEntropy
                Verbose = false
        }

        let momentumConfig = {
            TrainerCNN.defaultConfig with
                Epochs = 6
                BatchSize = 16
                LearningRate = 0.03
                Optimizer = Some (Optimizers.Momentum(0.03, 0.9))
                Loss = Losses.CrossEntropy
                Verbose = false
        }

        let _, trainedAdam = TrainerCNN.train adamConfig cnn trainSet 1 8 8
        let adamAcc = TrainerCNN.accuracy trainedAdam trainSet 1 8 8
        assertTrue (adamAcc >= 0.85) $"CNN Adam accuracy should be >= 85%%, got {adamAcc * 100.0:N2}%%"

        let _, trainedMomentum = TrainerCNN.train momentumConfig cnn trainSet 1 8 8
        let momentumAcc = TrainerCNN.accuracy trainedMomentum trainSet 1 8 8
        assertTrue (momentumAcc >= 0.85) $"CNN Momentum accuracy should be >= 85%%, got {momentumAcc * 100.0:N2}%%"

        printfn $"    CNN synthetic accuracy (Adam): {adamAcc * 100.0:N2}%%"
        printfn $"    CNN synthetic accuracy (Momentum): {momentumAcc * 100.0:N2}%%"
        printfn "    CNN block passed!\n"

    let testExperimentHelpers () =
        printfn "=== TEST 10: Experiment Helpers ==="

        printfn "  Testing confusion matrix and class metrics..."
        let preds = [|0; 1; 2; 1; 0|]
        let targets = [|0; 2; 2; 1; 0|]
        let cm = Experiments.confusionMatrix 3 preds targets
        assertEqual cm[0, 0] 2 "ConfusionMatrix true0/pred0"
        assertEqual cm[2, 1] 1 "ConfusionMatrix true2/pred1"

        let metrics = Experiments.perClassMetrics cm
        assertEqual metrics.Length 3 "Per-class metrics length"
        assertTrue (metrics[0].Precision > 0.5) "Class0 precision should be > 0.5"

        printfn "  Testing classification markdown export..."
        let reportPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "test_classification_report.md")
        Experiments.writeClassificationMarkdown reportPath "Test Classification" [|"0"; "1"; "2"|] cm
        assertTrue (File.Exists(reportPath)) "Classification report should be created"

        printfn "  Testing saliency PGM export..."
        let saliencyPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "test_saliency.pgm")
        let saliency = Array.init (4 * 4) (fun i -> float i)
        Experiments.writeSaliencyAsPgm saliencyPath 4 4 saliency
        assertTrue (File.Exists(saliencyPath)) "Saliency PGM should be created"

        printfn "    Experiment helpers passed!\n"

    let testRegularizationTransforms () =
        printfn "=== TEST 11: Regularization Transforms ==="

        let rows = 4
        let features = Array2D.init rows 784 (fun i j -> float (i * 1000 + j))
        let labels =
            Array2D.init rows 10 (fun i j ->
                if j = (i % 10) then 1.0 else 0.0)

        let dataset = Data.createDataset features labels

        printfn "  Testing label smoothing..."
        let smoothingCfg: RegularizationMNIST.RegConfig =
            { Name = "smooth"; LabelSmoothing = 0.1; WeightDecay = 0.0; Mixup = false; Cutmix = false }
        let smoothedDs = RegularizationMNIST.buildTrainDatasetForTesting dataset smoothingCfg
        let smoothed = smoothedDs.Labels
        for i in 0 .. rows - 1 do
            let rowSum = [|0 .. 9|] |> Array.sumBy (fun j -> smoothed[i, j])
            assertApproxEqual rowSum 1.0 1e-10 "Smoothed row should sum to 1"

        printfn "  Testing mixup transform..."
        let mixupCfg: RegularizationMNIST.RegConfig =
            { Name = "mixup"; LabelSmoothing = 0.0; WeightDecay = 0.0; Mixup = true; Cutmix = false }
        let mixed = RegularizationMNIST.buildTrainDatasetForTesting dataset mixupCfg
        assertEqual (mixed.Features.GetLength(0)) rows "Mixup rows"
        assertEqual (mixed.Features.GetLength(1)) 784 "Mixup cols"
        for i in 0 .. rows - 1 do
            let rowSum = [|0 .. 9|] |> Array.sumBy (fun j -> mixed.Labels[i, j])
            assertApproxEqual rowSum 1.0 1e-10 "Mixup label row should sum to 1"

        let mixupChanged =
            [ for i in 0 .. rows - 1 do
                for j in 0 .. 20 do
                    yield abs (mixed.Features[i, j] - dataset.Features[i, j]) ]
            |> List.exists (fun d -> d > 1e-10)
        assertTrue mixupChanged "Mixup should modify at least some feature values"

        printfn "  Testing cutmix transform..."
        let cutmixCfg: RegularizationMNIST.RegConfig =
            { Name = "cutmix"; LabelSmoothing = 0.0; WeightDecay = 0.0; Mixup = false; Cutmix = true }
        let cutmixed = RegularizationMNIST.buildTrainDatasetForTesting dataset cutmixCfg
        assertEqual (cutmixed.Features.GetLength(0)) rows "Cutmix rows"
        assertEqual (cutmixed.Features.GetLength(1)) 784 "Cutmix cols"
        for i in 0 .. rows - 1 do
            let rowSum = [|0 .. 9|] |> Array.sumBy (fun j -> cutmixed.Labels[i, j])
            assertApproxEqual rowSum 1.0 1e-10 "Cutmix label row should sum to 1"

        let cutmixChanged =
            [ for i in 0 .. rows - 1 do
                for j in 0 .. 783 do
                    yield abs (cutmixed.Features[i, j] - dataset.Features[i, j]) ]
            |> List.exists (fun d -> d > 1e-10)
        assertTrue cutmixChanged "Cutmix should copy at least one patch into features"

        printfn "    Regularization transforms passed!\n"
    
    let run () =
        printfn "\n========================================"
        printfn "COMPREHENSIVE FRAMEWORK TESTS"
        printfn "========================================"
        
        let stopwatch = Stopwatch()
        stopwatch.Start()
        
        try
            testDataFunctions()
            testMatrixOperations()
            testActivationFunctions()
            testLossFunctions()
            testOptimizers()
            testActivationsAndLosses()
            testFullTrainingCycle()
            testGradients()
            testCNNBlock()
            testExperimentHelpers()
            testRegularizationTransforms()
            
            stopwatch.Stop()
            
            printfn "\n========================================"
            printfn "  ALL TESTS PASSED!"
            printfn $"Total execution time: {stopwatch.Elapsed.TotalSeconds:N2} seconds"
            printfn "========================================\n"
            
        with
        | ex ->
            stopwatch.Stop()
            printfn "\n========================================"
            printfn "  TEST FAILED!"
            printfn $"Error: {ex.Message}"
            printfn "========================================\n"
            reraise()
