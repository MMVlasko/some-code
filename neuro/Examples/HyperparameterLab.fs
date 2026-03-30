namespace Examples
open System
open System.IO
module HyperparameterLab =
    let private activationToName = function
        | Activation.ReLU -> "ReLU"
        | Activation.Tanh -> "Tanh"
        | Activation.Sigmoid -> "Sigmoid"
        | Activation.Linear -> "Linear"
        | Activation.Softmax -> "Softmax"
    let private createNetwork hiddenActivation =
        [
            Layers.createLayer 4 16 hiddenActivation
            Layers.createLayer 16 3 Activation.Softmax
        ]
    let private runSingle name hiddenActivation lr optimizer batchSize seed trainSet testSet : Experiments.RunResult =
        let network = createNetwork hiddenActivation
        let config =
            {
                Trainer.defaultConfig with
                    Epochs = 240
                    BatchSize = batchSize
                    Optimizer = optimizer
                    Loss = Losses.CrossEntropy
                    Verbose = false
                    ValidationSplit = None
                    RandomSeed = seed
            }
        let (metrics, trainedNetwork), durationMs, memoryBytes =
            Experiments.measureRun (fun () -> Trainer.train config network trainSet)
        let trainAccuracy = Trainer.accuracy trainedNetwork trainSet
        let testAccuracy = Trainer.accuracy trainedNetwork testSet
        let finalLoss = metrics.TrainLoss |> List.last
        ({ 
            RunId = Experiments.createRunId "hpl"
            Name = name
            Params = $"act={activationToName hiddenActivation}; lr={lr}; opt={optimizer}; batch={batchSize}"
            Seed = seed
            TrainAccuracy = trainAccuracy
            TestAccuracy = testAccuracy
            FinalLoss = finalLoss
            DurationMs = durationMs
            MemoryBytes = memoryBytes
        } : Experiments.RunResult)
    let private gridRuns trainSet testSet =
        let activations = [ Activation.ReLU; Activation.Tanh ]
        let learningRates = [ 0.01; 0.001 ]
        let optimizers =
            [
                ("Adam", Optimizers.Adam(0.001, 0.9, 0.999, 1e-8))
                ("Momentum", Optimizers.Momentum(0.01, 0.9))
                ("SGD", Optimizers.SGD(0.01))
            ]
        let batchSizes = [ 8; 16; 32 ]
        [
            for activation in activations do
                for lr in learningRates do
                    for optName, _ in optimizers do
                        for batchSize in batchSizes do
                            let optimizer =
                                match optName with
                                | "Adam" -> Optimizers.Adam(lr, 0.9, 0.999, 1e-8)
                                | "Momentum" -> Optimizers.Momentum(lr, 0.9)
                                | _ -> Optimizers.SGD(lr)
                            let name = $"Grid-{activationToName activation}-{optName}-lr{lr}-b{batchSize}"
                            let seed = Some (1000 + batchSize + int (lr * 10000.0))
                            yield runSingle name activation lr optimizer batchSize seed trainSet testSet
        ]
    let private randomRuns trainSet testSet count =
        let rnd = Random(42)
        let activations = [| Activation.ReLU; Activation.Tanh; Activation.Sigmoid |]
        let lrs = [| 0.03; 0.01; 0.003; 0.001 |]
        let batchSizes = [| 8; 16; 32 |]
        let optimizers = [| Experiments.OptimizerChoice.Adam; Experiments.OptimizerChoice.Momentum; Experiments.OptimizerChoice.SGD; Experiments.OptimizerChoice.GradientClipping |]
        [
            for i in 1 .. count do
                let activation = activations[rnd.Next(activations.Length)]
                let lr = lrs[rnd.Next(lrs.Length)]
                let batchSize = batchSizes[rnd.Next(batchSizes.Length)]
                let optimizerChoice = optimizers[rnd.Next(optimizers.Length)]
                let optimizer = Experiments.buildOptimizer lr optimizerChoice
                let name = $"Random-{i}"
                let seed = Some (42 + i)
                yield runSingle name activation lr optimizer batchSize seed trainSet testSet
        ]
    let run () =
        printfn "\n========================================"
        printfn "MINI HYPERPARAMETER LAB (IRIS)"
        printfn "========================================"
        let irisPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "Iris.csv")
        if not (File.Exists(irisPath)) then
            printfn "ERROR: Iris dataset not found"
            printfn $"  {irisPath}"
        else
            let fullDataset = Iris.loadDataFromCsv irisPath |> Data.normalize
            let trainSet, testSet = Iris.stratifiedSplit 0.8 fullDataset
            printfn "Running grid search..."
            let grid = gridRuns trainSet testSet
            printfn $"Grid runs completed: {grid.Length}"
            printfn "Running random search..."
            let random = randomRuns trainSet testSet 10
            printfn $"Random runs completed: {random.Length}"
            let allRuns = grid @ random
            let ranked = Experiments.sortRunResults allRuns
            printfn "\nTop 10 runs:"
            ranked
            |> List.truncate 10
            |> List.iteri (fun idx row ->
                printfn $"{idx + 1,2}. {row.Name}: test={row.TestAccuracy * 100.0:N2}%% train={row.TrainAccuracy * 100.0:N2}%% loss={row.FinalLoss:N6} time={row.DurationMs}ms")
            let reportPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "hyperparameter_leaderboard.md")
            let csvPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "hyperparameter_leaderboard.csv")
            Experiments.writeLeaderboardMarkdown reportPath "Hyperparameter Lab Leaderboard" allRuns
            Experiments.writeLeaderboardCsv csvPath allRuns
            printfn $"\nLeaderboard written to: {reportPath}"
            printfn $"CSV written to: {csvPath}"
