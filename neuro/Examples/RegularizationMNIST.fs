namespace Examples
open System
open System.IO
module RegularizationMNIST =
    type RegConfig = {
        Name: string
        LabelSmoothing: float
        WeightDecay: float
        Mixup: bool
        Cutmix: bool
    }
    let private applyLabelSmoothing epsilon (labels: float[,]) =
        if epsilon <= 0.0 then
            labels
        else
            let rows = labels.GetLength(0)
            let cols = labels.GetLength(1)
            let offValue = epsilon / float cols
            Array2D.init rows cols (fun i j -> (1.0 - epsilon) * labels[i, j] + offValue)
    let private applyMixup (rng: Random) (dataset: Data.Dataset) =
        let rows = dataset.Features.GetLength(0)
        let featuresCols = dataset.Features.GetLength(1)
        let labelCols = dataset.Labels.GetLength(1)
        let mixedFeatures = Array2D.zeroCreate rows featuresCols
        let mixedLabels = Array2D.zeroCreate rows labelCols
        for i in 0 .. rows - 1 do
            let j = rng.Next(rows)
            let lambda = 0.3 + rng.NextDouble() * 0.4
            for c in 0 .. featuresCols - 1 do
                mixedFeatures[i, c] <- lambda * dataset.Features[i, c] + (1.0 - lambda) * dataset.Features[j, c]
            for c in 0 .. labelCols - 1 do
                mixedLabels[i, c] <- lambda * dataset.Labels[i, c] + (1.0 - lambda) * dataset.Labels[j, c]
        Data.createDataset mixedFeatures mixedLabels
    let private applyCutmix28x28 (rng: Random) (dataset: Data.Dataset) =
        let rows = dataset.Features.GetLength(0)
        let featuresCols = dataset.Features.GetLength(1)
        let labelCols = dataset.Labels.GetLength(1)
        if featuresCols <> 784 then
            failwith "Cutmix-style augmentation in this demo expects 28x28 flattened inputs"
        let mixedFeatures = Array2D.init rows featuresCols (fun i j -> dataset.Features[i, j])
        let mixedLabels = Array2D.init rows labelCols (fun i j -> dataset.Labels[i, j])
        let patch (row: int) (col: int) = row * 28 + col
        for i in 0 .. rows - 1 do
            let j = rng.Next(rows)
            let lambda = 0.3 + rng.NextDouble() * 0.4
            let cutRatio = sqrt (1.0 - lambda)
            let cutW = max 1 (int (28.0 * cutRatio))
            let cutH = max 1 (int (28.0 * cutRatio))
            let cx = rng.Next(28)
            let cy = rng.Next(28)
            let x1 = max 0 (cx - cutW / 2)
            let x2 = min 27 (cx + cutW / 2)
            let y1 = max 0 (cy - cutH / 2)
            let y2 = min 27 (cy + cutH / 2)
            for y in y1 .. y2 do
                for x in x1 .. x2 do
                    mixedFeatures[i, patch y x] <- dataset.Features[j, patch y x]
            let area = float ((x2 - x1 + 1) * (y2 - y1 + 1))
            let lamAdjusted = 1.0 - area / (28.0 * 28.0)
            for c in 0 .. labelCols - 1 do
                mixedLabels[i, c] <- lamAdjusted * dataset.Labels[i, c] + (1.0 - lamAdjusted) * dataset.Labels[j, c]
        Data.createDataset mixedFeatures mixedLabels
    let private buildTrainDataset (baseTrain: Data.Dataset) (cfg: RegConfig) (seed: int option) =
        let mixupSeed = seed |> Option.defaultValue 42
        let cutmixSeed = mixupSeed + 97
        let labels = applyLabelSmoothing cfg.LabelSmoothing baseTrain.Labels
        let initial = Data.createDataset baseTrain.Features labels
        let withMixup =
            if cfg.Mixup then applyMixup (Random(mixupSeed)) initial else initial
        if cfg.Cutmix then applyCutmix28x28 (Random(cutmixSeed)) withMixup else withMixup

    let internal buildTrainDatasetForTesting (baseTrain: Data.Dataset) (cfg: RegConfig) =
        buildTrainDataset baseTrain cfg (Some 42)
    let private runSingle (cfg: RegConfig) (seed: int option) (trainSet: Data.Dataset) (testSet: Data.Dataset) : Experiments.RunResult =
        let transformedTrain = buildTrainDataset trainSet cfg seed
        let network =
            [
                Layers.createLayer 784 128 Activation.ReLU
                Layers.createLayer 128 64 Activation.ReLU
                Layers.createLayer 64 10 Activation.Softmax
            ]
        let trainingConfig =
            {
                Trainer.defaultConfig with
                    Epochs = 4
                    BatchSize = 64
                    Optimizer = Optimizers.Adam(0.001, 0.9, 0.999, 1e-8)
                    Loss = Losses.CrossEntropy
                    Verbose = false
                    ValidationSplit = None
                    WeightDecay = cfg.WeightDecay
                    RandomSeed = seed
            }
        let (metrics, trained), durationMs, memoryBytes =
            Experiments.measureRun (fun () -> Trainer.train trainingConfig network transformedTrain)
        ({
            RunId = Experiments.createRunId "reg"
            Name = cfg.Name
            Params = $"smooth={cfg.LabelSmoothing}; wd={cfg.WeightDecay}; mixup={cfg.Mixup}; cutmix={cfg.Cutmix}"
            Seed = seed
            TrainAccuracy = Trainer.accuracy trained transformedTrain
            TestAccuracy = Trainer.accuracy trained testSet
            FinalLoss = metrics.TrainLoss |> List.last
            DurationMs = durationMs
            MemoryBytes = memoryBytes
        } : Experiments.RunResult)
    let run () =
        printfn "\n========================================"
        printfn "MNIST REGULARIZATION EXPERIMENTS"
        printfn "========================================"
        let trainPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_train.csv")
        let testPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_test.csv")
        if not (File.Exists(trainPath)) || not (File.Exists(testPath)) then
            printfn "ERROR: MNIST CSV files were not found in Examples/Data"
        else
            let trainSet = MNIST.loadMnistFromCsv trainPath (Some 2000)
            let testSet = MNIST.loadMnistFromCsv testPath (Some 400)
            let configs =
                [
                    { Name = "Baseline"; LabelSmoothing = 0.0; WeightDecay = 0.0; Mixup = false; Cutmix = false }
                    { Name = "LabelSmoothing"; LabelSmoothing = 0.1; WeightDecay = 0.0; Mixup = false; Cutmix = false }
                    { Name = "WeightDecay"; LabelSmoothing = 0.0; WeightDecay = 1e-4; Mixup = false; Cutmix = false }
                    { Name = "MixupStyle"; LabelSmoothing = 0.0; WeightDecay = 0.0; Mixup = true; Cutmix = false }
                    { Name = "CutmixStyle"; LabelSmoothing = 0.0; WeightDecay = 0.0; Mixup = false; Cutmix = true }
                    { Name = "Combo"; LabelSmoothing = 0.1; WeightDecay = 1e-4; Mixup = true; Cutmix = false }
                ]
            let results =
                configs
                |> List.mapi (fun idx cfg ->
                    printfn $"Running {cfg.Name}..."
                    let seed = Some (3000 + idx)
                    runSingle cfg seed trainSet testSet)
            let ranked = Experiments.sortRunResults results
            printfn "\nRegularization ranking:"
            ranked
            |> List.iteri (fun idx row ->
                printfn $"{idx + 1,2}. {row.Name}: test={row.TestAccuracy * 100.0:N2}%% train={row.TrainAccuracy * 100.0:N2}%%")
            let reportPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "regularization_leaderboard.md")
            let csvPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "regularization_leaderboard.csv")
            Experiments.writeLeaderboardMarkdown reportPath "MNIST Regularization Leaderboard" results
            Experiments.writeLeaderboardCsv csvPath results
            printfn $"\nReport written to: {reportPath}"
            printfn $"CSV written to: {csvPath}"
