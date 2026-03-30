namespace Examples
open System
open System.IO
module BenchmarkHarness =
    let private runIrisBenchmark () : Experiments.BenchmarkResult list =
        let irisPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "Iris.csv")
        if not (File.Exists(irisPath)) then
            []
        else
            let data = Iris.loadDataFromCsv irisPath |> Data.normalize
            let trainSet, testSet = Iris.stratifiedSplit 0.8 data
            let layerPresets =
                [
                    ("Shallow", [ Layers.createLayer 4 8 Activation.ReLU; Layers.createLayer 8 3 Activation.Softmax ])
                    ("Deeper", [ Layers.createLayer 4 16 Activation.Tanh; Layers.createLayer 16 8 Activation.Tanh; Layers.createLayer 8 3 Activation.Softmax ])
                ]
            let optimizers =
                [
                    ("SGD", Optimizers.SGD 0.01)
                    ("Momentum", Optimizers.Momentum(0.01, 0.9))
                    ("Adam", Optimizers.Adam(0.001, 0.9, 0.999, 1e-8))
                ]
            [
                for layerIdx, (layerPresetName, network) in layerPresets |> List.indexed do
                    for optIdx, (optimizerName, optimizer) in optimizers |> List.indexed do
                        let seed = Some (5000 + layerIdx * 100 + optIdx)
                        let cfg =
                            {
                                Trainer.defaultConfig with
                                    Epochs = 180
                                    BatchSize = 16
                                    Optimizer = optimizer
                                    Loss = Losses.CrossEntropy
                                    Verbose = false
                                    RandomSeed = seed
                            }
                        let (metrics, trained), durationMs, memoryBytes =
                            Experiments.measureRun (fun () -> Trainer.train cfg network trainSet)
                        yield
                            ({
                                RunId = Experiments.createRunId "bench"
                                Scenario = "Iris"
                                Optimizer = optimizerName
                                LayerPreset = layerPresetName
                                Seed = seed
                                TrainAccuracy = Trainer.accuracy trained trainSet
                                TestAccuracy = Trainer.accuracy trained testSet
                                FinalLoss = metrics.TrainLoss |> List.last
                                DurationMs = durationMs
                                MemoryBytes = memoryBytes
                            } : Experiments.BenchmarkResult)
            ]
    let private runMnistBenchmark () : Experiments.BenchmarkResult list =
        let trainPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_train.csv")
        let testPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "mnist_test.csv")
        if not (File.Exists(trainPath)) || not (File.Exists(testPath)) then
            []
        else
            let trainSet = MNIST.loadMnistFromCsv trainPath (Some 3000)
            let testSet = MNIST.loadMnistFromCsv testPath (Some 800)
            let layerPresets =
                [
                    ("Compact", [ Layers.createLayer 784 64 Activation.ReLU; Layers.createLayer 64 10 Activation.Softmax ])
                    ("Balanced", [ Layers.createLayer 784 128 Activation.ReLU; Layers.createLayer 128 64 Activation.ReLU; Layers.createLayer 64 10 Activation.Softmax ])
                ]
            let optimizers =
                [
                    ("SGD", Optimizers.SGD 0.01)
                    ("Momentum", Optimizers.Momentum(0.01, 0.9))
                    ("Adam", Optimizers.Adam(0.001, 0.9, 0.999, 1e-8))
                ]
            [
                for layerIdx, (layerPresetName, network) in layerPresets |> List.indexed do
                    for optIdx, (optimizerName, optimizer) in optimizers |> List.indexed do
                        let seed = Some (6000 + layerIdx * 100 + optIdx)
                        let cfg =
                            {
                                Trainer.defaultConfig with
                                    Epochs = 8
                                    BatchSize = 64
                                    Optimizer = optimizer
                                    Loss = Losses.CrossEntropy
                                    Verbose = false
                                    RandomSeed = seed
                            }
                        let (metrics, trained), durationMs, memoryBytes =
                            Experiments.measureRun (fun () -> Trainer.train cfg network trainSet)
                        yield
                            ({
                                RunId = Experiments.createRunId "bench"
                                Scenario = "MNIST"
                                Optimizer = optimizerName
                                LayerPreset = layerPresetName
                                Seed = seed
                                TrainAccuracy = Trainer.accuracy trained trainSet
                                TestAccuracy = Trainer.accuracy trained testSet
                                FinalLoss = metrics.TrainLoss |> List.last
                                DurationMs = durationMs
                                MemoryBytes = memoryBytes
                            } : Experiments.BenchmarkResult)
            ]
    let run () =
        printfn "\n========================================"
        printfn "BENCHMARK HARNESS"
        printfn "========================================"
        printfn "Running Iris benchmark matrix..."
        let irisRows = runIrisBenchmark ()
        printfn $"Iris runs: {irisRows.Length}"
        printfn "Running MNIST benchmark matrix..."
        let mnistRows = runMnistBenchmark ()
        printfn $"MNIST runs: {mnistRows.Length}"
        let allRows = irisRows @ mnistRows
        if allRows.IsEmpty then
            printfn "No benchmark rows produced (missing datasets)."
        else
            let ranked =
                allRows
                |> List.sortWith (fun a b ->
                    if a.TestAccuracy = b.TestAccuracy then compare a.DurationMs b.DurationMs
                    else compare b.TestAccuracy a.TestAccuracy)
            printfn "\nTop benchmark rows:"
            ranked
            |> List.truncate 10
            |> List.iteri (fun idx row ->
                printfn $"{idx + 1,2}. {row.Scenario}-{row.LayerPreset}-{row.Optimizer}: test={row.TestAccuracy * 100.0:N2}%% time={row.DurationMs}ms")
            let reportPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "benchmark_report.md")
            let csvPath = Path.Combine(__SOURCE_DIRECTORY__, "Reports", "benchmark_report.csv")
            Experiments.writeBenchmarkMarkdown reportPath "Benchmark Harness Report" allRows
            Experiments.writeBenchmarkCsv csvPath allRows
            printfn $"\nBenchmark report written to: {reportPath}"
            printfn $"Benchmark CSV written to: {csvPath}"
