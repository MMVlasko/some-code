module Experiments
open System
open System.Diagnostics
open System.IO
open System.Text
type OptimizerChoice =
    | SGD
    | Momentum
    | Adam
    | GradientClipping
type RunResult = {
    Name: string
    Params: string
    TrainAccuracy: float
    TestAccuracy: float
    FinalLoss: float
    DurationMs: int64
    MemoryBytes: int64
}
type BenchmarkResult = {
    Scenario: string
    Optimizer: string
    LayerPreset: string
    TrainAccuracy: float
    TestAccuracy: float
    FinalLoss: float
    DurationMs: int64
    MemoryBytes: int64
}
type ClassMetrics = {
    Precision: float
    Recall: float
    F1: float
    Support: int
}
let optimizerToName = function
    | OptimizerChoice.SGD -> "SGD"
    | OptimizerChoice.Momentum -> "Momentum"
    | OptimizerChoice.Adam -> "Adam"
    | OptimizerChoice.GradientClipping -> "GradientClipping"
let buildOptimizer lr choice =
    match choice with
    | OptimizerChoice.SGD -> Optimizers.SGD lr
    | OptimizerChoice.Momentum -> Optimizers.Momentum (lr, 0.9)
    | OptimizerChoice.Adam -> Optimizers.Adam (lr, 0.9, 0.999, 1e-8)
    | OptimizerChoice.GradientClipping -> Optimizers.GradientClipping (lr, 0.5)
let ensureDirectoryForFile (filePath: string) =
    let dir = Path.GetDirectoryName(filePath)
    if String.IsNullOrWhiteSpace(dir) |> not then
        Directory.CreateDirectory(dir) |> ignore
let sortRunResults (results: RunResult list) =
    results
    |> List.sortWith (fun a b ->
        if a.TestAccuracy = b.TestAccuracy then compare a.DurationMs b.DurationMs
        else compare b.TestAccuracy a.TestAccuracy)
let writeLeaderboardMarkdown (filePath: string) (title: string) (results: RunResult list) =
    ensureDirectoryForFile filePath
    let ranked = sortRunResults results
    let sb = StringBuilder()
    sb.AppendLine($"# {title}") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine("| Rank | Name | Params | Train Acc | Test Acc | Final Loss | Duration (ms) | Memory (MB) |") |> ignore
    sb.AppendLine("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |") |> ignore
    ranked
    |> List.iteri (fun idx row ->
        let memMb = float row.MemoryBytes / (1024.0 * 1024.0)
        let line =
            sprintf
                "| %d | %s | `%s` | %.2f%% | %.2f%% | %.6f | %d | %.2f |"
                (idx + 1)
                row.Name
                row.Params
                (row.TrainAccuracy * 100.0)
                (row.TestAccuracy * 100.0)
                row.FinalLoss
                row.DurationMs
                memMb
        sb.AppendLine(line) |> ignore)
    File.WriteAllText(filePath, sb.ToString())
let writeBenchmarkMarkdown (filePath: string) (title: string) (rows: BenchmarkResult list) =
    ensureDirectoryForFile filePath
    let ranked =
        rows
        |> List.sortWith (fun a b ->
            if a.TestAccuracy = b.TestAccuracy then compare a.DurationMs b.DurationMs
            else compare b.TestAccuracy a.TestAccuracy)
    let sb = StringBuilder()
    sb.AppendLine($"# {title}") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine("| Rank | Scenario | Optimizer | Layer Preset | Train Acc | Test Acc | Final Loss | Duration (ms) | Memory (MB) |") |> ignore
    sb.AppendLine("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |") |> ignore
    ranked
    |> List.iteri (fun idx row ->
        let memMb = float row.MemoryBytes / (1024.0 * 1024.0)
        let line =
            sprintf
                "| %d | %s | %s | %s | %.2f%% | %.2f%% | %.6f | %d | %.2f |"
                (idx + 1)
                row.Scenario
                row.Optimizer
                row.LayerPreset
                (row.TrainAccuracy * 100.0)
                (row.TestAccuracy * 100.0)
                row.FinalLoss
                row.DurationMs
                memMb
        sb.AppendLine(line) |> ignore)
    File.WriteAllText(filePath, sb.ToString())
let measureRun (f: unit -> 'T) =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let beforeMem = GC.GetTotalMemory(true)
    let sw = Stopwatch.StartNew()
    let value = f()
    sw.Stop()
    let afterMem = GC.GetTotalMemory(true)
    value, sw.ElapsedMilliseconds, max 0L (afterMem - beforeMem)
let argmaxRow (matrix: float[,]) rowIdx =
    [| 0 .. matrix.GetLength(1) - 1 |]
    |> Array.maxBy (fun j -> matrix[rowIdx, j])
let classesFromOneHot (labels: float[,]) =
    Array.init (labels.GetLength(0)) (fun i -> argmaxRow labels i)
let classesFromPredictions (predictions: float[,]) =
    Array.init (predictions.GetLength(0)) (fun i -> argmaxRow predictions i)
let confusionMatrix classCount (predictions: int[]) (targets: int[]) =
    if predictions.Length <> targets.Length then
        failwith "Predictions and targets must have the same length"
    let matrix = Array2D.zeroCreate classCount classCount
    for i in 0 .. predictions.Length - 1 do
        let t = targets[i]
        let p = predictions[i]
        matrix[t, p] <- matrix[t, p] + 1
    matrix
let perClassMetrics (cm: int[,]) =
    let classCount = cm.GetLength(0)
    Array.init classCount (fun c ->
        let tp = float cm[c, c]
        let fp =
            [| 0 .. classCount - 1 |]
            |> Array.sumBy (fun r -> if r = c then 0 else cm[r, c])
            |> float
        let fn =
            [| 0 .. classCount - 1 |]
            |> Array.sumBy (fun col -> if col = c then 0 else cm[c, col])
            |> float
        let support =
            [| 0 .. classCount - 1 |]
            |> Array.sumBy (fun col -> cm[c, col])
        let precision = if tp + fp = 0.0 then 0.0 else tp / (tp + fp)
        let recall = if tp + fn = 0.0 then 0.0 else tp / (tp + fn)
        let f1 = if precision + recall = 0.0 then 0.0 else 2.0 * precision * recall / (precision + recall)
        { Precision = precision; Recall = recall; F1 = f1; Support = support })
let writeClassificationMarkdown (filePath: string) (title: string) (classNames: string[]) (cm: int[,]) =
    ensureDirectoryForFile filePath
    let metrics = perClassMetrics cm
    let classCount = cm.GetLength(0)
    let sb = StringBuilder()
    sb.AppendLine($"# {title}") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine("## Confusion Matrix") |> ignore
    sb.AppendLine() |> ignore
    let headerClasses = classNames |> String.concat " | "
    sb.AppendLine($"| True \\ Pred | {headerClasses} |") |> ignore
    let divider = String.replicate classCount "--- | "
    sb.AppendLine($"| --- | {divider}") |> ignore
    for r in 0 .. classCount - 1 do
        let row = [| for c in 0 .. classCount - 1 -> string cm[r, c] |] |> String.concat " | "
        sb.AppendLine($"| {classNames[r]} | {row} |") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine("## Per-Class Metrics") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine("| Class | Precision | Recall | F1 | Support |") |> ignore
    sb.AppendLine("| --- | ---: | ---: | ---: | ---: |") |> ignore
    for i in 0 .. classCount - 1 do
        let m = metrics[i]
        sb.AppendLine($"| {classNames[i]} | {m.Precision:N4} | {m.Recall:N4} | {m.F1:N4} | {m.Support} |") |> ignore
    File.WriteAllText(filePath, sb.ToString())
let writeSaliencyAsPgm (filePath: string) width height (values: float[]) =
    ensureDirectoryForFile filePath
    if values.Length <> width * height then
        failwith "Saliency vector length does not match width*height"
    let maxValue = values |> Array.max
    let scale = if maxValue <= 1e-12 then 1.0 else 255.0 / maxValue
    use writer = new StreamWriter(filePath, false, Encoding.ASCII)
    writer.WriteLine("P2")
    writer.WriteLine($"{width} {height}")
    writer.WriteLine("255")
    for y in 0 .. height - 1 do
        let row =
            [| for x in 0 .. width - 1 ->
                let idx = y * width + x
                let v = int (Math.Round(values[idx] * scale))
                string (max 0 (min 255 v)) |]
            |> String.concat " "
        writer.WriteLine(row)
let cnnInputSaliency (network: ConvLayers.CNNNetwork) (sample: float[]) channels height width targetClass =
    let features = Array2D.init 1 sample.Length (fun _ j -> sample[j])
    let inputTensor = ConvLayers.matrixToTensor features channels height width
    let output, caches = ConvLayers.forwardNetwork network (ConvLayers.Tensor4D inputTensor)
    let predictions =
        match output with
        | ConvLayers.Matrix m -> m
        | _ -> failwith "CNN saliency requires matrix output"
    let gradOutput = Array2D.zeroCreate 1 (predictions.GetLength(1))
    gradOutput[0, targetClass] <- 1.0
    let gradInput, _ = ConvLayers.backwardNetwork network caches (ConvLayers.Matrix gradOutput)
    match gradInput with
    | ConvLayers.Tensor4D tensorGrad ->
        let saliency = Array.zeroCreate (channels * height * width)
        for c in 0 .. channels - 1 do
            for h in 0 .. height - 1 do
                for w in 0 .. width - 1 do
                    let idx = c * height * width + h * width + w
                    saliency[idx] <- abs tensorGrad[0, c, h, w]
        saliency
    | _ -> failwith "Unexpected gradient type for CNN saliency"
