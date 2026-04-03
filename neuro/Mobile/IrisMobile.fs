namespace Neuro.Mobile

open System
open System.Diagnostics
open System.IO
open System.Text.Json

type SerializableIrisModel() =
    member val Layers: SerializableLayer[] = Array.empty with get, set
    member val Mean: float[] = Array.empty with get, set
    member val Std: float[] = Array.empty with get, set

[<CLIMutable>]
type IrisPredictionResult = {
    PredictedClassIndex: int
    PredictedClassName: string
    Probabilities: float[]
}

type IrisModel internal (network: Layers.NeuralNetwork, normalization: Data.NormalizationStats) =
    member internal _.Network = network
    member internal _.Normalization = normalization

    member _.Predict(features: float[]) =
        let prediction = Examples.Iris.predictSample network normalization features
        {
            PredictedClassIndex = prediction.PredictedClassIndex
            PredictedClassName = prediction.PredictedClassName
            Probabilities = prediction.Probabilities
        }

[<CLIMutable>]
type IrisTrainingResultMobile = {
    Model: IrisModel
    TrainAccuracy: float
    ValidationAccuracy: float
    FirstLoss: float
    FinalLoss: float
    EpochsCompleted: int
}

[<AbstractClass; Sealed>]
type IrisMobile private () =
    static member Train(lines: string[], epochs: int, batchSize: int, learningRate: float, progress: Action<MobileTrainingProgress>) =
        let dataset = Examples.Iris.loadDataFromLines lines
        let stopwatch = Stopwatch.StartNew()
        let mutable previousMark = stopwatch.Elapsed

        let callback =
            if isNull progress then
                None
            else
                Some (fun (epoch: Trainer.EpochProgress) ->
                    let currentMark = stopwatch.Elapsed
                    let epochSeconds = (currentMark - previousMark).TotalSeconds
                    previousMark <- currentMark
                    let averageEpochSeconds = currentMark.TotalSeconds / float epoch.Epoch
                    let etaSeconds = max 0.0 (float (epoch.TotalEpochs - epoch.Epoch) * averageEpochSeconds)

                    progress.Invoke({
                        Epoch = epoch.Epoch
                        TotalEpochs = epoch.TotalEpochs
                        TrainLoss = epoch.TrainLoss
                        ProgressRatio = float epoch.Epoch / float epoch.TotalEpochs
                        EpochSeconds = epochSeconds
                        EtaSeconds = etaSeconds
                    }))

        let result = Examples.Iris.trainModel dataset epochs batchSize learningRate callback false
        let firstLoss = List.head result.Metrics.TrainLoss
        let finalLoss = List.last result.Metrics.TrainLoss

        {
            Model = IrisModel(result.Network, result.Normalization)
            TrainAccuracy = result.TrainAccuracy
            ValidationAccuracy = result.ValidationAccuracy
            FirstLoss = firstLoss
            FinalLoss = finalLoss
            EpochsCompleted = result.Metrics.EpochsCompleted
        }

    static member SaveModel(model: IrisModel, path: string) =
        let directory = Path.GetDirectoryName(path)
        if not (String.IsNullOrWhiteSpace(directory)) then
            Directory.CreateDirectory(directory) |> ignore

        let serialized = SerializableIrisModel()
        serialized.Layers <- (MobileInterop.serializeNetwork model.Network).Layers
        serialized.Mean <- Array.copy model.Normalization.Mean
        serialized.Std <- Array.copy model.Normalization.Std
        let json = JsonSerializer.Serialize(serialized, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(path, json)

    static member LoadModel(path: string) =
        let json = File.ReadAllText(path)
        let serialized = JsonSerializer.Deserialize<SerializableIrisModel>(json)
        if isNull (box serialized) then
            invalidOp "Model file is empty or invalid."

        let denseModel = SerializableModel()
        denseModel.Layers <- serialized.Layers
        let normalization: Data.NormalizationStats = {
            Mean = Array.copy serialized.Mean
            Std = Array.copy serialized.Std
        }
        IrisModel(MobileInterop.deserializeNetwork denseModel, normalization)

    static member ModelExists(path: string) =
        File.Exists(path)

    static member DeleteModel(path: string) =
        if File.Exists(path) then
            File.Delete(path)
