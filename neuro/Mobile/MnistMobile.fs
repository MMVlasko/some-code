namespace Neuro.Mobile

open System
open System.Diagnostics
open System.IO
open System.Text.Json

[<CLIMutable>]
type MnistPredictionResult = {
    PredictedDigit: int
    Confidence: float
    Probabilities: float[]
}

type MnistModel internal (network: Layers.NeuralNetwork) =
    member internal _.Network = network

    member _.Predict(pixels: float[]) =
        let prediction = Examples.MNIST.predictDigit network pixels
        {
            PredictedDigit = prediction.PredictedDigit
            Confidence = prediction.Confidence
            Probabilities = prediction.Probabilities
        }

[<CLIMutable>]
type MnistTrainingResultMobile = {
    Model: MnistModel
    TrainAccuracy: float
    TestAccuracy: float
    FirstLoss: float
    FinalLoss: float
    EpochsCompleted: int
}

[<AbstractClass; Sealed>]
type MnistMobile private () =
    static member Train(trainLines: string[], testLines: string[], epochs: int, batchSize: int, learningRate: float, progress: Action<MobileTrainingProgress>) =
        let trainSet = Examples.MNIST.loadMnistFromLines trainLines None false
        let testSet = Examples.MNIST.loadMnistFromLines testLines None false
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

        let result = Examples.MNIST.trainModel trainSet testSet epochs batchSize learningRate callback false
        let firstLoss = List.head result.Metrics.TrainLoss
        let finalLoss = List.last result.Metrics.TrainLoss

        {
            Model = MnistModel(result.Network)
            TrainAccuracy = result.TrainAccuracy
            TestAccuracy = result.TestAccuracy
            FirstLoss = firstLoss
            FinalLoss = finalLoss
            EpochsCompleted = result.Metrics.EpochsCompleted
        }

    static member SaveModel(model: MnistModel, path: string) =
        let directory = Path.GetDirectoryName(path)
        if not (String.IsNullOrWhiteSpace(directory)) then
            Directory.CreateDirectory(directory) |> ignore

        let serialized = MobileInterop.serializeNetwork model.Network
        let json = JsonSerializer.Serialize(serialized, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(path, json)

    static member LoadModel(path: string) =
        let json = File.ReadAllText(path)
        let serialized = JsonSerializer.Deserialize<SerializableModel>(json)

        if isNull (box serialized) then
            invalidOp "Model file is empty or invalid."

        serialized
        |> MobileInterop.deserializeNetwork
        |> MnistModel

    static member ModelExists(path: string) =
        File.Exists(path)

    static member DeleteModel(path: string) =
        if File.Exists(path) then
            File.Delete(path)
