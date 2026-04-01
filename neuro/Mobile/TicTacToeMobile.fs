namespace Neuro.Mobile

open System
open System.Diagnostics
open System.IO
open System.Text.Json

type MobileCell =
    | Empty = 0
    | X = 1
    | O = 2

type MobileGameStatus =
    | InProgress = 0
    | XWins = 1
    | OWins = 2
    | Draw = 3

[<CLIMutable>]
type MobileMove = {
    Row: int
    Col: int
}

[<CLIMutable>]
type MobileTrainingProgress = {
    Epoch: int
    TotalEpochs: int
    TrainLoss: float
    ProgressRatio: float
    EpochSeconds: float
    EtaSeconds: float
}

type SerializableLayer() =
    member val Weights: float[][] = Array.empty with get, set
    member val Bias: float[] = Array.empty with get, set
    member val Activation = "" with get, set
    member val InputSize = 0 with get, set
    member val OutputSize = 0 with get, set
    member val Kind = "" with get, set
    member val DropoutRate = 0.0 with get, set

type SerializableModel() =
    member val Layers: SerializableLayer[] = Array.empty with get, set

module private TicTacToeInterop =
    let toDomainCell (value: int) =
        match value with
        | 0 -> Examples.TicTacToe.Empty
        | 1 -> Examples.TicTacToe.X
        | 2 -> Examples.TicTacToe.O
        | _ -> invalidArg (nameof value) "Board values must be 0 (Empty), 1 (X), or 2 (O)."

    let toMobileCellValue cell =
        match cell with
        | Examples.TicTacToe.Empty -> 0
        | Examples.TicTacToe.X -> 1
        | Examples.TicTacToe.O -> 2

    let toDomainBoard (board: int[,]) =
        if board.GetLength(0) <> 3 || board.GetLength(1) <> 3 then
            invalidArg (nameof board) "Board must be a 3x3 matrix."

        Array.init 3 (fun row ->
            Array.init 3 (fun col ->
                toDomainCell board[row, col]))

    let writeBackBoard (source: Examples.TicTacToe.Board) (target: int[,]) =
        for row in 0 .. 2 do
            for col in 0 .. 2 do
                target[row, col] <- toMobileCellValue source.[row].[col]

    let evaluateStatus(board: Examples.TicTacToe.Board) =
        match Examples.TicTacToe.checkWinner board with
        | Some Examples.TicTacToe.X -> MobileGameStatus.XWins
        | Some Examples.TicTacToe.O -> MobileGameStatus.OWins
        | Some Examples.TicTacToe.Empty -> MobileGameStatus.InProgress
        | None when Examples.TicTacToe.isDraw board -> MobileGameStatus.Draw
        | None -> MobileGameStatus.InProgress

    let matrixToJagged (matrix: float[,]) =
        Array.init (matrix.GetLength(0)) (fun row ->
            Array.init (matrix.GetLength(1)) (fun col ->
                matrix[row, col]))

    let jaggedToMatrix (rows: float[][]) =
        let rowCount = rows.Length
        let colCount =
            if rowCount = 0 then 0
            else rows[0].Length

        Array2D.init rowCount colCount (fun row col -> rows[row][col])

    let activationToString activation =
        match activation with
        | Activation.Sigmoid -> "Sigmoid"
        | Activation.ReLU -> "ReLU"
        | Activation.Tanh -> "Tanh"
        | Activation.Softmax -> "Softmax"
        | Activation.Linear -> "Linear"

    let activationFromString value =
        match value with
        | "Sigmoid" -> Activation.Sigmoid
        | "ReLU" -> Activation.ReLU
        | "Tanh" -> Activation.Tanh
        | "Softmax" -> Activation.Softmax
        | "Linear" -> Activation.Linear
        | _ -> invalidArg (nameof value) $"Unsupported activation '{value}'."

    let layerKindToString kind =
        match kind with
        | Layers.Dense -> "Dense"
        | Layers.Dropout -> "Dropout"

    let layerKindFromString value =
        match value with
        | "Dense" -> Layers.Dense
        | "Dropout" -> Layers.Dropout
        | _ -> invalidArg (nameof value) $"Unsupported layer kind '{value}'."

    let serializeNetwork (network: Layers.NeuralNetwork) =
        let model = SerializableModel()
        model.Layers <-
            network
            |> List.map (fun (layer: Layers.DenseLayer) ->
                let serialized = SerializableLayer()
                serialized.Weights <- matrixToJagged layer.Weights
                serialized.Bias <- Array.copy layer.Bias
                serialized.Activation <- activationToString layer.Activation
                serialized.InputSize <- layer.InputSize
                serialized.OutputSize <- layer.OutputSize
                serialized.Kind <- layerKindToString layer.Kind
                serialized.DropoutRate <- layer.DropoutRate
                serialized)
            |> List.toArray
        model

    let deserializeNetwork (model: SerializableModel) : Layers.NeuralNetwork =
        model.Layers
        |> Array.map (fun (layer: SerializableLayer) ->
            ({
                Weights = jaggedToMatrix layer.Weights
                Bias = Array.copy layer.Bias
                Activation = activationFromString layer.Activation
                InputSize = layer.InputSize
                OutputSize = layer.OutputSize
                Kind = layerKindFromString layer.Kind
                DropoutRate = layer.DropoutRate
            }: Layers.DenseLayer))
        |> Array.toList

type TicTacToeModel internal (network: Layers.NeuralNetwork) =
    member internal _.Network = network

    member _.GetAiMove(board: int[,]) =
        let domainBoard = TicTacToeInterop.toDomainBoard board
        let features = Examples.TicTacToe.boardToFeatures domainBoard
        let predictions, _ = Layers.forwardNetwork network features

        let availableMoves =
            [ for row in 0 .. 2 do
                for col in 0 .. 2 do
                    if domainBoard[row][col] = Examples.TicTacToe.Empty then
                        let moveIndex = row * 3 + col
                        yield (row, col, predictions[0, moveIndex]) ]

        match availableMoves with
        | [] ->
            invalidOp "There are no valid moves left for the current board."
        | _ ->
            let row, col, _ = availableMoves |> List.maxBy (fun (_, _, score) -> score)
            { Row = row; Col = col }

    member this.ApplyAiMove(board: int[,]) =
        let move = this.GetAiMove(board)
        let domainBoard = TicTacToeInterop.toDomainBoard board
        let moved = Examples.TicTacToe.makeMove domainBoard move.Row move.Col Examples.TicTacToe.O
        if not moved then
            invalidOp "The model produced an invalid move for the current board."
        TicTacToeInterop.writeBackBoard domainBoard board
        move

[<CLIMutable>]
type MobileTrainingResult = {
    Model: TicTacToeModel
    TrainAccuracy: float
    TestAccuracy: float
    FirstLoss: float
    FinalLoss: float
    EpochsCompleted: int
}

[<AbstractClass; Sealed>]
type TicTacToeMobile private () =
    static member CreateEmptyBoard() : int[,] =
        Array2D.zeroCreate 3 3

    static member TryMakeMove(board: int[,], row: int, col: int, player: MobileCell) =
        let domainBoard = TicTacToeInterop.toDomainBoard board
        let moved = Examples.TicTacToe.makeMove domainBoard row col (TicTacToeInterop.toDomainCell (int player))

        if moved then
            TicTacToeInterop.writeBackBoard domainBoard board

        moved

    static member EvaluateBoard(board: int[,]) =
        board
        |> TicTacToeInterop.toDomainBoard
        |> TicTacToeInterop.evaluateStatus

    static member Train(epochs: int) =
        TicTacToeMobile.Train(epochs, null)

    static member Train(epochs: int, progress: Action<MobileTrainingProgress>) =
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

        let result = Examples.TicTacToe.trainModel epochs callback false
        let firstLoss = List.head result.Metrics.TrainLoss
        let finalLoss = List.last result.Metrics.TrainLoss

        {
            Model = TicTacToeModel(result.Network)
            TrainAccuracy = result.TrainAccuracy
            TestAccuracy = result.TestAccuracy
            FirstLoss = firstLoss
            FinalLoss = finalLoss
            EpochsCompleted = result.Metrics.EpochsCompleted
        }

    static member SaveModel(model: TicTacToeModel, path: string) =
        let directory = Path.GetDirectoryName(path)
        if not (String.IsNullOrWhiteSpace(directory)) then
            Directory.CreateDirectory(directory) |> ignore

        let serialized = TicTacToeInterop.serializeNetwork model.Network
        let json = JsonSerializer.Serialize(serialized, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(path, json)

    static member LoadModel(path: string) =
        let json = File.ReadAllText(path)
        let serialized = JsonSerializer.Deserialize<SerializableModel>(json)

        if isNull (box serialized) then
            invalidOp "Model file is empty or invalid."

        serialized
        |> TicTacToeInterop.deserializeNetwork
        |> TicTacToeModel

    static member ModelExists(path: string) =
        File.Exists(path)

    static member DeleteModel(path: string) =
        if File.Exists(path) then
            File.Delete(path)
