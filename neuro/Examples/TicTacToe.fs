// Examples/TicTacToe.fs
namespace Examples

open System

module TicTacToe =
    type Cell = 
        | Empty 
        | X 
        | O
    
    type Board = Cell array array

    type TrainingResult = {
        Network: Layers.NeuralNetwork
        Metrics: Trainer.TrainingMetrics
        TrainAccuracy: float
        TestAccuracy: float
    }
    
    let createEmptyBoard () : Board =
        Array.init 3 (fun _ -> Array.init 3 (fun _ -> Empty))
    
    let boardToString (board: Board) =
        let cellToChar = function
            | Empty -> "."
            | X -> "X"
            | O -> "O"
        
        [ for row in 0..2 ->
            [ for col in 0..2 -> cellToChar board[row].[col] ]
            |> String.concat " " ]
        |> String.concat "\n"
    
    let boardToFeatures (board: Board) =
        let features = Array.zeroCreate 27
        
        for row in 0..2 do
            for col in 0..2 do
                let idx = row * 3 + col
                match board[row].[col] with
                | Empty -> features[idx * 3] <- 1.0
                | X -> features[idx * 3 + 1] <- 1.0
                | O -> features[idx * 3 + 2] <- 1.0
        
        Array2D.init 1 27 (fun _ j -> features[j])
    
    let checkWinner (board: Board) =
        let lines = [
            [(0,0); (0,1); (0,2)]
            [(1,0); (1,1); (1,2)]
            [(2,0); (2,1); (2,2)]
            [(0,0); (1,0); (2,0)]
            [(0,1); (1,1); (2,1)]
            [(0,2); (1,2); (2,2)]
            [(0,0); (1,1); (2,2)]
            [(0,2); (1,1); (2,0)]
        ]
        
        lines
        |> List.tryPick (fun line ->
            let cells = line |> List.map (fun (r,c) -> board[r][c])
            if List.forall ((=) X) cells then Some X
            elif List.forall ((=) O) cells then Some O
            else None)
    
    let isDraw (board: Board) =
        board |> Array.forall (fun row -> row |> Array.forall ((<>) Empty))
    
    let gameOver (board: Board) =
        match checkWinner board with
        | Some _ -> true
        | None -> isDraw board
    
    let minimax (board: Board) (isMaximizing: bool) =
        let rec score board isMaximizing =
            match checkWinner board with
            | Some X -> -10
            | Some O -> 10
            | None when isDraw board -> 0
            | _ ->
                let moves = 
                    [ for r in 0..2 do
                        for c in 0..2 do
                            if board[r].[c] = Empty then (r, c) ]
                
                if isMaximizing then
                    moves
                    |> List.map (fun (r,c) ->
                        let newBoard = Array.map Array.copy board
                        newBoard[r].[c] <- O
                        score newBoard false)
                    |> List.max
                else
                    moves
                    |> List.map (fun (r,c) ->
                        let newBoard = Array.map Array.copy board
                        newBoard[r].[c] <- X
                        score newBoard true)
                    |> List.min
        
        score board isMaximizing
    
    let generatePositionsForPlayer (player: Cell) =
        let rec generate board positions =
            if gameOver board then
                positions
            else
                let moves = 
                    [ for r in 0..2 do
                        for c in 0..2 do
                            if board[r].[c] = Empty then (r, c) ]
                
                let xCount = 
                    board |> Array.sumBy (fun row -> 
                        row |> Array.sumBy (fun cell -> if cell = X then 1 else 0))
                let oCount = 
                    board |> Array.sumBy (fun row -> 
                        row |> Array.sumBy (fun cell -> if cell = O then 1 else 0))
                
                let currentPlayer = if xCount = oCount then X else O
                
                let newPositions =
                    moves
                    |> List.collect (fun (r,c) ->
                        let newBoard = Array.map Array.copy board
                        newBoard[r].[c] <- currentPlayer
                        generate newBoard positions)
                
                if currentPlayer = player then
                    let isMaximizing = (player = O)
                    let bestMove = 
                        moves
                        |> List.map (fun (r,c) ->
                            let newBoard = Array.map Array.copy board
                            newBoard[r].[c] <- player
                            (r, c), minimax newBoard (not isMaximizing))
                        |> (if isMaximizing then List.maxBy else List.minBy) snd
                        |> fst
                    (board, bestMove) :: newPositions
                else
                    newPositions
        
        generate (createEmptyBoard ()) []
    
    let private createDatasetForPlayerWithOutput (player: Cell) (showSamples: bool) =
        let allPositions = 
            generatePositionsForPlayer player
            |> List.distinctBy (fun (board,_) -> 
                let features = boardToFeatures board
                features[0,0..] |> Seq.map int |> Seq.toArray)
        
        let symbol = if player = X then "X" else "O"
        if showSamples then
            printfn $"Generated {List.length allPositions} unique positions for player {symbol}"
           
            printfn "\n=== SAMPLE TRAINING POSITIONS ==="
            for i in 0 .. min 4 (List.length allPositions - 1) do
                let board, (row, col) = allPositions[i]
                printfn $"Position {i+1}:"
                printfn $"{boardToString board}"
                printfn $"Optimal move for {symbol}: row={row}, col={col}"
                printfn ""
        
        let features = Array2D.zeroCreate (List.length allPositions) 27
        let labels = Array2D.zeroCreate (List.length allPositions) 9
        
        allPositions
        |> List.iteri (fun i (board, (row, col)) ->
            let boardFeatures = boardToFeatures board
            for j in 0..26 do
                features[i, j] <- boardFeatures[0, j]
            
            let moveIndex = row * 3 + col
            labels[i, moveIndex] <- 1.0)
        
        Data.createDataset features labels

    let createDatasetForPlayer (player: Cell) =
        createDatasetForPlayerWithOutput player true

    let createDatasetForPlayerSilent (player: Cell) =
        createDatasetForPlayerWithOutput player false

    let createTrainingNetwork () =
        [
            Layers.createLayer 27 128 Activation.ReLU
            Layers.createLayer 128 64 Activation.ReLU
            Layers.createLayer 64 9 Activation.Softmax
        ]

    let trainModel (epochs: int) (onEpochEnd: (Trainer.EpochProgress -> unit) option) (verbose: bool) =
        let dataset = createDatasetForPlayerSilent O
        let trainSet, testSet = Data.split 0.9 dataset
        let network = createTrainingNetwork ()

        let config = {
            Trainer.defaultConfig with
                Epochs = epochs
                BatchSize = 64
                Optimizer = Optimizers.Adam(0.001, 0.9, 0.999, 1e-8)
                Loss = Losses.CrossEntropy
                Verbose = verbose
                OnEpochEnd = onEpochEnd
                ValidationSplit = None
        }

        let metrics, trainedNetwork = Trainer.train config network trainSet
        let trainAccuracy = Trainer.accuracy trainedNetwork trainSet
        let testAccuracy = Trainer.accuracy trainedNetwork testSet

        {
            Network = trainedNetwork
            Metrics = metrics
            TrainAccuracy = trainAccuracy
            TestAccuracy = testAccuracy
        }

    let trainAI (epochs: int) =
        printfn "\n========================================"
        printfn "TIC-TAC-TOE AI TRAINING"
        printfn "========================================\n"
        
        printfn "Generating training positions for O (AI player)..."
        let dataset = createDatasetForPlayer O
        let trainSet, testSet = Data.split 0.9 dataset
        
        printfn $"Training samples: {trainSet.Features.GetLength(0)}"
        printfn $"Test samples: {testSet.Features.GetLength(0)}"
        printfn $"Features per sample: 27 (one-hot encoding of 3x3 board)"
        printfn $"Output classes: 9 (possible moves)"
        printfn ""
        
        let network = createTrainingNetwork ()
        
        printfn "=== NETWORK ARCHITECTURE ==="
        printfn "  Input: 27 (one-hot: empty, X, O for each of 9 cells)"
        printfn "  Hidden 1: 128 neurons (ReLU)"
        printfn "  Hidden 2: 64 neurons (ReLU)"
        printfn "  Output: 9 neurons (Softmax) - probabilities for each move"
        let totalParams = 27*128 + 128 + 128*64 + 64 + 64*9 + 9
        printfn $"  Total parameters: {totalParams}"
        printfn ""
        
        printfn "=== TRAINING CONFIGURATION ==="
        printfn "  Optimizer: Adam (lr=0.001)"
        printfn "  Loss: CrossEntropy"
        printfn $"  Epochs: {epochs}"
        printfn "  Batch size: 64"
        printfn ""
        printfn "Starting training...\n"
        
        let result = trainModel epochs None true
        
        printfn "\n=== FINAL RESULTS ==="
        printfn $"Train Accuracy: {result.TrainAccuracy * 100.0:N2}%%"
        printfn $"Test Accuracy: {result.TestAccuracy * 100.0:N2}%%"
        
        let finalLoss = List.rev result.Metrics.TrainLoss |> List.head
        let firstLoss = List.head result.Metrics.TrainLoss
        
        printfn "\n=== LOSS PROGRESSION ==="
        printfn $"  First epoch loss: {firstLoss:N6}"
        printfn $"  Final epoch loss: {finalLoss:N6}"
        
        result.Network
    
    let getNetworkMove (network: Layers.NeuralNetwork) (board: Board) =
        let features = boardToFeatures board
        let predictions, _ = Layers.forwardNetwork network features
        
        let bestMoveIndex = 
            [0..8] 
            |> List.maxBy (fun i -> predictions[0, i])
        
        let row = bestMoveIndex / 3
        let col = bestMoveIndex % 3
        (row, col)
    
    let makeMove (board: Board) (row: int) (col: int) (player: Cell) =
        if board[row].[col] = Empty then
            board[row].[col] <- player
            true
        else false
    
    let getPlayerMove () =
        printfn "Enter your move (row and column, 0-2, separated by space):"
        let input = Console.ReadLine().Split()
        if input.Length = 2 then
            try
                let row = int input[0]
                let col = int input[1]
                if row >= 0 && row <= 2 && col >= 0 && col <= 2 then
                    Some (row, col)
                else
                    printfn "Invalid coordinates! Use 0, 1, or 2."
                    None
            with _ ->
                printfn "Invalid input!"
                None
        else
            printfn "Please enter two numbers!"
            None
    
    let rec playGame (network: Layers.NeuralNetwork) =
        printfn "\n========================================"
        printfn "PLAY TIC-TAC-TOE AGAINST AI"
        printfn "========================================\n"
        printfn "You are X (first player), AI is O"
        printfn "Board coordinates: row 0-2, column 0-2"
        printfn ""
        
        let mutable board = createEmptyBoard ()
        let mutable gameActive = true
        
        while gameActive do
            printfn "\nCurrent board:"
            printfn $"{boardToString board}"
            printfn ""
            
            printfn "Your turn (X):"
            match getPlayerMove () with
            | Some (row, col) ->
                if makeMove board row col X then
                    match checkWinner board with
                    | Some X ->
                        printfn "\nFinal board:"
                        printfn $"{boardToString board}"
                        printfn "\nCongratulations! You won!"
                        gameActive <- false
                    | None when isDraw board ->
                        printfn "\nFinal board:"
                        printfn $"{boardToString board}"
                        printfn "\nIt's a draw!"
                        gameActive <- false
                    | _ ->
                        // AI turn
                        printfn "\nAI turn (O):"
                        let aiRow, aiCol = getNetworkMove network board
                        printfn $"AI plays: {aiRow} {aiCol}"
                        makeMove board aiRow aiCol O |> ignore
                        
                        match checkWinner board with
                        | Some O ->
                            printfn "\nFinal board:"
                            printfn $"{boardToString board}"
                            printfn "\nAI wins!"
                            gameActive <- false
                        | None when isDraw board ->
                            printfn "\nFinal board:"
                            printfn $"{boardToString board}"
                            printfn "\nIt's a draw!"
                            gameActive <- false
                        | _ -> ()
                else
                    printfn "Cell is not empty! Try again."
            | None -> ()
        
        printfn "\n========================================"
        printfn "Do you want to play again? (y/n)"
        let response = Console.ReadLine().Trim().ToLower()
        if response = "y" || response = "yes" then
            playGame network
    
    let testAI (network: Layers.NeuralNetwork) (gamesCount: int) =
        printfn "\n========================================"
        printfn "TESTING AI AGAINST RANDOM MOVES"
        printfn "========================================\n"
        
        let rnd = Random()
        let mutable aiWins = 0
        let mutable draws = 0
        let mutable randomWins = 0
        
        for game in 1 .. gamesCount do
            let mutable board = createEmptyBoard ()
            let mutable aiTurn = false
            let mutable gameActive = true
            
            while gameActive do
                if aiTurn then
                    let row, col = getNetworkMove network board
                    makeMove board row col O |> ignore
                else
                    let emptyCells = 
                        [ for r in 0..2 do
                            for c in 0..2 do
                                if board[r].[c] = Empty then (r, c) ]
                    if emptyCells.Length > 0 then
                        let row, col = emptyCells[rnd.Next(emptyCells.Length)]
                        makeMove board row col X |> ignore
                
                match checkWinner board with
                | Some O ->
                    aiWins <- aiWins + 1
                    gameActive <- false
                | Some X ->
                    randomWins <- randomWins + 1
                    gameActive <- false
                | None when isDraw board ->
                    draws <- draws + 1
                    gameActive <- false
                | _ -> ()
                
                aiTurn <- not aiTurn
        
        let total = float gamesCount
        printfn "=== RESULTS ==="
        printfn $"Games played: {gamesCount}"
        printfn $"AI wins: {aiWins} ({float aiWins / total * 100.0:N1}%%)"
        printfn $"Draws: {draws} ({float draws / total * 100.0:N1}%%)"
        printfn $"Random player wins: {randomWins} ({float randomWins / total * 100.0:N1}%%)"
    
    let run () =
        printfn "\n========================================"
        printfn "TIC-TAC-TOE NEURAL NETWORK AI"
        printfn "========================================"
        printfn ""
        printfn "This example trains a neural network to play Tic-Tac-Toe as O (second player)."
        printfn "The network is trained on optimal moves generated by Minimax algorithm."
        printfn ""
        
        let trainedNetwork = trainAI 90
        
        printfn "\n========================================"
        printfn "TESTING AI PERFORMANCE"
        printfn "========================================"
        testAI trainedNetwork 500
        
        printfn "\n========================================"
        printfn "PLAY AGAINST THE AI"
        printfn "========================================"
        printfn ""
        printfn "Do you want to play against the trained AI? (y/n)"
        let response = Console.ReadLine().Trim().ToLower()
        if response = "y" || response = "yes" then
            playGame trainedNetwork
        
        printfn "\n========================================"
        printfn "EXAMPLE COMPLETED"
        printfn "========================================\n"
