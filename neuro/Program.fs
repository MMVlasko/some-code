// Program.fs
open System

[<EntryPoint>]
let main argv =
    printfn "========================================"
    printfn "Neural Network Framework Examples"
    printfn "========================================"
    
    let rec showMenu() =
        printfn "\nSelect option to run:"
        printfn "1. Comprehensive Framework Tests"
        printfn "2. Iris Classification"
        printfn "3. MNIST Classification"
        printfn "4. TicTacToe"
        printfn "5. MNIST CNN (Conv2D)"
        printfn "6. Hyperparameter Lab"
        printfn "7. MNIST Regularization Experiments"
        printfn "8. Interpretability Tools"
        printfn "9. Benchmark Harness"
        printfn "0. Exit"
        printfn ""
        printf "Your choice: "
        
        match Console.ReadLine().Trim() with
        | "1" -> 
            printfn "\n" 
            Examples.Tests.run()
            showMenu()
        | "2" -> 
            Examples.Iris.run()
            showMenu()
        | "3" -> 
            Examples.MNIST.run()
            showMenu()
        | "4" -> 
            Examples.TicTacToe.run()
            showMenu()
        | "5" ->
            Examples.CNNMNIST.run()
            showMenu()
        | "6" ->
            Examples.HyperparameterLab.run()
            showMenu()
        | "7" ->
            Examples.RegularizationMNIST.run()
            showMenu()
        | "8" ->
            Examples.Interpretability.run()
            showMenu()
        | "9" ->
            Examples.BenchmarkHarness.run()
            showMenu()
        | "0" -> 
            printfn "\nGoodbye!"
            0
        | _ -> 
            printfn "Invalid choice, please try again"
            showMenu()
    
    showMenu()
