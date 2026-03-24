// Optimizers.fs
module Optimizers

open System

type Optimizer =
    | SGD of learningRate: float
    | Momentum of learningRate: float * momentum: float
    | Adam of learningRate: float * beta1: float * beta2: float * epsilon: float
    | GradientClipping of learningRate: float * threshold: float

type OptimizerState = {
    Velocities: (float[,] * float[]) list option
    AdamM: (float[,] * float[]) list option
    AdamV: (float[,] * float[]) list option
    Step: int
}

let initState (network: Layers.NeuralNetwork) optimizer =
    let layerParams = 
        network |> List.map (fun (layer: Layers.DenseLayer) -> (layer.Weights, layer.Bias))
    
    match optimizer with
    | SGD _ -> 
        { Velocities = None; AdamM = None; AdamV = None; Step = 0 }
    
    | Momentum _ ->
        let zeros = layerParams |> List.map (fun (w: float[,], b: float[]) -> 
            (Array2D.zeroCreate (w.GetLength(0)) (w.GetLength(1)),
             Array.zeroCreate b.Length))
        { Velocities = Some zeros; AdamM = None; AdamV = None; Step = 0 }
    
    | Adam _ ->
        let zerosM = layerParams |> List.map (fun (w: float[,], b: float[]) -> 
            (Array2D.zeroCreate (w.GetLength(0)) (w.GetLength(1)),
             Array.zeroCreate b.Length))
        let zerosV = layerParams |> List.map (fun (w: float[,], b: float[]) -> 
            (Array2D.zeroCreate (w.GetLength(0)) (w.GetLength(1)),
             Array.zeroCreate b.Length))
        { Velocities = None; AdamM = Some zerosM; AdamV = Some zerosV; Step = 1 }
    
    | GradientClipping _ ->
        { Velocities = None; AdamM = None; AdamV = None; Step = 0 }

let update (optimizer: Optimizer) (state: OptimizerState) (gradients: (float[,] * float[]) list) (network: Layers.NeuralNetwork) =
    let rec updateLayer 
        (idx: int) 
        (opt: Optimizer) 
        (currentState: OptimizerState) 
        (grads: (float[,] * float[]) list) 
        (layers: Layers.DenseLayer list) =
        
        match grads, layers with
        | [], [] -> 
            (currentState, [])
        | (gradW: float[,], gradB: float[]) :: restGrads, layer: Layers.DenseLayer :: restLayers ->
            let rows = layer.Weights.GetLength(0)
            let cols = layer.Weights.GetLength(1)
            
            let newW, newB, newState =
                match opt with
                | SGD lr ->
                    let newWArray = Array2D.init rows cols (fun i j -> 
                        layer.Weights[i, j] - lr * gradW[i, j])
                    let newBArray = Array.init layer.Bias.Length (fun i -> 
                        layer.Bias[i] - lr * gradB[i])
                    (newWArray, newBArray, currentState)
                
                | Momentum (lr, momentum) ->
                    match currentState.Velocities with
                    | Some velocities ->
                        let (velW: float[,], velB: float[]) = List.item idx velocities
                        
                        let newVelW = Array2D.init rows cols (fun i j -> 
                            momentum * velW[i, j] + lr * gradW[i, j])
                        let newVelB = Array.init layer.Bias.Length (fun i -> 
                            momentum * velB[i] + lr * gradB[i])
                        
                        let newWArray = Array2D.init rows cols (fun i j -> 
                            layer.Weights[i, j] - newVelW[i, j])
                        let newBArray = Array.init layer.Bias.Length (fun i -> 
                            layer.Bias[i] - newVelB[i])
                        
                        let newVelocities = 
                            velocities
                            |> List.mapi (fun i (vw, vb) -> 
                                if i = idx then (newVelW, newVelB) else (vw, vb))
                        
                        (newWArray, newBArray, { currentState with Velocities = Some newVelocities })
                    | None -> failwith "Invalid state for Momentum"
                
                | Adam (lr, beta1, beta2, eps) ->
                    match currentState.AdamM, currentState.AdamV with
                    | Some adamM, Some adamV ->
                        let (mW: float[,], mB: float[]) = List.item idx adamM
                        let (vW: float[,], vB: float[]) = List.item idx adamV
                        
                        let newMW = Array2D.init rows cols (fun i j -> 
                            beta1 * mW[i, j] + (1.0 - beta1) * gradW[i, j])
                        let newMB = Array.init layer.Bias.Length (fun i -> 
                            beta1 * mB[i] + (1.0 - beta1) * gradB[i])
                        
                        let newVW = Array2D.init rows cols (fun i j -> 
                            beta2 * vW[i, j] + (1.0 - beta2) * gradW[i, j] * gradW[i, j])
                        let newVB = Array.init layer.Bias.Length (fun i -> 
                            beta2 * vB[i] + (1.0 - beta2) * gradB[i] * gradB[i])
                        
                        let t = float currentState.Step
                        let beta1Pow = Math.Pow(beta1, t)
                        let beta2Pow = Math.Pow(beta2, t)
                        
                        let mHatW = Array2D.init rows cols (fun i j -> 
                            newMW[i, j] / (1.0 - beta1Pow))
                        let mHatB = Array.init layer.Bias.Length (fun i -> 
                            newMB[i] / (1.0 - beta1Pow))
                        
                        let vHatW = Array2D.init rows cols (fun i j -> 
                            newVW[i, j] / (1.0 - beta2Pow))
                        let vHatB = Array.init layer.Bias.Length (fun i -> 
                            newVB[i] / (1.0 - beta2Pow))
                        
                        let updateW = Array2D.init rows cols (fun i j -> 
                            lr * mHatW[i, j] / (Math.Sqrt(vHatW[i, j]) + eps))
                        let updateB = Array.init layer.Bias.Length (fun i -> 
                            lr * mHatB[i] / (Math.Sqrt(vHatB[i]) + eps))
                        
                        let newWArray = Array2D.init rows cols (fun i j -> 
                            layer.Weights[i, j] - updateW[i, j])
                        let newBArray = Array.init layer.Bias.Length (fun i -> 
                            layer.Bias[i] - updateB[i])
                        
                        let newAdamM = 
                            adamM
                            |> List.mapi (fun i (mw, mb) -> 
                                if i = idx then (newMW, newMB) else (mw, mb))
                        
                        let newAdamV = 
                            adamV
                            |> List.mapi (fun i (vw, vb) -> 
                                if i = idx then (newVW, newVB) else (vw, vb))
                        
                        let newStateRecord = { 
                            currentState with 
                                AdamM = Some newAdamM
                                AdamV = Some newAdamV
                                Step = currentState.Step + 1 
                        }
                        
                        (newWArray, newBArray, newStateRecord)
                    | _ -> failwith "Invalid state for Adam"
                
                | GradientClipping (lr, threshold) ->
                    let clipGrad (g: float) = 
                        if abs g > threshold then 
                            (if g > 0.0 then threshold else -threshold)
                        else g
                    
                    let clippedW = Array2D.init rows cols (fun i j -> 
                        clipGrad gradW[i, j])
                    let clippedB = gradB |> Array.map clipGrad
                    
                    let newWArray = Array2D.init rows cols (fun i j -> 
                        layer.Weights[i, j] - lr * clippedW[i, j])
                    let newBArray = Array.init layer.Bias.Length (fun i -> 
                        layer.Bias[i] - lr * clippedB[i])
                    
                    (newWArray, newBArray, currentState)
            
            let updatedLayer = { layer with Weights = newW; Bias = newB }
            let finalState, updatedLayers = updateLayer (idx + 1) opt newState restGrads restLayers
            (finalState, updatedLayer :: updatedLayers)
    
    let newState, updatedLayers = updateLayer 0 optimizer state gradients network
    (newState, updatedLayers)