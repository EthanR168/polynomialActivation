import numpy as np
import math

class NN:
    def __init__(self, polynomialType, layerSizes, batchSize, activationFunction, normalisation = "none", outputActivation = "softmax", lossFunction="cross"):
        self.polynomialType = polynomialType # layer, node
        self.layerSizes = layerSizes
        self.batchSize = batchSize
        self.normalisation = normalisation.lower()
        self.outputActivation = outputActivation.lower()
        self.lossFunction = lossFunction.lower()
        assert (self.lossFunction == "cross" and self.outputActivation == "softmax") or (self.lossFunction == "mse" and self.outputActivation == "linear")

        activations = {
            "relu": self.ReLU,
            "sigmoid": self.sigmoid,
            "quad": self.quadraticPolynomial, 
            "cubic": self.cubicPolynomial,
        }
        self.activationFunction = activations[activationFunction.lower()]
        derivs = {
            "relu": self.ReLUDerivative,
            "sigmoid": self.sigmoidDerivative,
            "quad": self.quadraticPolynomialDerivative,
            "cubic": self.cubicPolynomialDerivative,
        }
        self.activationFunctionDerivitive = derivs[activationFunction.lower()]

        self.weights = []
        self.biases = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []

        self.shift = []
        self.scale = []

    def createWeightsAndBiases(self):
        for i in range(1, len(self.layerSizes)):
            currSize = self.layerSizes[i]
            lastSize = self.layerSizes[i - 1]

            if(self.activationFunction == self.ReLU):
                bounds = math.sqrt(2 / lastSize) # He initialisation
            elif(self.activationFunction == self.sigmoid):
                bounds = math.sqrt(6/ (lastSize + currSize)) # Xavier initialisation
            else:
                bounds = 1
                
            self.weights.append(np.random.normal(0, bounds, size=(lastSize, currSize)).astype(np.float32))
            self.biases.append(np.random.normal(0, bounds, size=(currSize)).astype(np.float32))
            
            varBound = 0.01
            if(self.polynomialType == "layer"):
                shape = (1,)
            elif(self.polynomialType == "node"):
                shape = (currSize,)

            if(i != len(self.layerSizes) - 1 and self.activationFunction == self.quadraticPolynomial):
                self.a.append(np.random.normal(0, varBound, size=shape).astype(np.float32))
                self.b.append(np.random.normal(0, varBound, size=shape).astype(np.float32))
                self.c.append(np.random.normal(0, varBound, size=shape).astype(np.float32))
            
            elif(i != len(self.layerSizes) - 1 and self.activationFunction == self.cubicPolynomial):
                self.a.append(np.random.normal(0, varBound, size=shape).astype(np.float32))
                self.b.append(np.random.normal(0, varBound, size=shape).astype(np.float32))
                self.c.append(np.random.normal(0, varBound, size=shape).astype(np.float32))
                self.d.append(np.random.normal(0, varBound, size=shape).astype(np.float32))

            if(i != len(self.layerSizes) - 1 and (self.normalisation == "layer" or self.normalisation == "batch")):
                normShape = (currSize,)
                self.scale.append(np.ones(normShape, dtype=np.float32))
                self.shift.append(np.zeros(normShape, dtype=np.float32))

    def ReLU(self, inputs, _):
        return np.maximum(inputs, 0)
    
    def ReLUDerivative(self, inputs):
        return np.where(inputs > 0, 1, 0)

    def softMax(self, values, _): 
        values = np.array(values, dtype=np.float64)
        maxVal = np.max(values, axis=1, keepdims=True)
        values = values - maxVal
        summ = np.sum(np.exp(values), axis=1, keepdims=True)
        out = np.exp(values) / summ
        return out

    def linear(self, values, _):
        return values

    def sigmoid(self, values, _):
        return 1 / (1 + np.exp(-values))

    def sigmoidDerivative(self, values):
        sig = self.sigmoid(values, None)
        return sig * (1 - sig)

    def quadraticPolynomial(self, values, layerIndex):
        if(self.normalisation == "tanh"): # Apply TanH to inputs to limit them between -1 and 1
            values = np.tanh(values)
        elif(self.normalisation == "clip"): # Clip inputs between -10 and 10
            values = np.clip(values, -10, 10)


        if(self.polynomialType == "layer"): # Honestly could remove these if statements as they repeat code
            return self.a[layerIndex] * (values ** 2) + self.b[layerIndex] * values + self.c[layerIndex]   
        elif(self.polynomialType == "node"):
            return self.a[layerIndex] * (values ** 2) + self.b[layerIndex] * values + self.c[layerIndex]   
        
    def cubicPolynomial(self, values, layerIndex):
        if(self.normalisation == "tanh"): # Apply TanH to inputs to limit them between -1 and 1
            values = np.tanh(values)
        elif(self.normalisation == "clip"): # Clip inputs between -10 and 10
            values = np.clip(values, -10, 10)


        if(self.polynomialType == "layer"):
            return self.a[layerIndex] * (values ** 3) + self.b[layerIndex] * (values ** 2) + self.c[layerIndex] * values + self.d[layerIndex]
        elif(self.polynomialType == "node"):
            return self.a[layerIndex] * (values ** 3) + self.b[layerIndex] * (values ** 2) + self.c[layerIndex] * values + self.d[layerIndex]  
        
    def quadraticPolynomialDerivative(self, inputs, errorTerm, layerIndex):
        if(self.normalisation == "tanh"):
            # f(x) = a * tanh(x)^2 + b * tanh(x) + c
            # d/x (tanh[x]) = 1 - tanh(x)^2
            # d/x (a * tanh[x]^2) = a * 2 * tanh(x) * (d/dx)[tanh(x)]
            # d/x (b * tanh[x]) = b * (1 - tanh(x)^2)
            # d/dx = 2 * a * tanh(x) * (1 - tanh(x)^2) + b(1 - tanh(x)^2)
            #
            # d/dx = (1 - tanh(x)^2)(2 * a * tanh(x) + b)   <--- factored out (1 - tan(x)^2)
            # d/da = tanh(x)^2      
            # d/db = tanh(x)
            # d/dc = 1
            #
            # dL/dP = error * d/dP
            # where dP is the parameter

            tanh = np.tanh(inputs)
            inputDerivative = (1 - tanh ** 2) * (2 * self.a[layerIndex] * tanh + self.b[layerIndex])

            Aderivative = errorTerm * (tanh ** 2)
            Bderivative = errorTerm * tanh
            Cderivative = errorTerm
        elif(self.normalisation == "clip"):
            # clip(x) = clip(x, -10, 10)  <--- bounds them to -10 and 10
            # f(x) = a * clip(x)^2 + b * clip(x) + c
            # d/x (clip[x]) = 0 if x < -10 or x > 10, else 1
            #
            # d/dx = (2 * a * clip(x) + b) * d/dx clip(x)
            # d/da = clip(x)^2      
            # d/db = clip(x)
            # d/dc = 1
            #
            # dL/dP = error * d/dP
            # where dP is the parameter

            clipped = np.clip(inputs, -10, 10)
            derive = 2 * self.a[layerIndex] * clipped + self.b[layerIndex]
            clipDerivative = np.where((inputs >= -10) & (inputs <= 10), 1.0, 0.0)

            inputDerivative = derive * clipDerivative

            Aderivative = errorTerm * (clipped ** 2)
            Bderivative = errorTerm * clipped
            Cderivative = errorTerm
        else:
            # f(x) = ax^2 + bx + c
            # d/dx = 2ax + b     <- d/x [f(x) + g(x)] = f'(x) + g'(x)
            # d/da = x^2        
            # d/db = x
            # d/dc = 1
            #
            # dL/dP = error * d/dP
            # where dP is the parameter

            inputDerivative = 2 * self.a[layerIndex] * inputs + self.b[layerIndex]

            Aderivative = errorTerm * (inputs ** 2)
            Bderivative = errorTerm * inputs
            Cderivative = errorTerm

        if(self.polynomialType == "layer"): # abc are shared between whole layer
            Aderivative = np.mean(Aderivative) # averages all the gradients
            Bderivative = np.mean(Bderivative) # averages all the gradients
            Cderivative = np.mean(Cderivative) # averages all the gradients
        elif(self.polynomialType == "node"):
            Aderivative = np.sum(Aderivative, axis=0) 
            Bderivative = np.sum(Bderivative, axis=0) 
            Cderivative = np.sum(Cderivative, axis=0) 

        return Aderivative, Bderivative, Cderivative, inputDerivative
        
    def cubicPolynomialDerivative(self, inputs, errorTerm, layerIndex):
        if(self.normalisation == "tanh"):
            # f(x) = a * tanh(x)^3 + b * tanh(x)^2 + c * tanh(x) + d
            # d/x (tanh[x]) = 1 - tanh(x)^2
            # d/x (a * tanh[x]^3) = a * 3 * tanh(x)^2 * (d/dx)[tanh(x)]
            # d/x (b * tanh[x]^2) = b * 2 * tanh(x) * (d/dx)[tanh(x)]
            # d/x (c * tanh[x]) = c * (d/dx)[tanh(x)]
            # d/dx = a * 3 * tanh(x)^2 * (d/dx)[tanh(x)] + b * 2 * tanh(x) * (d/dx)[tanh(x)] + c * (d/dx)[tanh(x)]
            # d/dx = a * 3 * tanh(x)^2 * (1 - tanh(x)^2) + b * 2 * tanh(x) * (1 - tanh(x)^2) + c * (1 - tanh(x)^2)
            #
            # d/dx = (1 - tanh(x)^2) * (3 * a * tanh(x)^2 + 2 * b * tanh(x) + c)   <--- factored out (1 - tanh(x)^2)
            # d/da = tanh(x)^3      
            # d/db = tanh(x)^2
            # d/dc = tanh(x)
            # d/dd = 1
            #
            # dL/dP = error * d/dP
            # where dP is the parameter

            tanh = np.tanh(inputs)
            inputDerivative = (1 - tanh ** 2) * (3*self.a[layerIndex]*tanh**2 + 2*self.b[layerIndex]*tanh + self.c[layerIndex])

            Aderivative = errorTerm * (tanh ** 3)
            Bderivative = errorTerm * (tanh ** 2)
            Cderivative = errorTerm * tanh
            Dderivative = errorTerm

        elif(self.normalisation == "clip"):
            # clip(x) = clip(x, -10, 10)  <--- bounds them to -10 and 10
            # f(x) = a * clip(x)^2 + b * clip(x) + c
            # d/x (clip[x]) = 0 if x < -10 or x > 10, else 1
            # d/x (a*clip(x)^3) = 3 * a * clip(x)^2 * (d/x)[clip(x)]
            # d/x (b*clip(x)^2) = 2 * b * clip(x) * (d/x)[clip(x)]
            # d/x (c*clip(x)) = c * (d/x)[clip(x)]
            # d/x (d) = 0
            # d/dx = 3 * a * clip(x)^2 * (d/x)[clip(x)] + 2 * b * clip(x) * (d/x)[clip(x)] + c * (d/x)[clip(x)]
            # d/dx = (d/x)[clip(x)] * (3 * a * clip(x)^2 + 2 * b * clip(x) + c)
            #
            # d/dx = (d/x)[clip(x)] * (3 * a * clip(x)^2 + 2 * b * clip(x) + c)
            # d/da = clip(x)^3    
            # d/db = clip(x)^2      
            # d/dc = clip(x)
            # d/dd = 1
            #
            # dL/dP = error * d/dP
            # where dP is the parameter

            clipped = np.clip(inputs, -10, 10)
            derive = 3 * self.a[layerIndex] * (clipped ** 2) + 2 * self.b[layerIndex] * clipped + self.c[layerIndex]
            clipDerivative = np.where((inputs >= -10) & (inputs <= 10), 1.0, 0.0)

            inputDerivative = derive * clipDerivative

            Aderivative = errorTerm * (clipped ** 3)
            Bderivative = errorTerm * (clipped ** 2)
            Cderivative = errorTerm * clipped
            Dderivative = errorTerm

        else:
            # f(x) = ax^3 + bx^2 + cx + d
            # d/dx = 3ax^2 + 2bx + c     <- d/x [f(x) + g(x) + h(x)] = f'(x) + g'(x) + h'(x)
            # d/da = x^3        
            # d/db = x^2
            # d/dc = x
            # d/dd = 1
            #
            # dL/dP = error * d/dP
            # where dP is the parameter

            inputDerivative = 3 * self.a[layerIndex] * (inputs ** 2) + 2 * self.b[layerIndex] * inputs + self.c[layerIndex]

            Aderivative = errorTerm * (inputs ** 3)
            Bderivative = errorTerm * (inputs ** 2)
            Cderivative = errorTerm * inputs
            Dderivative = errorTerm

        if(self.polynomialType == "layer"): # abcd are shared between whole layer
            Aderivative = np.mean(Aderivative) # averages all the gradients
            Bderivative = np.mean(Bderivative) # averages all the gradients
            Cderivative = np.mean(Cderivative) # averages all the gradients
            Dderivative = np.mean(Dderivative) # averages all the gradients
        elif(self.polynomialType == "node"):
            Aderivative = np.sum(Aderivative, axis=0) 
            Bderivative = np.sum(Bderivative, axis=0) 
            Cderivative = np.sum(Cderivative, axis=0) 
            Dderivative = np.sum(Dderivative, axis=0) 
          
        return Aderivative, Bderivative, Cderivative, Dderivative, inputDerivative

    def batchNorm(self, values, layerIndex):
        # BatchNorm normalises the inputs using mean and variance across the batch
        # m = mean(values)      <--- mean per batch
        # v = variance(values)  <--- variance per batch
        # e = epsilon           <--- small value to stop division by 0
        # newInputs = (values - m) / sqrt(v + e)
        #
        # normalisedInputs = scale * newInputs + shift
        # scale and shift are learnable parameters

        mean = np.mean(values, axis=0, keepdims=True)
        variance = np.var(values, axis=0, keepdims=True)
        epsilon = 1e-8
        new = (values - mean) / np.sqrt(variance + epsilon)
        normalisedValues = self.scale[layerIndex] * new + self.shift[layerIndex]
        return normalisedValues

    def batchNormDerivative(self, inputs, errorTerm, layerIndex):
        mean = np.mean(inputs, axis=0)
        variance = np.var(inputs, axis=0)
        epsilon = 1e-8  
        x = (inputs - mean) / np.sqrt(variance + epsilon)

        dScale= np.sum(errorTerm * x, axis=0)
        dShift = np.sum(errorTerm, axis=0)

        dx = errorTerm * self.scale[layerIndex]
        N = inputs.shape[0]

        inputDerivative = (1.0 / N) * (1 / np.sqrt(variance + epsilon)) * (
            N * dx - np.sum(dx, axis=0) - x * np.sum(dx * x, axis=0)
        )

        return dScale, dShift, inputDerivative

    def layerNorm(self, values, layerIndex):
        # LayerNorm normalises the inputs using mean and variance across each sample
        # m = mean(values)      <--- mean per sample
        # v = variance(values)  <--- variance per sample
        # e = epsilon           <--- small value to stop division by 0
        # newInputs = (values - m) / sqrt(v + e)
        #
        # normalisedInputs = scale * newInputs + shift
        # scale and shift are learnable parameters

        mean = np.mean(values, axis=1, keepdims=True)
        variance = np.var(values, axis=1, keepdims=True)
        epsilon = 1e-8
        new = (values - mean) / np.sqrt(variance + epsilon)
        normalisedValues = self.scale[layerIndex] * new + self.shift[layerIndex]
        return normalisedValues

    def layerNormDerivative(self, inputs, errorTerm, layerIndex):
        mean = np.mean(inputs, axis=1, keepdims=True)
        variance = np.var(inputs, axis=1, keepdims=True)
        epsilon = 1e-8  
        x = (inputs - mean) / np.sqrt(variance + epsilon)

        dScale= np.sum(errorTerm * x, axis=0)
        dShift = np.sum(errorTerm, axis=0)

        N = inputs.shape[1]
        dx = errorTerm * self.scale[layerIndex]
        
        inputDerivative = (1.0 / N) * (1 / np.sqrt(variance + epsilon)) * (
            N * dx - np.sum(dx, axis=1, keepdims=True) - x * np.sum(dx * x, axis=1, keepdims=True)
        )

        return dScale, dShift, inputDerivative

    def forward(self, inputs):
        self.normPreActivations = []
        self.preActivations = []
        self.activations = [inputs]
        
        for i in range(len(self.weights)):
            preActivation = np.dot(inputs, self.weights[i]) + self.biases[i]
            self.preActivations.append(preActivation)

            if(i != len(self.weights) - 1):
                if(self.normalisation == "batch"):
                    preActivation = self.batchNorm(preActivation, i)
                elif(self.normalisation == "layer"):
                    preActivation = self.layerNorm(preActivation, i)

            self.normPreActivations.append(preActivation)

            if(i == len(self.weights) - 1):
                if(self.outputActivation == "softmax"):
                    inputs = self.softMax(preActivation, i)
                elif(self.outputActivation == "linear"):
                    inputs = self.linear(preActivation, i)
                    
            else:
                inputs = self.activationFunction(preActivation, i)

            self.activations.append(inputs)
        return inputs

    def backPropagation(self, outputs, labels):
        errorTerms = outputs - labels # this simplification is due to cross entropy and softmax being used (for output layer)

        allWeightGradients = []
        allBiasGradients = []
        aGradients = []
        bGradients = []
        cGradients = []
        dGradients = []
        scaleGradients = []
        shiftGradients = []

        for i in reversed(range(len(self.weights))):
            previousActivation = self.activations[i]
            weightGrad = previousActivation.T @ errorTerms
            biasGrad = np.sum(errorTerms, axis=0)
            allWeightGradients.insert(0, weightGrad)
            allBiasGradients.insert(0, biasGrad)

            if(i != 0):
                if(self.activationFunctionDerivitive == self.ReLUDerivative or self.activationFunctionDerivitive == self.sigmoidDerivative):
                    errorTerms = (errorTerms @ self.weights[i].T) * self.activationFunctionDerivitive(self.normPreActivations[i-1])
                elif(self.activationFunctionDerivitive == self.quadraticPolynomialDerivative):
                    backpropError = errorTerms @ self.weights[i].T
                    aG, bG, cG, inputDerivative = self.quadraticPolynomialDerivative(self.normPreActivations[i-1], backpropError, i-1)
                    errorTerms = backpropError * inputDerivative
                    aGradients.insert(0, aG)
                    bGradients.insert(0, bG)
                    cGradients.insert(0, cG)
                elif(self.activationFunctionDerivitive == self.cubicPolynomialDerivative):
                    backpropError = errorTerms @ self.weights[i].T
                    aG, bG, cG, dG, inputDerivative = self.cubicPolynomialDerivative(self.normPreActivations[i-1], backpropError, i-1)
                    errorTerms = backpropError * inputDerivative
                    aGradients.insert(0, aG)
                    bGradients.insert(0, bG)
                    cGradients.insert(0, cG)
                    dGradients.insert(0, dG)
                else:
                    raise ValueError("Incorrect Activation function derivative passed")
                
                
                if(self.normalisation == "batch"):
                    dScale, dShift, inputDerivative = self.batchNormDerivative(self.preActivations[i-1], errorTerms, i-1)
                    errorTerms = inputDerivative
                    scaleGradients.insert(0, dScale)
                    shiftGradients.insert(0, dShift)
                elif(self.normalisation == "layer"):
                    dScale, dShift, inputDerivative = self.layerNormDerivative(self.preActivations[i-1], errorTerms, i-1)
                    errorTerms = inputDerivative
                    scaleGradients.insert(0, dScale)
                    shiftGradients.insert(0, dShift)

        Gradients = {
            "Weight": allWeightGradients,
            "bias": allBiasGradients,
            "a": aGradients,
            "b": bGradients,
            "c": cGradients,
            "d": dGradients,
            "scale": scaleGradients,
            "shift": shiftGradients,
        }
        Parameters = {
            "Weight": self.weights,
            "bias": self.biases,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "scale": self.scale,
            "shift": self.shift,
        }
        return Gradients, Parameters
    
    def optimise(self, inputs, labels, learningRate):
        allOutputs = []
        allLabels = []
        for i in range(0, len(inputs), self.batchSize):
            batchData = inputs[i: i + self.batchSize]
            batchLabels = labels[i: i + self.batchSize]

            outputs = self.forward(batchData)
            allOutputs.append(outputs)
            allLabels.append(batchLabels)

            Gradients, Parameters = self.backPropagation(outputs, batchLabels)

            self.updateParameters(Gradients, Parameters, learningRate)
        return allOutputs, allLabels

    def updateParameters(self, Gradients, Parameters, learningRate):
        for key in Parameters:
            if(len(Gradients[key]) == 0):
                continue # when polynomial isnt used a,b,c,d will be empty
            for param in range(len(Parameters[key])): # Every paramter is inhomgous in this NN 
                Parameters[key][param] -= learningRate * (Gradients[key][param] / self.batchSize)
        
        self.weights = Parameters["Weight"]
        self.biases = Parameters["bias"]
        self.a = Parameters["a"]
        self.b = Parameters["b"]
        self.c = Parameters["c"]
        self.d = Parameters["d"]
        self.scale = Parameters["scale"]
        self.shift = Parameters["shift"]

        for i in range(len(self.a)): # clip to stop abcd from growing to fast
            self.a[i] = np.clip(self.a[i], -1, 1)
            self.b[i] = np.clip(self.b[i], -1, 1)
            self.c[i] = np.clip(self.c[i], -1, 1)
            if(self.activationFunction == self.cubicPolynomial):
                self.d[i] = np.clip(self.d[i], -1, 1)

    def CrossEntropyLossFunction(self, outputs, labels):
        epsilon = 1e-12
        outputs = np.clip(outputs, epsilon, 1 - epsilon)
        return -np.mean(np.sum(labels * np.log(outputs), axis=1))

    def MSE(self, outputs, labels):
        sqauredError = (labels - outputs) ** 2
        return np.mean(np.sum(sqauredError, axis=1))

    def train(self, inputs, labels, learningRate):
        correct = 0
        loss = 0
        allOutputs, allLabels = self.optimise(inputs, labels, learningRate) #allOutpus shape: (numBatches, batchSize, outputSize)

        totalSamples = 0
        for b in range(len(allOutputs)):
            totalSamples += len(allOutputs[b])

            if(self.lossFunction == "cross"):
                loss += self.CrossEntropyLossFunction(allOutputs[b], allLabels[b]) * len(allOutputs[b])
            elif(self.lossFunction == "mse"):
                loss += self.MSE(allOutputs[b], allLabels[b]) * len(allOutputs[b])
            
            predicted = np.argmax(allOutputs[b], axis=1)
            true = np.argmax(allLabels[b], axis=1)
            correct += np.sum(predicted == true)
        
        average = float(correct / totalSamples) * 100 
        loss = float(loss / totalSamples)
        return average, loss
    
