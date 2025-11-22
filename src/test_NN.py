import numpy as np
from .NNClass import NN

def test_relu_and_derivative():
    x = np.array([-1.0, 0.0, 2.0])
    expected_relu = np.array([0.0, 0.0, 2.0])
    expected_deriv = np.array([0, 0, 1])
    assert np.allclose(NN.ReLU(None, x, None), expected_relu)
    assert np.allclose(NN.ReLUDerivative(None, x), expected_deriv)

def test_softmax():
    x = np.array([[1.0, 2.0],[0.5, 0.5]])
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected_softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    out = NN.softMax(None, x, None)
    assert np.allclose(out, expected_softmax, atol=1e-8)

def test_forward():
    np.random.seed(0)
    inputs = np.array([[1.0,2.0],[0.5,1.5]])
    nn = NN(polynomialType=None, layerSizes=[2, 2], batchSize=[1], activationFunction="relu")
    nn.weights = [np.array([[0.1,0.2],[0.3,0.4]]),np.array([[0.5,0.6],[0.7,0.8]])]
    nn.biases = [np.array([0.1,0.2]), np.array([0.3,0.4])]
    out = nn.forward(inputs)
    
    hidden = np.maximum(np.dot(inputs, nn.weights[0]) + nn.biases[0],0)
    logits = np.dot(hidden, nn.weights[1]) + nn.biases[1]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    expected_out = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    assert np.allclose(out, expected_out, atol=1e-6)

def test_backprop_shapes():
    np.random.seed(0)

    inputs = np.random.randn(3, 4)  
    labels = np.array([
        [1, 0],
        [0, 1],
        [1, 0]
    ])

    nn = NN(polynomialType=None, layerSizes=[4, 5, 2], batchSize=[1], activationFunction="relu")
    
    nn.weights = [
        np.random.randn(4, 5),  
        np.random.randn(5, 2)   
    ]
    nn.biases = [
        np.random.randn(5),
        np.random.randn(2)
    ]

    outputs = nn.forward(inputs)

    Grads, Params = nn.backPropagation(outputs, labels)

    for layer in range(2):
        print(layer, len(Grads["Weight"]), len(Grads["bias"]))
        assert np.allclose(np.array(Grads["Weight"][layer]).shape, nn.weights[layer].shape)
        assert np.allclose(np.array(Grads["bias"][layer]).shape, nn.biases[layer].shape)
        assert np.allclose(np.array(Params["Weight"][layer]).shape, nn.weights[layer].shape)
        assert np.allclose(np.array(Params["bias"][layer]).shape, nn.biases[layer].shape)

def test_polynomial_sharedParams():
    nn = NN(polynomialType="layer", layerSizes=[3], batchSize=None, activationFunction="quad")
    nn.a = [1.0]
    nn.b = [1.0]
    nn.c = [1.0]
    inputs = np.array([0, 0.5, 1])
    output = nn.quadraticPolynomial(inputs, 0)
    expected = [1.0, 1.75, 3.0]
    assert np.allclose(output, expected)

    errorTerm = [1.0, 1.0, 1.0]
    Aderivative, Bderivative, Cderivative, inputDerivative = nn.quadraticPolynomialDerivative(inputs, errorTerm, 0)

    expectedAderiv = np.mean(errorTerm * inputs ** 2)
    expectedBderiv = np.mean(errorTerm * inputs)
    expectedCderiv = np.mean(errorTerm)
    expectedInputDeriv = nn.a[0] * 2 * inputs + nn.b[0]

    assert np.allclose(expectedAderiv, Aderivative)
    assert np.allclose(expectedBderiv, Bderivative)
    assert np.allclose(expectedCderiv, Cderivative)
    assert np.allclose(expectedInputDeriv, inputDerivative)

def test_polynomial_singleParams():
    nn = NN(polynomialType="node", layerSizes=[3], batchSize=None, activationFunction="quad")
    nn.a = np.array([[1.0, 0.5, 0.25]])
    nn.b = np.array([[1.0, 0.5, 0.25]])
    nn.c = np.array([[1.0, 0.5, 0.25]])
    inputs = np.array([0, 0.5, 1.0])
    output = nn.quadraticPolynomial(inputs, 0)
    expected = [
        1.0  * (0.0**2) + 1.0  * (0.0) + 1.0,
        0.5  * (0.5**2) + 0.5  * (0.5) + 0.5,
        0.25 * (1.0**2) + 0.25 * (1.0) + 0.25,
    ]
    assert np.allclose(output, expected)

    errorTerm = [1.0, 1.0, 1.0]
    Aderivative, Bderivative, Cderivative, inputDerivative = nn.quadraticPolynomialDerivative(inputs, errorTerm, 0)

    expectedAderiv = np.sum(errorTerm * (inputs ** 2), axis=0)
    expectedBderiv = np.sum(errorTerm * inputs, axis=0)
    expectedCderiv = np.sum(errorTerm, axis=0)
    expectedInputDeriv = nn.a[0] * 2 * inputs + nn.b[0]

    assert np.allclose(expectedAderiv, Aderivative)
    assert np.allclose(expectedBderiv, Bderivative)
    assert np.allclose(expectedCderiv, Cderivative)
    assert np.allclose(expectedInputDeriv, inputDerivative)