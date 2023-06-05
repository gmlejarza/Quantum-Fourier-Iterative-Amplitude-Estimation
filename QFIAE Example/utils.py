import pennylane as qml
import matplotlib.pyplot as plt
#from pennylane import numpy as np
import numpy as np
from scipy.linalg import expm

def funct_example(x):
    """
    Generate the target function

    Args:
     - z the variable
     -m the mass (a parameter that should go from 5 to 175 changing from 5 to 5 (5,175,5)
     -j a scaling parameter(0, or 2) when j=1 g(z)=0
    Returns:
     - g: target function """

    z = x.copy()
    # We define the function as a piecewise to take advantage of the even symmetry:
    z[z < 0] = -1 * z[z < 0]

    # We define the function

    g = 1+z**2
    return np.real(g)

def cost_function(xtrain,function_train, weights):
    """
   Cost function consisting on the squared difference between the function output of the circuit (expceted value) and the real value of
   the target function. It is averaged aver all training points.

    Args:
    -params: The params will be updated after each step of the minimization method.
    Returns:
    - Value of the cost function.
    """
    #TODO the current version is not optimized since this function always require to load xtrain
    # if params is not None:
    # params = params.reshape(layers+1, ansatz)
    loss = 0.0
    x1 = xtrain
    #function_train = funct(xtrain, m,j)
    N = len(function_train)
    for i in range(N):
        y_pred = linear_ansatz(weights, x=x1[i])
        loss += (function_train[i ] -y_pred )**2
    loss /= N
    return loss


#### Quantum MODEL
dev = qml.device('lightning.qubit', wires=1)
#dev = qml.device('default.qubit', wires=1)

def S(x):
    """Data encoding circuit block."""
    qml.RZ(x, wires=0)



def A(theta):
    """Trainable circuit block.
    :rtype: object
    """
    qml.RZ(theta[0], wires=0)
    qml.RY(theta[1], wires=0)
    qml.RZ(theta[2], wires=0)

@qml.qnode(dev)
def linear_ansatz(weights, x=None):
    ## First unitary, Layer 0
    A(weights[0])
    for i in range(1, len(weights)):
        
        S(x)
        A(weights[i])
    return qml.expval(qml.PauliZ(wires=0))


def fourier_series(coeffs, num_points, period=2 * np.pi):
    """
    Computes the Fourier series for a given set of coefficients.
    """

    series = [complex(0, 0) for _ in range(num_points)]
    x_vals = np.linspace(-np.pi, np.pi, num_points)
    n = 0
    series += coeffs[0] * np.exp(1j * n * x_vals * (2 * np.pi / period))
    for i in range(1, int((len(coeffs) - 1) / 2) + 1):
        n += 1
        series += coeffs[i] * np.exp(1j * n * x_vals * (2 * np.pi / period))
    for i in range(int((len(coeffs) - 1) / 2 + 1), len(coeffs), 1):
        series += coeffs[i] * np.exp(1j * -n * x_vals * (2 * np.pi / period))
        n -= 1

    # x_vals = np.linspace(0, period, num_points)
    y_vals = np.real(series)
    #plt.plot(x_vals, y_vals)
    #plt.show()
    return (y_vals)

def plot_data_fourier(function_test,norm, r, weights, xtest, y_fourier, fig=None):
    """
    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
        params_opt (array): params after the optimization
    """

    #params = weights.reshape(r + 1, 3)

    SSE = 0
    SST = 0
    average = np.sum(function_test) / function_test.size

    x1 = xtest
    N = len(function_test)
    f_pred = np.zeros(N, dtype='complex')

    for i in range(N):
        f_pred_x = linear_ansatz(weights, x=x1[i])
        f_pred[i] = f_pred_x
        SSE += (function_test[i] - f_pred_x) ** 2
        SST += (function_test[i] - average) ** 2

    R = 1 - SSE / SST
    
    print('accuracy: ', R * 100, '%')
    R=round(float(R)*100, 1)
    #plt.plot(x1, np.real(f_pred), label='Quantum')
    plt.plot(x1, function_test*norm, label='Target')
    plt.plot(x1, y_fourier, label=f'Quantum Fourier \nAccuracy: {R}%')
    #plt.plot(, , label=f'Accuracy: {R})')
    # Add legend
    plt.legend()

    # Mostrar el gráfico
    #plt.show()

    return plt

#export the fourier coefficients to the format needed to be integrated into a .txt
def exp_fourier_to_trig(fourier_coeffs):
    # Obtener la longitud de la lista de coeficientes de Fourier
    n = len(fourier_coeffs)

    # Obtener el coeficiente c0
    c0 = fourier_coeffs[0]

    # Inicializar las listas para los coeficientes b y a
    b_coeffs = [2 * np.real(fourier_coeffs[i]) for i in range(1, int(n / 2) + 1)]
    a_coeffs = [-2 * np.imag(fourier_coeffs[i]) for i in range(1, int(n / 2) + 1)]

    # Devolver los coeficientes en la forma trigonométrica
    return [c0] + b_coeffs + a_coeffs
def trig_to_final_array(trig_coeffs):
        array_list = np.zeros(shape=(len(trig_coeffs), 3))
        array_list[0, 0] = -1
        array_list[0, 1] = 0
        array_list[0, 2] = np.real(trig_coeffs[0])
        for i in range(1, int((len(trig_coeffs) - 1) / 2) + 1):
            # cosine
            array_list[i, 0] = 0
            array_list[i, 1] = i
            array_list[i, 2] = trig_coeffs[i]
        for i in range(int((len(trig_coeffs) - 1) / 2) + 1, len(trig_coeffs)):
            # cosine
            array_list[i, 0] = 1
            array_list[i, 1] = i - (len(trig_coeffs) - 1) / 2
            array_list[i, 2] = trig_coeffs[i]
        return(array_list)
