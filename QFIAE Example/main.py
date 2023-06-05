## here read the parameters and run the script
import sys
import numpy as np
from pennylane import numpy as npy
import argparse
import json
from utils import cost_function, funct_example, linear_ansatz, fourier_series, plot_data_fourier, exp_fourier_to_trig, trig_to_final_array
from utils_qae import coeffs_from_txt,integrate_fx_coeffs
import pennylane as qml
from functools import partial
from pennylane.fourier import coefficients


'''
   Read from config file.
   USAGE example: python main.py -in myconfig.json 
 '''
parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', type=str, required=True)
args = parser.parse_args()

print(args.input)

try:
    with open(args.input) as run_config:
        run_configuration = json.load(run_config)
        print(' Reading config file...')
except Exception as exc:
    print("Error in reading temp config: Process aborted")
    print(f" Error retriven: ({exc})")

if __name__ == "__main__":
    np.random.seed(4)
    
    ##READ CONFIGURATION
    layers = int(run_configuration['layers'])
    qubits = int(run_configuration['qubits'])
    max_steps = int(run_configuration['max_steps'])
    step_size=float(run_configuration['step_size'])
    verbose=bool(run_configuration['verbose'])
    verbose_plot=bool(run_configuration['verbose_plot'])
    save_coeffs=bool(run_configuration['save_coeffs'])
    which_func=str(run_configuration['function'])
    
    x_max=float(run_configuration['x_max'])
    x_min=float(run_configuration['x_min'])
    nbit = int(run_configuration['nbit'])
    shots = int(run_configuration['shots'])
    
    train_samples = int(run_configuration['train_samples'])
    test_samples = int(run_configuration['test_samples'])

    lins_tr: float = np.pi
    lins_te = np.pi + 0.0001
    xtrain = np.linspace(-lins_tr, lins_tr, train_samples)
    xtest = np.linspace(-lins_te, lins_te, test_samples)
    
    if which_func== "example":
        function_train = funct_example(xtrain)
        function_test = funct_example(xtest)
        out_txt = 'example'
    else:
        print('no predefined function choosen')

    #We normalize the function
    norm=max(max(abs(function_train)),max(abs(function_test)))
    function_train=function_train/norm
    function_test=function_test/norm

    # QUANTUM PART
    dev = qml.device('lightning.qubit', wires=1)

    r = layers  # number of times the encoding gets repeated (here equal to the number of layers)
    weights = 2 * npy.pi * npy.random.random(size=(r + 1, 3), requires_grad=True)  # some random initial weights

    opt = qml.AdamOptimizer(step_size)
    cst = [cost_function(xtrain,function_train, weights)]  # initial cost
    w_list = [weights]
    for step in range(max_steps):
        #weights,c = opt.step_and_cost(lambda w: cost_function(xtrain, w), weights)
        weights = opt.step(lambda w: cost_function(xtrain,function_train, w), weights)

        if (verbose):
            w_list.append(weights)
            c = cost_function(xtrain,function_train,  weights)
            cst.append(c)
            print("Cost at step {0:3}: {1}".format(step + 1, c))

    if(verbose):
        print('finish max steps')

    # ## Obtain the Fourier Coeff
    partial_circuit = partial(linear_ansatz, weights)
    num_inputs = 1
    degree = layers 
    coeffs = coefficients(partial_circuit, num_inputs, degree, lowpass_filter=True)
    #Non normalized coeffs
    coeffs=coeffs*norm
    

    ## Obtain the fourier estimation of the function and we plot it:
    if (verbose_plot):
        y_fourier = fourier_series(coeffs, len(xtest),)
        #We plot the original (non-normalized) function
        plot_fourier= plot_data_fourier(function_test,norm, r, weights, xtest, y_fourier)
        plot_fourier.savefig("output_plots_" + out_txt + "/plot_{}.png".format(which_func),dpi=500)
    trig_coeffs = exp_fourier_to_trig(coeffs)
    
    
    fourier_array=trig_to_final_array(trig_coeffs)
    print(fourier_array)
    if(save_coeffs):
        np.savetxt("output_coeff_" + out_txt + "/coefs_{}.txt".format(which_func),fourier_array)
    
    coeffsexample=coeffs_from_txt(fourier_array)
    resulexample=integrate_fx_coeffs(coeffsexample,x_max,x_min,nbit,shots)
    with open ('QFIAE_{}_ESTIMATIONS.txt'.format(which_func), 'a') as file:    
        file.write(str(resulexample))
        file.write('\n')
