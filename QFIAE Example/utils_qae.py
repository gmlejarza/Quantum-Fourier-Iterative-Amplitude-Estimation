import math
from math import pi
from qiskit import Aer

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.algorithms import EstimationProblem
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit import BasicAer
from qiskit.utils import QuantumInstance


def P(qc, qx, nbit):
    """
        Generating uniform probability distribution
            qc: quantum circuit
            qx: quantum register
            nbit: number of qubits
        The inverse of P = P
    """
    qc.h(qx)

def R(qc, qx, qx_measure, nbit, b_max,b_min):
    """
        Computing the integral function f()
            qc: quantum circuit
            qx: quantum register
            qx_measure: quantum register for measurement
            nbit: number of qubits
            b_max: upper limit of integral            
    """
    #qc.ry(2*b_min, qx_measure)
    qc.ry((b_max-b_min) / 2**nbit * 2 * 0.5+2*b_min, qx_measure)
    for i in range(nbit):
        qc.cu(2**i * (b_max-b_min) / 2**nbit * 2, 0, 0,0, qx[i], qx_measure[0])
        #qc.cu3(2*b_min, 0, 0, qx[i], qx_measure[0])
def Rinv(qc, qx, qx_measure, nbit, b_max,b_min):
    """
        The inverse of R
            qc: quantum circuit
            qx: quantum register
            qx_measure : quantum register for measurement
            nbit: number of qubits
            b_max: upper limit of integral
    """
    for i in range(nbit)[::-1]:
        qc.cu(-2**i * (b_max-b_min) / 2**nbit * 2, 0,0, 0, qx[i], qx_measure[0])
        #qc.cu3(-2*b_min, 0, 0, qx[i], qx_measure[0])
    qc.ry(-(b_max-b_min) / 2**nbit * 2 * 0.5-2*b_min, qx_measure)
    #qc.ry(-2*b_min, qx_measure)
def reflect(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
        Computing reflection operator (I - 2|0><0|)
            qc: quantum circuit
            qx: quantum register
            qx_measure: quantum register for measurement
            qx_ancilla: temporal quantum register for decomposing multi controlled NOT gate
            nbit: number of qubits
            b_max: upper limit of integral
    """
    for i in range(nbit):
        qc.x(qx[i])
    qc.x(qx_measure[0])
    qc.barrier()    #format the circuits visualization
    multi_control_NOT(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
    qc.barrier()    #format the circuits visualization
    qc.x(qx_measure[0])
    for i in range(nbit):
        qc.x(qx[i])
def multi_control_NOT(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
        Computing multi controlled NOT gate
            qc: quantum circuit
            qx: quantum register
            qx_measure: quantum register for measurement
            qx_ancilla: temporal quantum register for decomposing multi controlled NOT gate
            nbit: number of qubits
            b_max: upper limit of integral
    """

    if nbit == 1:
        qc.cz(qx[0], qx_measure[0])
    elif nbit == 2:
        qc.h(qx_measure[0])
        qc.ccx(qx[0], qx[1], qx_measure[0])
        qc.h(qx_measure[0])
    elif nbit > 2.0:
        qc.ccx(qx[0], qx[1], qx_ancilla[0])
        for i in range(nbit - 3):
            qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
        qc.h(qx_measure[0])
        qc.ccx(qx[nbit - 1], qx_ancilla[nbit - 3], qx_measure[0])
        qc.h(qx_measure[0])
        for i in range(nbit - 3)[::-1]:
            qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
        qc.ccx(qx[0], qx[1], qx_ancilla[0])
def A_operator(nbit,b_max,b_min):
    #defining A
    qx = QuantumRegister(nbit)
    qx_measure = QuantumRegister(1)
    A = QuantumCircuit(qx, qx_measure)
    
    #P
    P(A,qx,nbit)
    
    #R
    R(A, qx, qx_measure, nbit, b_max,b_min)
    
    return A
def Q_operator(nbit,b_max,b_min):
    qx_ancilla = QuantumRegister(nbit - 2)
    qx = QuantumRegister(nbit)
    qx_measure = QuantumRegister(1)
    Q = QuantumCircuit(qx, qx_ancilla,qx_measure)
    
    Q.z(qx_measure[0])
    Rinv(Q, qx, qx_measure, nbit, b_max,b_min)
    #Q.barrier()    #format the circuits visualization
    P(Q, qx, nbit)
    reflect(Q, qx, qx_measure, qx_ancilla, nbit, b_max)
    P(Q, qx, nbit)
    #qc.barrier()    #format the circuits visualization
    R(Q, qx, qx_measure, nbit, b_max,b_min)
    
    return Q
def IQAE(nbit,b_max,b_min,shots):
    A=A_operator(nbit,b_max,b_min)
    Q=Q_operator(nbit,b_max,b_min)
    problem = EstimationProblem(
    state_preparation=A,  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=[nbit],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
    )
    backend = BasicAer.get_backend("statevector_simulator")
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend,shots=shots)
    iae = IterativeAmplitudeEstimation(
    epsilon_target=0.005,  # target accuracy
    alpha=0.05,  # width of the confidence interval
    quantum_instance=quantum_instance,
    )
    iae_result = iae.estimate(problem)
    return(iae_result.estimation)
class Fourier_terms:
    def __init__(self, coefficient, angle,function):
        self.coeff = coefficient
        self.angle=angle
        self.function =function #-1 independent term, 1 sine, 0 cosine
    

def integrate_fx_coeffs(coeffs_f,x_max,x_min,nbit,shots):
    
    #L=2*x_max
    mc_integral=0
    mc_integral_ind=0
    if (coeffs_f[0].function==-1):
        mc_integral_ind=coeffs_f[0].coeff            
    for n in range(len(coeffs_f)):
        angle=coeffs_f[n].angle
        if(angle!=0):

            if (coeffs_f[n].function==0):#cosine
                a_estimated=IQAE(nbit,float(x_max*angle/2),float(x_min*angle/2),shots)
                mc_integral+=(1-a_estimated-a_estimated)*coeffs_f[n].coeff
            if (coeffs_f[n].function==1):#sine
                a_estimated=IQAE(nbit,float(x_max*angle/2-math.pi/4),
                                       float(x_min*angle/2-math.pi/4),shots)
                mc_integral+=(1-a_estimated-a_estimated)*coeffs_f[n].coeff
            
        else:
            if(coeffs_f[n].function==0):
                mc_integral_ind+=coeffs_f[n].coeff
    return ((mc_integral+mc_integral_ind)*(x_max-x_min)**2)
def coeffs_from_txt(fouriertxt):
    outclass=[]
    for i in range(len(fouriertxt)):
        out_class=Fourier_terms(fouriertxt[i,2],int(fouriertxt[i,1]),int(fouriertxt[i,0]))
        outclass.append(out_class)
    return outclass

