import numpy as np
import H3_A
from itertools import zip_longest
from sklearn import svm


######################################################
################    IMPORTANT    #####################
######################################################
#######                                         ######
#######    No additional imports are allowed    ######
#######    You are just allowed to use:         ######
#######      numpy, H3_A, zip_longest and svm   ######
#######                                         ######
######################################################
######################################################

def unitize(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    vector1 = vector1 / norm1
    vector2 = vector2 / norm2

    return vector1, vector2, norm1, norm2


def make_quantum_vector(vector1d, vector2d):
    length_vector = len(vector1d)
    n = np.log2(length_vector)
    num_qubits = int(np.ceil(n))
    zerovector = [0] * 2 ** num_qubits
    quantum_vector1 = np.array([x + y for x, y in zip_longest(zerovector, vector1d, fillvalue=0)])
    quantum_vector2 = np.array([x + y for x, y in zip_longest(zerovector, vector2d, fillvalue=0)])

    return quantum_vector1, quantum_vector2, num_qubits


def initializer_circuit(num_qubits, quantum_vector1, quantum_vector2):
    quantum_vector = 1/np.sqrt(2) * (np.kron(quantum_vector1, np.array([1, 0])) +
                                     np.kron(quantum_vector2, np.array([0, 1])))
    # produce the state phi
    inner_product_circuit = H3_A.QuantumCircuit(num_qubits + 1)
    inner_product_circuit.init(quantum_vector)

    return inner_product_circuit


def compute_q_inner_product(a, b):  # 5p

    vector1d, vector2d, norm1, norm2 = unitize(a, b)
    quantum_vector1, quantum_vector2, num_qubits = make_quantum_vector(vector1d, vector2d)

    # Generate the quantum circuit to estimate the inner product
    inner_product_circuit = initializer_circuit(num_qubits, quantum_vector1, quantum_vector2)
    inner_product_circuit.h(0)

    num_shots = 1000
    count = {'0': 0, '1': 0}
    for _ in range(num_shots):

        first_qubit = inner_product_circuit.measure()[-1]
        if str(first_qubit) == "0":
            count["0"] += 1
        elif str(first_qubit) == "1":
            count["1"] += 1

    if (count['0'] == num_shots): count['1'] = 0
    p = count["0"] / (count["0"] + count["1"])

    estimated_inner_product = 2 * p - 1
    estimated_inner_product_denormed = estimated_inner_product * norm1 * norm2

    return estimated_inner_product_denormed


def QKernelMatrix(X1, X2):  # 2p
    qkernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            qkernel_matrix[i, j] = compute_q_inner_product(x1, x2)

    return qkernel_matrix


def main():  # 3p

    X = np.array([[0.1, 0.75], [0.2, 0.55], [0.3, 0.5], [0.3, 0.7], [0.5, 0.5], [0.5, 0.8], [0.6, 0.65], [0.7, 0.8]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    q_svm = svm.SVC(kernel=QKernelMatrix)
    q_svm.fit(X, y)

    return q_svm
