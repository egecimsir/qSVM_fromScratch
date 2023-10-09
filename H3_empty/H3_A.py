import numpy as np



######################################################
#######             IMPORTANT                   ######
#######                                         ######
#######    No additional imports are allowed    ######
#######    You are just allowed to use numpy    ######
#######                                         ######
#######    Use the Qiskit definition for the    ######
#######    qubit ordering!                      ######
#######                                         ######
######################################################



class QuantumCircuit():

    def __init__(self, num_qubits): #4p
        self.num_qubits = None # TODO
        self.state = None # TODO: start in state |00..00>
        self.unitary = None # TODO: keep track of the overall unity of the quantum circuit

    def init(self, state): #1p
        self.state = None # TODO: initialize the quantum circuit with a specified starting state

    def h(self, qubit_nr): #6p
        t = None # TODO: calculate the unitary matrix t which represents a hadamard gate on qubit 'qubit_nr'
                 # TODO:    and identity matrices on all other qubits
        self.unitary = np.matmul(t, self.unitary)

    def x(self, qubit_nr): #3p
        t = None # TODO: calculate the unitary matrix t which represents a X gate on qubit 'qubit_nr'
                 # TODO:    and identity matrices on all other qubits
        self.unitary = np.matmul(t, self.unitary)

    def y(self, qubit_nr): #3p
        t = None # TODO: calculate the unitary matrix t which represents a Y gate on qubit 'qubit_nr'
                 # TODO:    and identity matrices on all other qubits
        self.unitary = np.matmul(t, self.unitary)

    def z(self, qubit_nr): #3p
        t = None # TODO: calculate the unitary matrix t which represents a Z gate on qubit 'qubit_nr'
                 # TODO:    and identity matrices on all other qubits
        self.unitary = np.matmul(t, self.unitary)

    # swap qubits 'qubit_nr' and 'qubit_nr + 1'
    def swap(self, qubit_nr): #6
        t = None # TODO: calculate the unitary matrix t which represents a SWAP gate on qubits
                 # TODO:    'qubit_nr' and 'qubit_nr + 1' and identity matrices on all other qubits
        self.unitary = np.matmul(t, self.unitary)

    def cnot(self, control_qubit, target_qubit): #8p

        # TODO: apply multiple SWAPs

        t = None # TODO: calculate the unitary matrix t which represents a CNOT gate on qubits
                 # TODO:    'control_qubit' and 'target_qubit' and identity matrices on all other qubits.
                 # TODO:    For simplicity, you can assume that control_qubit < target_qubit
        self.unitary = np.matmul(t, self.unitary)

        # TODO: apply multiple SWAPs

    # apply the start_vector to the circuit and sample a measure state
    # this function returns a binary vector of length num_qubits
    # use the function np.random.choice to determine the collapse of the state vector
    # Example: the state   1/sqrt(2) * [1 0 1 0]^T   should lead to the measure state [0 0]^T with probability 0.5
    #          and to the measure state [1 0]^T with probability 0.5
    def measure(self): #4p
        p = None # TODO: calculate the probability distribution over all possible states
        sample_nr = None # TODO: sample a state using np.random.choice
        sample = np.array([int(c) for c in list(np.binary_repr(sample_nr, width=self.num_qubits))])
        return sample

    def get_unitary(self): #2p
        return None # TODO: return the overall unitary matrix which represents the quantum circuit


