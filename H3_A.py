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

    I = np.identity(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, 0-1j], [0+1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = (2 ** (-1 / 2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    SWAP = (np.kron(I, I) + np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) / 2
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.array([1 if i == 0 else 0 for i in range(2**num_qubits)], dtype=int)
        self.unitary = np.identity(2 ** num_qubits, dtype=int)

    def init(self, state: iter):
        self.state = np.asarray(state) / np.linalg.norm(np.asarray(state))
        self.unitary = np.identity(2**self.num_qubits)

    def h(self, qbit):
        mats = [QuantumCircuit.H if i == qbit else QuantumCircuit.I for i in range(self.num_qubits)]
        U = 1
        for m in mats: U = np.kron(m, U)

        self.unitary = np.matmul(U, self.unitary)
        self.state = self.unitary.dot(self.state)

    def x(self, qbit):
        mats = [QuantumCircuit.X if i == qbit else QuantumCircuit.I for i in range(self.num_qubits)]
        U = 1
        for m in mats: U = np.kron(m, U)

        self.unitary = np.matmul(U, self.unitary)
        self.state = self.unitary.dot(self.state)

    def y(self, qbit):
        mats = [QuantumCircuit.Y if i == qbit else QuantumCircuit.I for i in range(self.num_qubits)]
        U = 1
        for m in mats: U = np.kron(m, U)

        self.unitary = np.matmul(U, self.unitary)
        self.state = self.unitary.dot(self.state)

    def z(self, qbit):
        mats = [QuantumCircuit.Z if i == qbit else QuantumCircuit.I for i in range(self.num_qubits)]
        U = 1
        for m in mats: U = np.kron(m, U)

        self.unitary = np.matmul(U, self.unitary)
        self.state = self.unitary.dot(self.state)

    def swap(self, qubit_nr):
        I = QuantumCircuit.I
        sw = QuantumCircuit.SWAP
        mats = [sw if i == qubit_nr else I for i in range(self.num_qubits - 1)]
        U = 1
        for m in mats: U = np.kron(m, U)

        self.unitary = np.matmul(U, self.unitary)
        self.state = np.dot(U, self.state)

    def cnot(self, control_qubit, target_qubit):
        ################################
        ### Multiple Swaps ###
        for i in range(control_qubit, target_qubit-1):
            QuantumCircuit.swap(self, i)
        ################################
        I = QuantumCircuit.I
        sCNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        mats = [sCNOT if i == target_qubit - 1 else I for i in range(self.num_qubits - 1)]
        U = 1
        for m in mats: U = np.kron(m, U)
        self.unitary = np.matmul(U, self.unitary)
        self.state = np.dot(U, self.state)
        ################################
        ### Multiple Swaps ###
        for i in range(target_qubit-2, control_qubit-1, -1):
            QuantumCircuit.swap(self, i)
        ################################

    def measure(self):
        vecs = [np.array([1 if z == i else 0 for z in
                          range(2 ** self.num_qubits)]) for i in range(2 ** self.num_qubits)]
        p = [abs(np.dot(x.conj().T, self.state)) ** 2 for x in vecs]
        sample_nr = np.random.choice([i for i in range(2 ** self.num_qubits)], p=p)
        sample = np.array([int(c) for c in list(np.binary_repr(sample_nr, width=self.num_qubits))])

        return sample

    def get_unitary(self):
        return self.unitary
