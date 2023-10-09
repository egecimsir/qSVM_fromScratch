import H3_A
import H3_B
from qiskit import QuantumCircuit, Aer, execute
import numpy as np



##############################################
###   IMPORTANT: DO NOT CHANGE THIS FILE   ###
##############################################


# You can use this file to check you code
# If all 9 test are correct this is a good indicator that your code is correct
# Note: passing all tests does not imply getting all 50p.
# We will check your code for cheating and we will run additional tests
# Further, we want to remind you that you have to solve these tasks on your own.
# We will check all submissions for plagiarism!



def test_1():

    qc = H3_A.QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 1 COMPLETED")
    else:
        print("TEST 1 FAILED")

def test_2():

    qc = H3_A.QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.x(1)
    qc.x(2)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.x(1)
    qc.x(2)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 2 COMPLETED")
    else:
        print("TEST 2 FAILED")

def test_3():

    qc = H3_A.QuantumCircuit(3)
    qc.y(0)
    qc.y(1)
    qc.y(2)
    qc.y(2)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(3)
    qc.y(0)
    qc.y(1)
    qc.y(2)
    qc.y(2)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 3 COMPLETED")
    else:
        print("TEST 3 FAILED")

def test_4():

    qc = H3_A.QuantumCircuit(3)
    qc.z(0)
    qc.z(1)
    qc.z(0)
    qc.z(2)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(3)
    qc.z(0)
    qc.z(1)
    qc.z(0)
    qc.z(2)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 4 COMPLETED")
    else:
        print("TEST 4 FAILED")

def test_5():

    qc = H3_A.QuantumCircuit(4)
    qc.swap(0)
    qc.swap(1)
    qc.swap(1)
    qc.swap(1)
    qc.swap(0)
    qc.swap(2)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(4)
    qc.swap(0,1)
    qc.swap(1,2)
    qc.swap(1,2)
    qc.swap(1,2)
    qc.swap(0,1)
    qc.swap(2,3)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 5 COMPLETED")
    else:
        print("TEST 5 FAILED")

def test_6():

    qc = H3_A.QuantumCircuit(3)
    qc.cnot(0,2)
    qc.cnot(1,2)
    qc.cnot(0,2)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(3)
    qc.cnot(0,2)
    qc.cnot(1,2)
    qc.cnot(0,2)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 6 COMPLETED")
    else:
        print("TEST 6 FAILED")

def test_7():

    qc = H3_A.QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.cnot(0,2)
    qc.swap(0)
    qc.x(0)
    qc.swap(0)
    qc.y(2)
    qc.h(1)
    qc.cnot(1,2)
    qc.swap(0)
    qc.h(2)
    qc.x(1)
    qc.cnot(0,2)
    qc.x(2)
    qc.swap(1)
    qc.y(0)
    qc.z(1)
    my_unitary = qc.get_unitary()

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.cnot(0,2)
    qc.swap(0, 1)
    qc.x(0)
    qc.swap(0, 1)
    qc.y(2)
    qc.h(1)
    qc.cnot(1,2)
    qc.swap(0, 1)
    qc.h(2)
    qc.x(1)
    qc.cnot(0,2)
    qc.x(2)
    qc.swap(1, 2)
    qc.y(0)
    qc.z(1)
    backend = Aer.get_backend('unitary_simulator')
    qiskit_unitary = execute(qc, backend).result().get_unitary()

    if np.allclose(my_unitary, np.asarray(qiskit_unitary)):
        print("TEST 7 COMPLETED")
    else:
        print("TEST 7 FAILED")

def test_8():

    correct = True

    a = [0.1, 0.75]
    b = [0.2, 0.55]
    if np.abs(H3_B.compute_q_inner_product(a,b) - np.dot(a,b)) > 0.03:
        correct = False

    a = [-0.3, 0.45]
    b = [0.25, 0.35]
    if np.abs(H3_B.compute_q_inner_product(a,b) - np.dot(a,b)) > 0.03:
        correct = False

    a = [0.44, 0.15]
    b = [-0.3, -0.45]
    if np.abs(H3_B.compute_q_inner_product(a,b) - np.dot(a,b)) > 0.03:
        correct = False

    if correct:
        print("TEST 8 COMPLETED")
    else:
        print("TEST 8 FAILED")

def test_9():
    q_svm = H3_B.main()
    X_test = np.array([[0.2, 0.65], [0.2, 0.5], [0.25, 0.6], [0.55, 0.45], [0.55, 0.45], [0.8, 1.0]])
    y_test = np.array([1, 1, 1, -1, -1, -1])
    q_pred = q_svm.predict(X_test)
    if np.allclose(y_test, q_pred):
        print("TEST 9 COMPLETED")
    else:
        print("TEST 9 FAILED")



test_1()
test_2()
test_3()
test_4()
test_5()
test_6()
test_7()
test_8()
test_9()
