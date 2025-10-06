from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import hashlib
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Global Constants ------------------
EAVESDROP_PROB = 0.005  # 0.5% eavesdropping probability
ERROR_THRESHOLD = 0.05  # 5% error rate threshold
SHOTS = 1# Number of quantum measurements
# ------------------ Parameter Settings ------------------
def get_N():
    while True:
        try:
            N = int(input('Please enter the number of GHZ-like states N to prepare (positive integer): '))
            if N > 0:
                return N
            else:
                print('N must be a positive integer!')
        except Exception:
            print('Please enter a valid positive integer!')

# ------------------ Utility Functions ------------------
def random_bitstring(n):
    return ''.join(str(random.randint(0, 1)) for _ in range(n))

def H1(x, n):
    # Hash function, outputs binary string of length n
    h = hashlib.sha256(x.encode()).hexdigest()
    b = bin(int(h, 16))[2:].zfill(256)
    return b[:n]

def H2(x, n):
    # Another hash function, outputs binary string of length n
    h = hashlib.sha256(x.encode()).hexdigest()
    b = bin(int(h, 16))[2:].zfill(256)
    return b[:n]

def xor_bitstring(a, b):
    return ''.join(str(int(x)^int(y)) for x, y in zip(a, b))


# ------------------ GHZ-like State Preparation ------------------
def create_GHZ_like(qc, qubits, G_type):
    # qubits: [q0, q1, q2, q3]
    # According to requirements: first prepare four-particle |GHZ>_0 or |GHZ>_1, then apply Hadamard to four qubits to get |G0> or |G1>
    # |GHZ>_0=(|0000>+|1111>)/√2；|GHZ>_1=(|0101>+|1010>)/√2
    q0, q1, q2, q3 = qubits
    # Prepare |GHZ>_0
    qc.h(q0)
    qc.cx(q0, q1)
    qc.cx(q0, q2)
    qc.cx(q0, q3)
    if G_type == 'G1':
        # Transform from |GHZ>_0 to |GHZ>_1: apply X to 2nd and 4th particles
        qc.x(q1)
        qc.x(q3)
    elif G_type != 'G0':
        raise ValueError('Invalid GHZ-like type')
    # Apply H to four particles to get |G0> or |G1>
    qc.h(q0)
    qc.h(q1)
    qc.h(q2)
    qc.h(q3)

# ------------------ Decoy State Processing ------------------
def prepare_decoy_state(qc, q, decoy_type):
    # decoy_type: '0', '1', '+', '-'
    if decoy_type == '0':
        pass  # |0>
    elif decoy_type == '1':
        qc.x(q)
    elif decoy_type == '+':
        qc.h(q)
    elif decoy_type == '-':
        qc.x(q)
        qc.h(q)
        qc.z(q)
    else:
        raise ValueError('Invalid decoy type')

def measure_decoy_state(qc, q, basis, c):
    # basis: 'Z' or 'X'
    if basis == 'Z':
        qc.measure(q, c)
    elif basis == 'X':
        qc.h(q)
        qc.measure(q, c)
    else:
        raise ValueError('Invalid measurement basis')

def insert_decoy(qc, qubits, decoy_type, position, basis, c):
    # position: 'before' or 'after'
    # basis: 'Z' or 'X'
    decoy_qubit = max(qubits) + 1
    if position == 'before':
        prepare_decoy_state(qc, decoy_qubit, decoy_type)
        measure_decoy_state(qc, decoy_qubit, basis, c)
    else:  # after
        prepare_decoy_state(qc, decoy_qubit, decoy_type)
        measure_decoy_state(qc, decoy_qubit, basis, c)

def verify_decoy_state(measured, key_bit, ghz_type, position):

    if key_bit == '0':  # Z-basis measurement
        if position == 'before':
            expected = '0' if ghz_type == 'G0' else '1'
            return (measured == expected, expected, measured)
        else:  # position == 'after'
            return (False, None, measured)  # Should not insert after Z-basis measurement
    else:  # key_bit == '1', X-basis measurement
        if position == 'after':
            expected = '0' if ghz_type == 'G0' else '1'  # |+> state measures as 0, |-> state measures as 1
            return (measured == expected, expected, measured)
        else:  # position == 'before'
            return (False, None, measured)  # Should not insert before X-basis measurement

def calculate_error_rate(decoy_results):
    # Calculate error rate of decoy states
    total = len(decoy_results)
    if total == 0:
        return 0.0
    errors = sum(1 for result in decoy_results if not result[0])
    return errors / total

def check_eavesdropping(error_rate, threshold=ERROR_THRESHOLD):
    # Determine if eavesdropping exists based on error rate
    # threshold: error rate threshold, exceeding this value indicates eavesdropping
    return error_rate > threshold

def remove_decoy_states(sequence):
    # Remove decoy states from sequence, keep only GHZ-like state particle pairs
    return [item for item in sequence if isinstance(item, tuple)]

# --------- Type and Key Derivation Based on Decoy States (Common Implementation for Alice/Bob) ---------
def deduce_types_and_keys_from_Alice_sequences(S_A1, S_A2, K_AC1, N):
    TypeA_deduced, TypeB_deduced = [], []
    KR1_deduced, KR2_deduced = '', ''
    S_A1_clean, S_A2_clean = [], []
    for i in range(N):
        seg1 = S_A1[3*i:3*i+3]
        if K_AC1[i] == '0':
            dec = seg1[0]
            if dec == '|0>':
                TypeA_deduced.append('G0'); KR1_deduced += '0'
            else:
                TypeA_deduced.append('G1'); KR1_deduced += '1'
            S_A1_clean.extend([seg1[1], seg1[2]])
        else:
            dec = seg1[2]
            if dec == '|+>':
                TypeA_deduced.append('G0'); KR1_deduced += '0'
            else:
                TypeA_deduced.append('G1'); KR1_deduced += '1'
            S_A1_clean.extend([seg1[0], seg1[1]])
        seg2 = S_A2[3*i:3*i+3]
        if K_AC1[i] == '0':
            dec2 = seg2[0]
            if dec2 == '|0>':
                TypeB_deduced.append('G0'); KR2_deduced += '0'
            else:
                TypeB_deduced.append('G1'); KR2_deduced += '1'
            S_A2_clean.extend([seg2[1], seg2[2]])
        else:
            dec2 = seg2[2]
            if dec2 == '|+>':
                TypeB_deduced.append('G0'); KR2_deduced += '0'
            else:
                TypeB_deduced.append('G1'); KR2_deduced += '1'
            S_A2_clean.extend([seg2[0], seg2[1]])
    return S_A1_clean, S_A2_clean, TypeA_deduced, TypeB_deduced, KR1_deduced, KR2_deduced

def deduce_types_and_keys_from_Bob_sequences(S_B1, S_B2, K_BC2, N):
    TypeA_deduced, TypeB_deduced = [], []
    KR1_deduced, KR2_deduced = '', ''
    S_B1_clean, S_B2_clean = [], []
    for i in range(N):
        seg1 = S_B1[3*i:3*i+3]
        if K_BC2[i] == '0':
            dec = seg1[0]
            if dec == '|0>':
                TypeA_deduced.append('G0'); KR1_deduced += '0'
            else:
                TypeA_deduced.append('G1'); KR1_deduced += '1'
            S_B1_clean.extend([seg1[1], seg1[2]])
        else:
            dec = seg1[2]
            if dec == '|+>':
                TypeA_deduced.append('G0'); KR1_deduced += '0'
            else:
                TypeA_deduced.append('G1'); KR1_deduced += '1'
            S_B1_clean.extend([seg1[0], seg1[1]])
        seg2 = S_B2[3*i:3*i+3]
        if K_BC2[i] == '0':
            dec2 = seg2[0]
            if dec2 == '|0>':
                TypeB_deduced.append('G0'); KR2_deduced += '0'
            else:
                TypeB_deduced.append('G1'); KR2_deduced += '1'
            S_B2_clean.extend([seg2[1], seg2[2]])
        else:
            dec2 = seg2[2]
            if dec2 == '|+>':
                TypeB_deduced.append('G0'); KR2_deduced += '0'
            else:
                TypeB_deduced.append('G1'); KR2_deduced += '1'
            S_B2_clean.extend([seg2[0], seg2[1]])
    return S_B1_clean, S_B2_clean, TypeA_deduced, TypeB_deduced, KR1_deduced, KR2_deduced

# ------------------ Bell State Measurement ------------------
def bell_measure_and_result(qc, q0, q1, q2, q3, c0, c1, c2, c3):
    """
    Perform Bell measurement on four-particle GHZ-like state
    q0,q1: first pair of particles (1,2)
    q2,q3: second pair of particles (3,4)
    c0,c1: measurement results of first pair
    c2,c3: measurement results of second pair
    """
    # Bell measurement on first pair (1,2)
    qc.cx(q0, q1)  # CNOT gate
    qc.h(q0)       # Hadamard gate
    qc.measure(q0, c1)  # Measure first particle
    qc.measure(q1, c0)  # Measure second particle

    # Bell measurement on second pair (3,4)
    qc.cx(q2, q3)  # CNOT gate
    qc.h(q2)       # Hadamard gate
    qc.measure(q2, c3)  # Measure third particle
    qc.measure(q3, c2)  # Measure fourth particle

# Bell state mapping
bell_map = {
    # Complete mapping: |B00>->00, |B01>->01, |B10>->10, |B11>->11
    '00': '00',
    '01': '01',
    '10': '10',
    '11': '11',
}

def bitwise_not(bits):
    return ''.join('1' if b == '0' else '0' for b in bits)

def get_bell_result(measured):
    # measured: 2-bit string, high bit first
    return bell_map.get(measured, None)

def infer_other(ms, ghz_type):
    # Under new definition, both pairs have same distribution, so other party's measurement result equals own result
    return ms

# ------------------ Main Process ------------------
def main():
    global N
    N = get_N()
    
    # -------- Output Formatting Helper --------
    def fmt_seq(seq, max_items=12):
        if len(seq) <= max_items:
            return str(seq)
        head = ', '.join(map(str, seq[:max_items]))
        return f"[{head}, ...]  (len={len(seq)})"

    # 1. Pre-shared keys
    K_AC1 = random_bitstring(N)  # Key shared between Alice and S_C1
    K_BC2 = random_bitstring(N)  # Key shared between Bob and S_C2
    K_C1C2 = random_bitstring(N)  # Key shared between S_C1 and S_C2
    print(f"Alice and SC1 pre-share K_AC1={K_AC1}")
    print(f"Bob and SC2 pre-share K_BC2={K_BC2}")
    print(f"SC1 and SC2 pre-share K_C1C2={K_C1C2}")

    # 2. S_C1 and S_C2 generate random numbers R1 and R2 respectively, calculate KR1 and KR2
    R1 = random_bitstring(N)  # Random number generated by S_C1
    R2 = random_bitstring(N)  # Random number generated by S_C2
    KR1 = H1(K_C1C2 + R1, N)  # Key calculated by S_C1
    KR2 = H1(K_C1C2 + R2, N)  # Key calculated by S_C2
    print(f"R1: {R1}\nR2: {R2}")
    print(f"KR1: {KR1}\nKR2: {KR2}")

    # 3. Determine TypeA and TypeB based on KR1 and KR2
    TypeA = ['G0' if bit == '0' else 'G1' for bit in KR1]  # GHZ-like state type prepared by S_C1
    TypeB = ['G0' if bit == '0' else 'G1' for bit in KR2]  # GHZ-like state type prepared by S_C2
    print(f"TypeA: {TypeA}\nTypeB: {TypeB}")

    # 4. Exchange pre-shared keys
    H2_KC1C2 = H2(K_C1C2, N)
    S_C1_to_S_C2 = xor_bitstring(H2_KC1C2, K_AC1)  # Information sent from S_C1 to S_C2
    S_C2_to_S_C1 = xor_bitstring(H2_KC1C2, K_BC2)  # Information sent from S_C2 to S_C1
    K_BC2_recovered = xor_bitstring(S_C2_to_S_C1, H2_KC1C2)  # K_BC2 recovered by S_C1
    K_AC1_recovered = xor_bitstring(S_C1_to_S_C2, H2_KC1C2)  # K_AC1 recovered by S_C2
    print(f"K_BC2 recovered: {K_BC2_recovered == K_BC2}")
    print(f"K_AC1 recovered: {K_AC1_recovered == K_AC1}")

    # Initialize sequences
    S_A1 = []  # Sequence prepared by S_C1, containing {P1i(1),P1i(2)}
    S_B1 = []  # Sequence prepared by S_C1, containing {P1i(3),P1i(4)}
    S_A2 = []  # Sequence prepared by S_C2, containing {P2i(1),P2i(2)}
    S_B2 = []  # Sequence prepared by S_C2, containing {P2i(3),P2i(4)}
    
    # Initialize simulator
    simulator = AerSimulator()
    
    # Initialize keys
    KA = ''  # Alice's final key
    KB = ''  # Bob's final key

    # 1. S_C1 and S_C2 prepare GHZ-like states and group them respectively
    for i in range(N):
        # S_C1 prepares GHZ-like state
        qc1 = QuantumCircuit(4, 4)  # 4 qubits for GHZ-like state, 4 classical bits for measurement
        create_GHZ_like(qc1, [0,1,2,3], TypeA[i])  # Use TypeA[i] to prepare |G0> or |G1>
        
        # S_C2 prepares GHZ-like state
        qc2 = QuantumCircuit(4, 4)  # 4 qubits for GHZ-like state, 4 classical bits for measurement
        create_GHZ_like(qc2, [0,1,2,3], TypeB[i])  # Use TypeB[i] to prepare |G0> or |G1>
        
        # Group particles into sequences
        # S_C1's sequences
        S_A1.extend([(i,0), (i,1)])  # P1i(1),P1i(2)
        S_B1.extend([(i,2), (i,3)])  # P1i(3),P1i(4)
        
        # S_C2's sequences
        S_A2.extend([(i,0), (i,1)])  # P2i(1),P2i(2)
        S_B2.extend([(i,2), (i,3)])  # P2i(3),P2i(4)

    # 2. Insert decoy states
    # S_C1 inserts decoy states into S_A1
    S_A1 = []  # Reinitialize sequence
    for i in range(N):
        if K_AC1[i] == '0':  # Z-basis measurement
            if TypeA[i] == 'G0':
                S_A1.extend(['|0>', (i,0), (i,1)])  # Insert |0> before {P1i(1),P1i(2)}
            else:  # TypeA[i] == 'G1'
                S_A1.extend(['|1>', (i,0), (i,1)])  # Insert |1> before {P1i(1),P1i(2)}
        else:  # K_AC1[i] == '1', X-basis measurement
            if TypeA[i] == 'G0':
                S_A1.extend([(i,0), (i,1), '|+>'])  # Insert |+> after {P1i(1),P1i(2)}
            else:  # TypeA[i] == 'G1'
                S_A1.extend([(i,0), (i,1), '|->'])  # Insert |-> after {P1i(1),P1i(2)}
    
    # S_C1 inserts decoy states into S_B1
    S_B1 = []  # Reinitialize sequence
    for i in range(N):
        if K_BC2[i] == '0':  # Z-basis measurement
            if TypeA[i] == 'G0':
                S_B1.extend(['|0>', (i,2), (i,3)])  # Insert |0> before {P1i(3),P1i(4)}
            else:  # TypeA[i] == 'G1'
                S_B1.extend(['|1>', (i,2), (i,3)])  # Insert |1> before {P1i(3),P1i(4)}
        else:  # K_BC2[i] == '1', X-basis measurement
            if TypeA[i] == 'G0':
                S_B1.extend([(i,2), (i,3), '|+>'])  # Insert |+> after {P1i(3),P1i(4)}
            else:  # TypeA[i] == 'G1'
                S_B1.extend([(i,2), (i,3), '|->'])  # Insert |-> after {P1i(3),P1i(4)}
    
    # S_C2 inserts decoy states into S_A2
    S_A2 = []  # Reinitialize sequence
    for i in range(N):
        if K_AC1[i] == '0':  # Z-basis measurement
            if TypeB[i] == 'G0':
                S_A2.extend(['|0>', (i,0), (i,1)])  # Insert |0> before {P2i(1),P2i(2)}
            else:  # TypeB[i] == 'G1'
                S_A2.extend(['|1>', (i,0), (i,1)])  # Insert |1> before {P2i(1),P2i(2)}
        else:  # K_AC1[i] == '1', X-basis measurement
            if TypeB[i] == 'G0':
                S_A2.extend([(i,0), (i,1), '|+>'])  # Insert |+> after {P2i(1),P2i(2)}
            else:  # TypeB[i] == 'G1'
                S_A2.extend([(i,0), (i,1), '|->'])  # Insert |-> after {P2i(1),P2i(2)}
    
    # S_C2 inserts decoy states into S_B2
    S_B2 = []  # Reinitialize sequence
    for i in range(N):
        if K_BC2[i] == '0':  # Z-basis measurement
            if TypeB[i] == 'G0':
                S_B2.extend(['|0>', (i,2), (i,3)])  # Insert |0> before {P2i(3),P2i(4)}
            else:  # TypeB[i] == 'G1'
                S_B2.extend(['|1>', (i,2), (i,3)])  # Insert |1> before {P2i(3),P2i(4)}
        else:  # K_BC2[i] == '1', X-basis measurement
            if TypeB[i] == 'G0':
                S_B2.extend([(i,2), (i,3), '|+>'])  # Insert |+> after {P2i(3),P2i(4)}
            else:  # TypeB[i] == 'G1'
                S_B2.extend([(i,2), (i,3), '|->'])  # Insert |-> after {P2i(3),P2i(4)}
    
    print("\n================ Quantum State Sequence Generation Complete ================")
    print(f"SA_1' = {S_A1}")
    print(f"SB_1' = {S_B1}")
    print(f"SA_2' = {S_A2}")
    print(f"SB_2' = {S_B2}")
    print(f"Sequence length with decoys: SA_1'={len(S_A1)}, SB_1'={len(S_B1)}, SA_2'={len(S_A2)}, SB_2'={len(S_B2)}")
    print(f"Each sequence contains: {N} GHZ-like state particle pairs and {N} single-photon decoy states")
    print(f"Decoy state types: |0>, |1>, |+>, |->")
    print("\nDecoy state measurement bases:")
    print(f"K_AC1 = {K_AC1}  # where 0 represents Z-basis, 1 represents X-basis")
    print(f"K_BC2 = {K_BC2}  # where 0 represents Z-basis, 1 represents X-basis")
    
    # 3. Alice/Bob measure decoys and remove them based on their respective keys, derive Type and KR
    S_A1_measured, S_A2_measured, TypeA_A, TypeB_A, KR1_A, KR2_A = deduce_types_and_keys_from_Alice_sequences(S_A1, S_A2, K_AC1, N)
    S_B1_measured, S_B2_measured, TypeA_B, TypeB_B, KR1_B, KR2_B = deduce_types_and_keys_from_Bob_sequences(S_B1, S_B2, K_BC2, N)
    print("\n================ After Decoy State Measurement and Removal ================")
    print(f"Alice: | TypeA={TypeA_A} | TypeB={TypeB_A} | KR1={KR1_A} | KR2={KR2_A}")
    print(f"Bob  : | TypeA={TypeA_B} | TypeB={TypeB_B} | KR1={KR1_B} | KR2={KR2_B}")
    # Consistency check (should be consistent in protocol, shown here for demonstration)
    print(f"Type consistency (TypeA/TypeB): {TypeA_A==TypeA_B} / {TypeB_A==TypeB_B}")
    print(f"Key segment consistency (KR1/KR2): {KR1_A==KR1_B} / {KR2_A==KR2_B}")
    print(f"Original sequence length after decoy removal: SA_1={len(S_A1_measured)}, SB_1={len(S_B1_measured)}, SA_2={len(S_A2_measured)}, SB_2={len(S_B2_measured)}")
    
    # Decoy state verification
    decoy_results = []
    failed_verifications = []
    
    # Verify decoy states in S_A1
    for i in range(N):
        if K_AC1[i] == '0':  # Z-basis measurement
            position = 'before'
            expected_type = '0' if TypeA[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_AC1[i], TypeA[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_A1 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
        else:  # X-basis measurement
            position = 'after'
            expected_type = '0' if TypeA[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_AC1[i], TypeA[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_A1 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
    
    # Verify decoy states in S_B1
    for i in range(N):
        if K_BC2[i] == '0':  # Z-basis measurement
            position = 'before'
            expected_type = '0' if TypeA[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_BC2[i], TypeA[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_B1 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
        else:  # X-basis measurement
            position = 'after'
            expected_type = '0' if TypeA[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_BC2[i], TypeA[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_B1 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
    
    # Verify decoy states in S_A2
    for i in range(N):
        if K_AC1[i] == '0':  # Z-basis measurement
            position = 'before'
            expected_type = '0' if TypeB[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_AC1[i], TypeB[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_A2 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
        else:  # X-basis measurement
            position = 'after'
            expected_type = '0' if TypeB[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_AC1[i], TypeB[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_A2 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
    
    # Verify decoy states in S_B2
    for i in range(N):
        if K_BC2[i] == '0':  # Z-basis measurement
            position = 'before'
            expected_type = '0' if TypeB[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_BC2[i], TypeB[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_B2 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
        else:  # X-basis measurement
            position = 'after'
            expected_type = '0' if TypeB[i] == 'G0' else '1'
            measured = expected_type if random.random() > EAVESDROP_PROB else ('1' if expected_type == '0' else '0')
            result = verify_decoy_state(measured, K_BC2[i], TypeB[i], position)
            decoy_results.append(result)
            if not result[0]:
                failed_verifications.append(f"S_B2 decoy state {i+1}: position={position}, expected={expected_type}, actual={measured}")
    
    # Calculate error rate and eavesdropping detection
    error_rate = calculate_error_rate(decoy_results)
    is_eavesdropping = check_eavesdropping(error_rate)
    
    print("\n================ Decoy State Verification Results ================")
    if not failed_verifications:
        print("✓ All decoy state verifications passed")
    else:
        print("✗ Found failed decoy state verifications:")
        for failed in failed_verifications:
            print(f"  - {failed}")
        print(f"\nNote: Error rate ({error_rate:.2%}) {'exceeds' if is_eavesdropping else 'does not exceed'} security threshold (5%)")
    
    print(f"\nStatistics: Total={len(decoy_results)}, Failed={len(failed_verifications)}, Error rate={error_rate:.2%}")
    print(f"Eavesdropping detection: {'Eavesdropping detected' if is_eavesdropping else 'No eavesdropping detected'} (threshold 5%)")
    
    if is_eavesdropping:
        print("\nWarning: Possible eavesdropping detected, recommend terminating protocol!")
        return
    
    print("\n================ Continue with Bell State Measurement and Key Generation ================")
    
    # --- Bell measurement and key derivation ---
    MS_A1, MS_A2, MS_B1, MS_B2 = [], [], [], []
    for i in range(N):
        # S_C1 prepares GHZ-like state, Alice measures (1,2), Bob measures (3,4)
        qc1 = QuantumCircuit(4, 4)
        create_GHZ_like(qc1, [0,1,2,3], TypeA_A[i])
        bell_measure_and_result(qc1, 0, 1, 2, 3, 0, 1, 2, 3)
        result1 = simulator.run(qc1, shots=1).result()
        measured1 = list(result1.get_counts().keys())[0]
        ms_a1 = get_bell_result(measured1[0] + measured1[1])
        ms_b1 = get_bell_result(measured1[2] + measured1[3])
        MS_A1.append(ms_a1)
        MS_B1.append(ms_b1)
        # S_C2 prepares GHZ-like state, Alice measures (1,2), Bob measures (3,4)
        qc2 = QuantumCircuit(4, 4)
        create_GHZ_like(qc2, [0,1,2,3], TypeB_A[i])
        bell_measure_and_result(qc2, 0, 1, 2, 3, 0, 1, 2, 3)
        result2 = simulator.run(qc2, shots=1).result()
        measured2 = list(result2.get_counts().keys())[0]
        ms_a2 = get_bell_result(measured2[0] + measured2[1])
        ms_b2 = get_bell_result(measured2[2] + measured2[3])
        MS_A2.append(ms_a2)
        MS_B2.append(ms_b2)
    # Directly concatenate keys according to protocol: KA=MS_A1||MS_A2||KR1||KR2; KB similarly
    KA = ''.join(MS_A1) + ''.join(MS_A2) + KR1_A + KR2_A
    KB = ''.join(MS_B1) + ''.join(MS_B2) + KR1_B + KR2_B
    print("\nMeasurement strings:")
    print(f"MS_A1={MS_A1}")
    print(f"MS_A2={MS_A2}")
    print(f"MS_B1={MS_B1}")
    print(f"MS_B2={MS_B2}")
    print("\n================ Final Results ================")
    print(f"KA length: {len(KA)}, KB length: {len(KB)}, Expected length: {6*N}")
    print(f"KA: {KA}")
    print(f"KB: {KB}")
    print(f"Keys are consistent: {KA == KB}")
    if KA == KB:
        print("\nKey quality assessment:")
        print(f"Key length: {len(KA)} bits")
        print(f"Key randomness: {sum(1 for bit in KA if bit == '1')/len(KA):.2%} of bits are 1")
        print("Key can be used for secure communication")

    # ------- Draw quantum circuits for |G0> and |G1> preparation used in this experiment (colored) and display -------
    def build_ghz_like_circuit(g_type: str) -> QuantumCircuit:
        qc = QuantumCircuit(4, 4)
        # Consistent with create_GHZ_like
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        if g_type == 'G1':
            qc.x(1)
            qc.x(3)
        elif g_type != 'G0':
            raise ValueError('invalid type')
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        # Add Bell measurement for this experiment: pairs (0,1) and (2,3)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.cx(2, 3)
        qc.h(2)
        qc.measure(2, 2)
        qc.measure(3, 3)
        return qc

    def draw_ghz_like_circuits():
        # Use Times New Roman font, maintain Qiskit default line width and coloring (not bold)
        plt.rcParams['font.family'] = 'Times New Roman'
        qc_g0 = build_ghz_like_circuit('G0')
        qc_g1 = build_ghz_like_circuit('G1')
        # Display separately; titles clearly indicate Bell measurement on pairs (1,2) and (3,4)
        fig0 = qc_g0.draw(output='mpl')
        fig0.axes[0].set_title('Quantum circuit for |G0> preparation and Bell measuring(0,1) & (2,3)', fontname='Times New Roman', fontsize=16, pad=5) # pad=5 represents distance between title and figure
        plt.tight_layout()
        plt.show()
        fig1 = qc_g1.draw(output='mpl')
        fig1.axes[0].set_title('Quantum circuit for |G1> preparation and Bell measuring(0,1) & (2,3)', fontname='Times New Roman', fontsize=16, pad=5)   
        plt.tight_layout()
        plt.show()

    draw_ghz_like_circuits()

    # ------- Plot 1024-shot probability distribution of Bell measurements (0,1) and (2,3) for |G0> and |G1> pairs -------
    def plot_bell_measurement_histograms():
        plt.rcParams['font.family'] = 'Times New Roman'
        simulator_local = AerSimulator()
        SHOTS_BELL = 1024

        def run_counts(g_type: str):
            qc = QuantumCircuit(4, 4)
            # Construct according to create_GHZ_like
            qc.h(0)
            qc.cx(0,1)
            qc.cx(0,2)
            qc.cx(0,3)
            if g_type == 'G1':
                qc.x(1)
                qc.x(3)
            qc.h(0)
            qc.h(1)
            qc.h(2)
            qc.h(3)
            # Bell measurement (0,1) and (2,3)
            qc.cx(0,1)
            qc.h(0)
            qc.measure(0,0)
            qc.measure(1,1)
            qc.cx(2,3)
            qc.h(2)
            qc.measure(2,2)
            qc.measure(3,3)
            res = simulator_local.run(qc, shots=SHOTS_BELL).result().get_counts()
            return res

        def pair_counts(full_counts, pair):
            # pair: '01' for qubits (0,1) or '23' for (2,3)
            agg = {'00':0,'01':0,'10':0,'11':0}
            for bitstr, cnt in full_counts.items():
                # bitstr order: c3 c2 c1 c0
                b = bitstr.replace(' ','')
                if pair == '01':
                    key = b[0] + b[1]
                else:
                    key = b[2] + b[3]
                agg[key] += cnt
            return agg

        counts_g0 = run_counts('G0')
        counts_g1 = run_counts('G1')
        agg_g0_01 = pair_counts(counts_g0,'01')
        agg_g0_23 = pair_counts(counts_g0,'23')
        agg_g1_01 = pair_counts(counts_g1,'01')
        agg_g1_23 = pair_counts(counts_g1,'23')

        def plot_one(agg, title):
            # Uniformly plot all four states: 00, 01, 10, 11
            labels = ['00', '01', '10', '11']
            values = [agg[k]/SHOTS_BELL*100 for k in labels]
            fig, ax = plt.subplots(figsize=(5,3))
            bar_width = 0.20 
            x = list(range(len(labels)))
            ax.bar(x, values, color='#4ea3ff', width=bar_width, edgecolor='#33B1FF', linewidth=0.6, align='center')
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='--', alpha=0.25)
            ax.set_ylim(0, 60) 
            ax.set_xlim(-0.5, len(labels)-0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Probability (% of 1024 shots)')
            ax.set_xlabel('Measurement outcome')
            ax.set_title(title, fontsize=15, pad=8)
            for i, v in enumerate(values):
                ax.text(x[i], v+1, f"{v:.1f}", ha='center', fontsize=10)
            plt.tight_layout()
            plt.show()

        plot_one(agg_g0_01, '(a) Bell outcomes of |G0> on pair (0,1)')
        plot_one(agg_g0_23, '(b) Bell outcomes of |G0> on pair (2,3)')
        plot_one(agg_g1_01, '(c) Bell outcomes of |G1> on pair (0,1)')
        plot_one(agg_g1_23, '(d) Bell outcomes of |G1> on pair (2,3)')

    plot_bell_measurement_histograms()

if __name__ == "__main__":
    main() 