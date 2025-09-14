#Quantum Knapsack Solver: A "From Scratch" VQE Implementation
This repository contains a Python script that solves the classic Knapsack problem using a Variational Quantum Eigensolver (VQE) algorithm implemented from scratch with only the core Qiskit library. The goal of this project is to provide a clear, educational, and self-contained example of how a real-world optimization problem can be translated into a quantum-solvable format without relying on high-level application modules.

#The chosen scenario is a strategic resource allocation problem for a non-profit organization aiming to maximize its social impact within a limited budget.

#The Problem: Strategic Resource Allocation
The Knapsack problem is a famous challenge in combinatorial optimization. In our scenario:

The Knapsack: A non-profit's limited budget.

The Items: A list of potential projects to fund.

Item Weight: The cost of each project.

Item Value: The social impact score of each project.

Objective: Select the combination of projects that yields the maximum possible impact without exceeding the budget. As the number of projects grows, the number of possible combinations becomes too large for classical computers to check exhaustively.

#The Quantum Approach: VQE from Scratch
This project uses the Variational Quantum Eigensolver (VQE), a hybrid quantum-classical algorithm well-suited for near-term quantum devices. Instead of using pre-built libraries, this implementation demonstrates the fundamental steps from the ground up.

#Key Features
Core Qiskit Only: Uses only qiskit and qiskit-aer. It does not use qiskit-algorithms or qiskit-optimization, making it a pure, foundational example.

Manual Hamiltonian Construction: The business problem (objective and constraints) is manually translated into a Quadratic Unconstrained Binary Optimization (QUBO) problem and then mapped to a quantum Ising Hamiltonian.

Custom VQE Loop: The entire VQE optimization loop is implemented from scratch, including a custom gradient descent optimizer that uses the parameter-shift rule to calculate gradients.

Self-Contained & Executable: The script is designed to be run in a single block in an interactive environment like a Jupyter Notebook or Google Colab, with no external files.

Text-Based Reporting: The final result is presented in a clear, formatted text report without any plotting library dependencies.

# --- Step 1: Install Core Qiskit Libraries ---
# This command installs only the necessary core components.
import sys
try:
    import qiskit
    import qiskit_aer
except ImportError:
    print("Installing core Qiskit libraries...")
    !{sys.executable} -m pip install qiskit qiskit-aer

# --- Step 2: Import Core Qiskit and Standard Libraries ---
import math
import random
from collections import Counter

# Core Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal

# --- Step 3: Define the Knapsack Problem ---
# We represent the problem using basic Python data structures.
projects = [
    {'name': 'Water Wells',  'cost': 5, 'impact': 8},
    {'name': 'School Build', 'cost': 7, 'impact': 10},
    {'name': 'Medical Camp', 'cost': 3, 'impact': 5},
    {'name': 'Job Training', 'cost': 4, 'impact': 7},
]
BUDGET_LIMIT = 12
PENALTY = 15 # A large constant to penalize invalid solutions

print("--- Problem Definition ---")
print(f"Budget Limit: ${BUDGET_LIMIT}k")
print("Available Projects:")
for p in projects:
    print(f"  - {p['name']}: Cost=${p['cost']}k, Impact Score={p['impact']}")
print("-" * 28 + "\n")


# --- Step 4: Manually Construct the QUBO and Ising Hamiltonian ---
def build_hamiltonian_from_scratch(projects, budget, penalty):
    """
    Constructs an Ising Hamiltonian for the Knapsack problem manually.
    """
    num_projects = len(projects)
    max_slack_value = budget
    num_slack_qubits = math.floor(math.log2(max_slack_value)) + 1
    num_qubits = num_projects + num_slack_qubits
    print(f"Building Hamiltonian with {num_projects} project qubits and {num_slack_qubits} slack qubits (Total: {num_qubits}).\n")

    hamiltonian_terms = {}

    # --- Part 1: Cost Function (Objective) ---
    for i in range(num_projects):
        impact = projects[i]['impact']
        pauli_str = 'I' * i + 'Z' + 'I' * (num_qubits - i - 1)
        hamiltonian_terms[pauli_str] = hamiltonian_terms.get(pauli_str, 0) + impact / 2
        identity_str = 'I' * num_qubits
        hamiltonian_terms[identity_str] = hamiltonian_terms.get(identity_str, 0) - impact / 2

    # --- Part 2: Penalty Function (Constraint) ---
    linear_expr = {}
    for i in range(num_projects):
        linear_expr[i] = projects[i]['cost']
    for i in range(num_slack_qubits):
        linear_expr[num_projects + i] = 2**i
    constant_offset = -budget

    final_linear_expr = {}
    final_constant = constant_offset
    for i, coeff in linear_expr.items():
        final_linear_expr[i] = final_linear_expr.get(i, 0) - coeff / 2
        final_constant += coeff / 2

    identity_str = 'I' * num_qubits
    hamiltonian_terms[identity_str] = hamiltonian_terms.get(identity_str, 0) + penalty * (final_constant**2)

    for i, coeff in final_linear_expr.items():
        pauli_str = 'I' * i + 'Z' + 'I' * (num_qubits - i - 1)
        hamiltonian_terms[pauli_str] = hamiltonian_terms.get(pauli_str, 0) + penalty * 2 * final_constant * coeff

    for i, coeff_i in final_linear_expr.items():
        for j, coeff_j in final_linear_expr.items():
            if i == j:
                hamiltonian_terms[identity_str] = hamiltonian_terms.get(identity_str, 0) + penalty * (coeff_i**2)
            else:
                p_str = list('I' * num_qubits)
                p_str[i] = 'Z'; p_str[j] = 'Z'
                pauli_str = "".join(p_str)
                hamiltonian_terms[pauli_str] = hamiltonian_terms.get(pauli_str, 0) + penalty * coeff_i * coeff_j
    
    return SparsePauliOp.from_list(list(hamiltonian_terms.items()))


# --- Step 5: Implement the VQE Algorithm from Scratch ---
def get_expectation_value(circuit_params, ansatz, hamiltonian, backend):
    bound_circuit = ansatz.assign_parameters(circuit_params)
    measurable_circuit = bound_circuit.copy()
    measurable_circuit.measure_all()
    t_circ = transpile(measurable_circuit, backend)
    result = backend.run(t_circ, shots=2048).result()
    counts = result.get_counts()
    
    total_energy = 0
    for pauli, coeff in hamiltonian.to_list():
        pauli_energy = 0
        for bitstring, count in counts.items():
            eigenvalue = 1
            for i, pauli_char in enumerate(reversed(pauli)):
                if pauli_char == 'Z' and bitstring[i] == '1':
                    eigenvalue *= -1
            pauli_energy += eigenvalue * count
        total_energy += coeff.real * (pauli_energy / sum(counts.values()))
    return total_energy

def vqe_from_scratch(hamiltonian, num_qubits):
    backend = AerSimulator()
    ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=2)
    num_params = ansatz.num_parameters
    params = [random.uniform(0, 2 * math.pi) for _ in range(num_params)]
    
    learning_rate = 0.1
    iterations = 80
    
    print("--- Starting VQE Loop ---")
    history = []
    for i in range(iterations):
        gradients = []
        for j in range(num_params):
            params_plus = params.copy(); params_plus[j] += math.pi / 2
            energy_plus = get_expectation_value(params_plus, ansatz, hamiltonian, backend)
            params_minus = params.copy(); params_minus[j] -= math.pi / 2
            energy_minus = get_expectation_value(params_minus, ansatz, hamiltonian, backend)
            gradients.append(0.5 * (energy_plus - energy_minus))
        
        for j in range(num_params):
            params[j] -= learning_rate * gradients[j]
            
        current_energy = get_expectation_value(params, ansatz, hamiltonian, backend)
        history.append(current_energy)
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}, Energy: {current_energy:.4f}")
            
    print("--- VQE Loop Finished ---\n")
    return params, ansatz, history

# --- Step 6: Execute the Full Workflow ---
knapsack_hamiltonian = build_hamiltonian_from_scratch(projects, BUDGET_LIMIT, PENALTY)
num_qubits = knapsack_hamiltonian.num_qubits
optimal_params, final_ansatz, energy_history = vqe_from_scratch(knapsack_hamiltonian, num_qubits)

print("--- Analyzing Final Result ---")
final_circuit = final_ansatz.assign_parameters(optimal_params)
final_circuit.measure_all()

backend = AerSimulator()
t_final_circuit = transpile(final_circuit, backend)
result = backend.run(t_final_circuit, shots=4096).result()
counts = result.get_counts()
sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

best_solution = None
max_impact = -1

print("Top 5 measured outcomes:")
for bitstring, count in sorted_counts[:5]:
    solution_bits = [int(b) for b in reversed(bitstring)][:len(projects)]
    total_cost = sum(p['cost'] for i, p in enumerate(projects) if solution_bits[i] == 1)
    total_impact = sum(p['impact'] for i, p in enumerate(projects) if solution_bits[i] == 1)
    selected_names = [p['name'] for i, p in enumerate(projects) if solution_bits[i] == 1)
    
    validity = "VALID" if total_cost <= BUDGET_LIMIT else "INVALID"
    
    print(f"  - Bitstring: {bitstring}, Projects: {selected_names}, "
          f"Cost: {total_cost}, Impact: {total_impact} -> {validity}")

    if validity == "VALID" and total_impact > max_impact:
        max_impact = total_impact
        best_solution = {
            'projects': selected_names, 'cost': total_cost, 'impact': total_impact,
            'probability': count / sum(counts.values())
        }

# --- Step 7: Display Final Report ---
print("\n" + "="*35)
print("= Final Optimal Portfolio Report  =")
print("="*35)
if best_solution:
    print(f"\nOptimal combination of projects found:")
    for project_name in best_solution['projects']:
        print(f"  - {project_name}")
    print(f"\nTotal Cost:   ${best_solution['cost']}k (Budget: ${BUDGET_LIMIT}k)")
    print(f"Total Impact: {best_solution['impact']}")
    print(f"Probability of this result: {best_solution['probability']:.2%}")
else:
    print("\nNo valid solution was found in the top measurements.")
print("\n" + "="*35)

