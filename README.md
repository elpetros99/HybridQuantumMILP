# Hybrid Quantum-Classical MILP Solvers for Power Systems

This repository provides tutorials demonstrating how to solve Mixed-Integer Linear Programs (MILPs) using a hybrid quantum-classical approach with Benders decomposition. It integrates classical optimization methods with quantum annealing platforms (D-Wave and IBM Qiskit), applied specifically to power systems.

## Repository Structure

```
elpetros99-hybridquantummilp/
├── MILP_Gurobi_Discrete_Dwave_ready_to_upload.ipynb
├── MILP_OTS_Benders_6Bus_upload.ipynb
└── MILP_OTS_Benders_IBM_6Bus_upload.ipynb
```

## Notebooks Overview

- **MILP_Gurobi_Discrete_Dwave_ready_to_upload.ipynb**  
  Solves a MILP for Neural Network Verification within DC Optimal Power Flow. The notebook demonstrates data preparation (via MATLAB), problem formulation using Gurobi, Benders decomposition, and solving the optimization using a D-Wave quantum annealer.

- **MILP_OTS_Benders_6Bus_upload.ipynb**  
  Demonstrates Optimal Transmission Switching (OTS) optimization on a 6-bus system using Gurobi, Benders decomposition, and D-Wave quantum annealing.

- **MILP_OTS_Benders_IBM_6Bus_upload.ipynb**  
  Implements OTS optimization using IBM's Qiskit framework, illustrating how to convert a MILP into QUBO form, set up a QAOA ansatz, and execute on IBM’s quantum backends.

## Prerequisites

- **Python 3.x** and **Jupyter Notebook/JupyterLab**
- Required Python packages:
  ```bash
  pip install numpy pandas matplotlib scipy gurobipy dimod docplex qiskit pyomo
  ```
- **MATLAB** with MATLAB Engine API for Python (for data processing)
- Quantum computing account tokens:
  - [D-Wave Leap](https://cloud.dwavesys.com/leap/login/)
  - [IBM Quantum](https://quantum.ibm.com/) for Qiskit Runtime

## Getting Started

### Clone the repository:
```bash
git clone https://github.com/your-username/elpetros99-hybridquantummilp.git
cd HybridQuantumMILP
```

### Install Python dependencies:
```bash
pip install numpy pandas matplotlib scipy gurobipy dimod docplex qiskit pyomo
```

### Set up MATLAB Engine API:
Follow instructions from [MathWorks](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

### Configure Quantum Access:
- Replace placeholders with your actual tokens for D-Wave and IBM Quantum services in the notebooks.

## Running the Notebooks
- Launch Jupyter:
```bash
jupyter notebook
```
- Open and run notebooks sequentially.

Each notebook clearly illustrates:
- Formulating the MILP problem
- Applying Benders decomposition
- Converting MILP problems into QUBO
- Solving with quantum annealers (D-Wave or IBM Qiskit)

## Contributing
Contributions are welcome! Open issues or submit pull requests for enhancements or bug fixes.

## License
Licensed under the MIT License.
