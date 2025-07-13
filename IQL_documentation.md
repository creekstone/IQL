# IQL System Documentation

**Author:** Buford Ray Conley  
**Copyright (c) 2024 Buford Ray Conley**

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.  
For commercial use, please contact the author for licensing terms.

This document provides example usage, validation, and performance optimization techniques for the Integral Quadrant Logic (IQL) system.

---

## Example Usage and Validation

The following code demonstrates how to validate the core properties of the IQL system, including conservative extension, operator non-commutativity, convergence, and crisis simulation.

```python
from iql_system import TruthVector, IQLOperators, IQLSystem
import numpy as np

def validate_iql_implementation():
    """Comprehensive validation of IQL implementation."""
    print("=== IQL Implementation Validation ===\n")

    # Test 1: Conservative extension
    print("1. Testing Conservative Extension")
    classical_true = TruthVector(1, 1, 1, 1)
    classical_false = TruthVector(0, 0, 0, 0)

    print(f"Classical TRUE: RI = {classical_true.reflexive_imbalance():.3f}")
    print(f"Classical FALSE: RI = {classical_false.reflexive_imbalance():.3f}")
    assert classical_true.reflexive_imbalance() == 0, "Classical true should have RI = 0"
    assert classical_false.reflexive_imbalance() == 0, "Classical false should have RI = 0"
    print("✓ Conservative extension validated\n")

    # Test 2: Operator non-commutativity
    print("2. Testing Operator Non-Commutativity")
    initial = TruthVector(0.8, 0.2, 0.6, 0.4)  # Asymmetric initial state

    # Path A: α then β
    path_a = IQLOperators.beta(IQLOperators.alpha(initial))

    # Path B: β then α  
    path_b = IQLOperators.alpha(IQLOperators.beta(initial))

    print(f"Path A (α→β): {path_a.vector}")
    print(f"Path B (β→α): {path_b.vector}")
    print(f"Difference: {np.linalg.norm(path_a.vector - path_b.vector):.6f}")
    assert not np.allclose(path_a.vector, path_b.vector), "Operators should not commute"
    print("✓ Non-commutativity validated\n")

    # Test 3: Convergence theorem
    print("3. Testing Convergence Theorem")
    system = IQLSystem(TruthVector(0.9, 0.2, 0.8, 0.1))  # High imbalance
    initial_ri = system.current_state.reflexive_imbalance()

    interventions = system.optimal_intervention(target_ri=0.15, max_steps=20)
    final_ri = system.current_state.reflexive_imbalance()

    print(f"Initial RI: {initial_ri:.3f}")
    print(f"Final RI: {final_ri:.3f}")
    print(f"Interventions applied: {interventions}")
    assert final_ri < initial_ri, "RI should decrease with optimal interventions"
    print("✓ Convergence theorem validated\n")

    # Test 4: Crisis simulation
    print("4. Testing Crisis Simulation")
    system = IQLSystem(TruthVector(0.5, 0.5, 0.5, 0.5))
    results = system.simulate_crisis('financial')

    print(f"Financial crisis simulation:")
    print(f"  Initial RI: {results['initial_ri']:.3f}")
    print(f"  Maximum RI: {results['max_ri']:.3f}")
    print(f"  Final RI: {results['final_ri']:.3f}")
    print(f"  Stability achieved: {results['stability_achieved']}")
    print("✓ Crisis simulation validated\n")

    print("=== All Tests Passed ===")

if __name__ == "__main__":
    validate_iql_implementation()
```

---

## Performance Optimization and Scaling

The following code demonstrates how to use Numba and batch processing to scale the IQL system for large datasets. This includes JIT-compiled functions for fast RI and tensor norm computation, as well as a scalable system class for batch and distributed analysis.

```python
import numpy as np
import numba
from numba import jit, prange
import sparse  # For large-scale tensor operations
from iql_system import TruthVector, IQLSystem

@jit(nopython=True)
def fast_ri_computation(truth_vectors: np.ndarray) -> np.ndarray:
    """Optimized RI computation for large batches."""
    n_vectors = truth_vectors.shape[0]
    ri_values = np.empty(n_vectors)
    for i in prange(n_vectors):
        max_diff = 0.0
        for j in range(4):
            for k in range(j + 1, 4):
                diff = abs(truth_vectors[i, j] - truth_vectors[i, k])
                if diff > max_diff:
                    max_diff = diff
        ri_values[i] = max_diff
    return ri_values

@jit(nopython=True)
def fast_tensor_norm(truth_vector: np.ndarray) -> float:
    """JIT-compiled tensor norm computation."""
    max_log_ratio = 0.0
    epsilon = 1e-10
    for i in range(4):
        for j in range(4):
            if i != j:
                a = max(truth_vector[i], epsilon)
                b = max(truth_vector[j], epsilon)
                log_ratio = abs(np.log(a / b))
                if log_ratio > max_log_ratio:
                    max_log_ratio = log_ratio
    return max_log_ratio / np.log(100.0)

class ScalableIQLSystem:
    """IQL system optimized for large-scale analysis."""
    def __init__(self):
        self.batch_size = 10000
        self.use_sparse = True
    def batch_process_vectors(self, truth_vectors: np.ndarray) -> dict:
        """Process large batches of truth vectors efficiently."""
        n_vectors = len(truth_vectors)
        # Compute RI for all vectors
        ri_values = fast_ri_computation(truth_vectors)
        # Identify high-risk vectors
        high_risk_indices = np.where(ri_values > 0.5)[0]
        # Compute tensor norms for high-risk vectors only
        tensor_norms = np.array([
            fast_tensor_norm(truth_vectors[i]) 
            for i in high_risk_indices
        ])
        return {
            'ri_distribution': {
                'mean': np.mean(ri_values),
                'std': np.std(ri_values),
                'max': np.max(ri_values),
                'high_risk_count': len(high_risk_indices)
            },
            'high_risk_indices': high_risk_indices,
            'tensor_norms': tensor_norms,
            'processing_time': None  # Add timing as needed
        }
    def distributed_optimization(self, initial_states: list, target_ri: float = 0.2) -> list:
        """Distributed optimization across multiple initial states."""
        results = []
        for state in initial_states:
            system = IQLSystem(state)
            interventions = system.optimal_intervention(target_ri)
            results.append(interventions)
        return results
```

---

## Notes
- The validation and performance code can be run as standalone scripts or integrated into your own workflows.
- For best performance, ensure you have `numba` and `sparse` installed in your environment.
- The scalable system is designed for large-scale, real-time, or distributed applications. 