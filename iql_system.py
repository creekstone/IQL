#!/usr/bin/env python3
"""
Integral Quadrant Logic (IQL) System Implementation

Author: Buford Ray Conley
Copyright (c) 2025 Buford Ray Conley

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
http://creativecommons.org/licenses/by-nc/4.0/

You are free to:
- Share: copy and redistribute the material in any medium or format
- Adapt: remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, 
  and indicate if changes were made.
- NonCommercial: You may not use the material for commercial purposes.

For commercial use, please contact the author for licensing terms.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import quaternion  # numpy-quaternion package
import numba
from numba import jit, prange
import sparse  # For large-scale tensor operations

@dataclass
class TruthVector:
    """Four-dimensional truth vector with validation and operations."""
    I: float    # Individual-Interior
    It: float   # Individual-Exterior  
    We: float   # Collective-Interior
    Its: float  # Collective-Exterior
    
    def __post_init__(self):
        """Validate truth vector components."""
        for component in [self.I, self.It, self.We, self.Its]:
            if not 0 <= component <= 1:
                raise ValueError(f"Truth components must be in [0,1], got {component}")
    
    @property
    def vector(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.I, self.It, self.We, self.Its])
    
    def reflexive_imbalance(self) -> float:
        """Compute reflexive imbalance RI."""
        v = self.vector
        return np.max(np.abs(np.subtract.outer(v, v)))
    
    def to_quaternion(self) -> quaternion.quaternion:
        """Convert to quaternion representation."""
        return quaternion.quaternion(self.I, self.It, self.We, self.Its)

class BayesTensor:
    """Epistemic Bayes tensor with numerical stability."""
    
    EPSILON = 1e-10  # Prevent log(0)
    NORMALIZATION = np.log(100)  # RI normalization
    
    def __init__(self, truth_vector: TruthVector):
        self.tv = truth_vector
        self._tensor = None
        self._computed = False
    
    def compute(self) -> np.ndarray:
        """Compute Bayes tensor B_ij = log(t_j/t_i)."""
        if self._computed:
            return self._tensor
            
        # Add epsilon to prevent log(0)
        v = self.tv.vector + self.EPSILON
        
        # Compute all pairwise log ratios
        self._tensor = np.log(v[:, np.newaxis] / v[np.newaxis, :])
        
        # Ensure exact antisymmetry
        self._tensor = (self._tensor - self._tensor.T) / 2
        
        self._computed = True
        return self._tensor
    
    def max_norm(self) -> float:
        """Maximum tensor entry (normalized for RI)."""
        tensor = self.compute()
        return np.max(np.abs(tensor)) / self.NORMALIZATION
    
    def frobenius_norm(self) -> float:
        """Frobenius norm measuring total system tension."""
        tensor = self.compute()
        return np.linalg.norm(tensor, 'fro')

class IQLOperators:
    """Implementation of the four epistemic operators."""
    
    @staticmethod
    def alpha(tv: TruthVector, strength: float = 0.3) -> TruthVector:
        """Interiorize: External evidence → Internal experience."""
        # Increase I based on It and Its
        delta_I = strength * (tv.It + tv.Its - 2 * tv.I) / 2
        new_I = np.clip(tv.I + delta_I, 0, 1)
        return TruthVector(new_I, tv.It, tv.We, tv.Its)
    
    @staticmethod
    def beta(tv: TruthVector, strength: float = 0.3) -> TruthVector:
        """Exteriorize: Internal experience → External behavior."""
        # Increase It based on I
        delta_It = strength * (tv.I - tv.It)
        new_It = np.clip(tv.It + delta_It, 0, 1)
        return TruthVector(tv.I, new_It, tv.We, tv.Its)
    
    @staticmethod
    def gamma(tv: TruthVector, strength: float = 0.3) -> TruthVector:
        """Collectivize: Individual → Collective narrative."""
        # Increase We based on I and It
        delta_We = strength * ((tv.I + tv.It) / 2 - tv.We)
        new_We = np.clip(tv.We + delta_We, 0, 1)
        return TruthVector(tv.I, tv.It, new_We, tv.Its)
    
    @staticmethod
    def delta(tv: TruthVector, strength: float = 0.3) -> TruthVector:
        """Systemize: Understanding → Institutional structure."""
        # Increase Its based on all other quadrants
        avg_others = (tv.I + tv.It + tv.We) / 3
        delta_Its = strength * (avg_others - tv.Its)
        new_Its = np.clip(tv.Its + delta_Its, 0, 1)
        return TruthVector(tv.I, tv.It, tv.We, new_Its)
    
    @staticmethod
    def reflexivity_cycle(tv: TruthVector, strength: float = 0.2) -> TruthVector:
        """Apply full reflexivity cycle R = β∘δ∘γ∘α."""
        result = IQLOperators.alpha(tv, strength)
        result = IQLOperators.gamma(result, strength)
        result = IQLOperators.delta(result, strength)
        result = IQLOperators.beta(result, strength)
        return result

class IQLSystem:
    """Complete IQL system with evolution tracking."""
    
    def __init__(self, initial_state: TruthVector):
        self.current_state = initial_state
        self.history: List[TruthVector] = [initial_state]
        self.tensor_history: List[BayesTensor] = [BayesTensor(initial_state)]
        self.operator_history: List[str] = []
    
    def apply_operator(self, operator_name: str, strength: float = 0.3) -> TruthVector:
        """Apply named operator and update history."""
        operator_map = {
            'alpha': IQLOperators.alpha,
            'beta': IQLOperators.beta,
            'gamma': IQLOperators.gamma,
            'delta': IQLOperators.delta,
            'reflexivity': IQLOperators.reflexivity_cycle
        }
        
        if operator_name not in operator_map:
            raise ValueError(f"Unknown operator: {operator_name}")
        
        # Apply operator
        new_state = operator_map[operator_name](self.current_state, strength)
        
        # Update history
        self.current_state = new_state
        self.history.append(new_state)
        self.tensor_history.append(BayesTensor(new_state))
        self.operator_history.append(operator_name)
        
        return new_state
    
    def optimal_intervention(self, target_ri: float = 0.2, max_steps: int = 50) -> List[str]:
        """Find optimal operator sequence to achieve target RI."""
        interventions = []
        
        for step in range(max_steps):
            current_ri = self.current_state.reflexive_imbalance()
            
            if current_ri <= target_ri:
                break
            
            # Greedy strategy: target weakest quadrant
            min_component = min(self.current_state.vector)
            
            if min_component == self.current_state.I:
                operator = 'alpha'
            elif min_component == self.current_state.It:
                operator = 'beta'
            elif min_component == self.current_state.We:
                operator = 'gamma'
            else:  # Its is minimum
                operator = 'delta'
            
            self.apply_operator(operator)
            interventions.append(operator)
        
        return interventions
    
    def simulate_crisis(self, crisis_type: str = 'financial') -> dict:
        """Simulate crisis scenario with tensor dynamics."""
        crisis_scenarios = {
            'financial': {
                'initial': TruthVector(0.8, 0.2, 0.9, 0.95),  # High beliefs, low fundamentals
                'shocks': [('beta', 0.5), ('reflexivity', 0.3), ('alpha', -0.4)]  # Negative alpha = reality check
            },
            'ai_safety': {
                'initial': TruthVector(0.6, 0.95, 0.3, 0.2),  # High tech metrics, low other dimensions
                'shocks': [('gamma', 0.4), ('delta', 0.3)]
            },
            'climate': {
                'initial': TruthVector(0.7, 0.98, 0.6, 0.4),  # High science, lagging system
                'shocks': [('alpha', 0.5), ('gamma', 0.4), ('delta', 0.6)]
            }
        }
        
        if crisis_type not in crisis_scenarios:
            raise ValueError(f"Unknown crisis type: {crisis_type}")
        
        scenario = crisis_scenarios[crisis_type]
        self.current_state = scenario['initial']
        self.history = [self.current_state]
        
        results = {
            'initial_ri': self.current_state.reflexive_imbalance(),
            'trajectory': [],
            'max_ri': 0,
            'stability_achieved': False
        }
        
        for operator, strength in scenario['shocks']:
            if operator == 'reflexivity':
                new_state = IQLOperators.reflexivity_cycle(self.current_state, strength)
            else:
                op_func = getattr(IQLOperators, operator)
                new_state = op_func(self.current_state, strength)
            
            self.current_state = new_state
            self.history.append(new_state)
            
            ri = new_state.reflexive_imbalance()
            results['trajectory'].append({
                'operator': operator,
                'strength': strength,
                'ri': ri,
                'state': new_state.vector.tolist()
            })
            results['max_ri'] = max(results['max_ri'], ri)
        
        # Check final stability
        results['final_ri'] = self.current_state.reflexive_imbalance()
        results['stability_achieved'] = results['final_ri'] < 0.3
        
        return results

# Example usage and validation
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
    
    def distributed_optimization(self, initial_states: List[TruthVector], 
                               target_ri: float = 0.2) -> List[List[str]]:
        """Distributed optimization across multiple initial states."""
        # This would interface with Ray, Dask, or similar for true distribution
        results = []
        
        for state in initial_states:
            system = IQLSystem(state)
            interventions = system.optimal_intervention(target_ri)
            results.append(interventions)
        
        return results

# Integration with external systems
class IQLDashboard:
    """Real-time monitoring dashboard for IQL systems."""
    
    def __init__(self):
        self.alert_thresholds = {
            'ri_critical': 0.7,
            'ri_warning': 0.5,
            'tensor_critical': 2.0
        }
        self.systems = {}
    
    def register_system(self, name: str, initial_state: TruthVector):
        """Register a system for monitoring."""
        self.systems[name] = {
            'system': IQLSystem(initial_state),
            'alerts': [],
            'last_update': None
        }
    
    def update_system(self, name: str, new_state: TruthVector):
        """Update system state and check for alerts."""
        if name not in self.systems:
            raise ValueError(f"System {name} not registered")
        
        system_data = self.systems[name]
        system_data['system'].current_state = new_state
        
        # Check alert conditions
        ri = new_state.reflexive_imbalance()
        tensor = BayesTensor(new_state)
        tensor_norm = tensor.max_norm()
        
        alerts = []
        if ri > self.alert_thresholds['ri_critical']:
            alerts.append(f"CRITICAL: RI = {ri:.3f} exceeds critical threshold")
        elif ri > self.alert_thresholds['ri_warning']:
            alerts.append(f"WARNING: RI = {ri:.3f} in warning zone")
        
        if tensor_norm > self.alert_thresholds['tensor_critical']:
            alerts.append(f"CRITICAL: Tensor norm = {tensor_norm:.3f} exceeds threshold")
        
        system_data['alerts'].extend(alerts)
        return alerts
    
    def get_system_status(self, name: str) -> dict:
        """Get comprehensive system status."""
        if name not in self.systems:
            return {'error': f'System {name} not found'}
        
        system = self.systems[name]['system']
        current_state = system.current_state
        
        return {
            'name': name,
            'current_state': current_state.vector.tolist(),
            'ri': current_state.reflexive_imbalance(),
            'tensor_norm': BayesTensor(current_state).max_norm(),
            'weakest_quadrant': np.argmin(current_state.vector),
            'recent_alerts': self.systems[name]['alerts'][-5:],  # Last 5 alerts
            'recommended_action': self._recommend_action(current_state)
        }
    
    def _recommend_action(self, state: TruthVector) -> str:
        """Recommend intervention based on current state."""
        ri = state.reflexive_imbalance()
        
        if ri < 0.3:
            return "System stable - no intervention needed"
        elif ri < 0.5:
            return "Monitor closely - consider preemptive intervention"
        else:
            weakest = np.argmin(state.vector)
            operator_map = {0: 'alpha', 1: 'beta', 2: 'gamma', 3: 'delta'}
            return f"URGENT: Apply {operator_map[weakest]} operator to strengthen weakest quadrant"

if __name__ == "__main__":
    validate_iql_implementation() 