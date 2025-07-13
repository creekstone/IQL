#!/usr/bin/env python3
"""
Comprehensive Examples for the IQL System
Demonstrates all capabilities including core library, operators, crisis simulation, and monitoring.

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
import time
from iql_system import (
    TruthVector, BayesTensor, IQLOperators, IQLSystem, 
    ScalableIQLSystem, IQLDashboard, fast_ri_computation, fast_tensor_norm
)

def example_core_library():
    """Example 1: Core IQL Library Implementation"""
    print("=== Core IQL Library Examples ===\n")
    
    # Create truth vectors
    print("1. Creating Truth Vectors:")
    tv1 = TruthVector(0.8, 0.2, 0.9, 0.95)  # High beliefs, low fundamentals
    tv2 = TruthVector(0.6, 0.95, 0.3, 0.2)  # High tech metrics, low other dimensions
    tv3 = TruthVector(0.7, 0.98, 0.6, 0.4)  # High science, lagging system
    
    print(f"Truth Vector 1: {tv1.vector}")
    print(f"Truth Vector 2: {tv2.vector}")
    print(f"Truth Vector 3: {tv3.vector}")
    
    # Compute reflexive imbalance
    print(f"\n2. Reflexive Imbalance (RI):")
    print(f"RI for TV1: {tv1.reflexive_imbalance():.3f}")
    print(f"RI for TV2: {tv2.reflexive_imbalance():.3f}")
    print(f"RI for TV3: {tv3.reflexive_imbalance():.3f}")
    
    # Convert to quaternion
    print(f"\n3. Quaternion Representation:")
    q1 = tv1.to_quaternion()
    print(f"Quaternion for TV1: {q1}")
    
    print("\n" + "="*50 + "\n")

def example_bayes_tensor():
    """Example 2: Bayes Tensor Operations"""
    print("=== Bayes Tensor Examples ===\n")
    
    # Create truth vector and tensor
    tv = TruthVector(0.8, 0.2, 0.9, 0.95)
    tensor = BayesTensor(tv)
    
    print("1. Computing Bayes Tensor:")
    B = tensor.compute()
    print(f"Bayes Tensor B_ij = log(t_j/t_i):")
    print(B)
    
    print(f"\n2. Tensor Norms:")
    print(f"Max norm (normalized): {tensor.max_norm():.3f}")
    print(f"Frobenius norm: {tensor.frobenius_norm():.3f}")
    
    print(f"\n3. Numerical Stability:")
    # Test with near-zero values
    tv_near_zero = TruthVector(0.001, 0.999, 0.001, 0.999)
    tensor_stable = BayesTensor(tv_near_zero)
    print(f"Tensor with near-zero values computed successfully: {tensor_stable.max_norm():.3f}")
    
    print("\n" + "="*50 + "\n")

def example_operators():
    """Example 3: IQL Operators"""
    print("=== IQL Operators Examples ===\n")
    
    # Initial state
    initial = TruthVector(0.5, 0.3, 0.7, 0.4)
    print(f"Initial state: {initial.vector}")
    print(f"Initial RI: {initial.reflexive_imbalance():.3f}")
    
    # Apply individual operators
    print(f"\n1. Individual Operators:")
    
    alpha_result = IQLOperators.alpha(initial, strength=0.3)
    print(f"α (Interiorize): {alpha_result.vector} | RI: {alpha_result.reflexive_imbalance():.3f}")
    
    beta_result = IQLOperators.beta(initial, strength=0.3)
    print(f"β (Exteriorize): {beta_result.vector} | RI: {beta_result.reflexive_imbalance():.3f}")
    
    gamma_result = IQLOperators.gamma(initial, strength=0.3)
    print(f"γ (Collectivize): {gamma_result.vector} | RI: {gamma_result.reflexive_imbalance():.3f}")
    
    delta_result = IQLOperators.delta(initial, strength=0.3)
    print(f"δ (Systemize): {delta_result.vector} | RI: {delta_result.reflexive_imbalance():.3f}")
    
    # Reflexivity cycle
    print(f"\n2. Reflexivity Cycle R = β∘δ∘γ∘α:")
    reflexivity_result = IQLOperators.reflexivity_cycle(initial, strength=0.2)
    print(f"Reflexivity result: {reflexivity_result.vector} | RI: {reflexivity_result.reflexive_imbalance():.3f}")
    
    # Non-commutativity demonstration
    print(f"\n3. Non-Commutativity:")
    path_a = IQLOperators.beta(IQLOperators.alpha(initial))
    path_b = IQLOperators.alpha(IQLOperators.beta(initial))
    difference = np.linalg.norm(path_a.vector - path_b.vector)
    print(f"Path A (α→β): {path_a.vector}")
    print(f"Path B (β→α): {path_b.vector}")
    print(f"Difference: {difference:.6f}")
    
    print("\n" + "="*50 + "\n")

def example_system_evolution():
    """Example 4: IQL System Evolution"""
    print("=== IQL System Evolution Examples ===\n")
    
    # Create system with high imbalance
    initial_state = TruthVector(0.9, 0.2, 0.8, 0.1)
    system = IQLSystem(initial_state)
    
    print(f"1. Initial System State:")
    print(f"   State: {system.current_state.vector}")
    print(f"   RI: {system.current_state.reflexive_imbalance():.3f}")
    
    # Apply operators and track evolution
    print(f"\n2. System Evolution:")
    operators_to_apply = ['alpha', 'beta', 'gamma', 'delta', 'reflexivity']
    
    for i, op in enumerate(operators_to_apply):
        new_state = system.apply_operator(op, strength=0.3)
        print(f"   Step {i+1} ({op}): {new_state.vector} | RI: {new_state.reflexive_imbalance():.3f}")
    
    print(f"\n3. System History:")
    print(f"   Total steps: {len(system.history)}")
    print(f"   Operator sequence: {system.operator_history}")
    
    print("\n" + "="*50 + "\n")

def example_optimal_intervention():
    """Example 5: Optimal Intervention"""
    print("=== Optimal Intervention Examples ===\n")
    
    # Create system with high imbalance
    initial_state = TruthVector(0.9, 0.2, 0.8, 0.1)
    system = IQLSystem(initial_state)
    
    print(f"1. Initial State:")
    print(f"   State: {system.current_state.vector}")
    print(f"   RI: {system.current_state.reflexive_imbalance():.3f}")
    
    # Find optimal intervention
    print(f"\n2. Finding Optimal Intervention (target RI = 0.2):")
    interventions = system.optimal_intervention(target_ri=0.2, max_steps=20)
    
    print(f"   Final RI: {system.current_state.reflexive_imbalance():.3f}")
    print(f"   Interventions applied: {len(interventions)}")
    print(f"   Sequence: {interventions}")
    
    print("\n" + "="*50 + "\n")

def example_crisis_simulation():
    """Example 6: Crisis Simulation"""
    print("=== Crisis Simulation Examples ===\n")
    
    # Test different crisis types
    crisis_types = ['financial', 'ai_safety', 'climate']
    
    for crisis_type in crisis_types:
        print(f"1. {crisis_type.upper()} Crisis Simulation:")
        
        system = IQLSystem(TruthVector(0.5, 0.5, 0.5, 0.5))
        results = system.simulate_crisis(crisis_type)
        
        print(f"   Initial RI: {results['initial_ri']:.3f}")
        print(f"   Maximum RI: {results['max_ri']:.3f}")
        print(f"   Final RI: {results['final_ri']:.3f}")
        print(f"   Stability achieved: {results['stability_achieved']}")
        
        print(f"   Trajectory:")
        for step in results['trajectory']:
            print(f"     {step['operator']} (strength={step['strength']}): RI={step['ri']:.3f}")
        
        print()
    
    print("="*50 + "\n")

def example_dashboard_monitoring():
    """Example 7: Dashboard Monitoring"""
    print("=== Dashboard Monitoring Examples ===\n")
    
    # Create dashboard
    dashboard = IQLDashboard()
    
    # Register multiple systems
    systems = {
        "financial_market": TruthVector(0.8, 0.2, 0.9, 0.95),
        "ai_safety": TruthVector(0.6, 0.95, 0.3, 0.2),
        "climate_system": TruthVector(0.7, 0.98, 0.6, 0.4),
        "stable_system": TruthVector(0.5, 0.48, 0.52, 0.49)
    }
    
    print("1. Registering Systems:")
    for name, state in systems.items():
        dashboard.register_system(name, state)
        print(f"   Registered: {name}")
    
    # Get initial status
    print(f"\n2. Initial System Status:")
    for name in systems.keys():
        status = dashboard.get_system_status(name)
        print(f"   {name.upper()}:")
        print(f"     RI: {status['ri']:.3f}")
        print(f"     Tensor norm: {status['tensor_norm']:.3f}")
        print(f"     Weakest quadrant: {status['weakest_quadrant']}")
        print(f"     Recommendation: {status['recommended_action']}")
    
    # Simulate updates and check alerts
    print(f"\n3. System Updates and Alerts:")
    
    # Crisis scenario
    crisis_state = TruthVector(0.9, 0.1, 0.95, 0.98)
    alerts = dashboard.update_system("financial_market", crisis_state)
    
    if alerts:
        print(f"   🚨 ALERTS for financial_market:")
        for alert in alerts:
            print(f"     {alert}")
    
    # Improvement scenario
    improved_state = TruthVector(0.7, 0.8, 0.6, 0.5)
    alerts = dashboard.update_system("ai_safety", improved_state)
    
    if alerts:
        print(f"   🚨 ALERTS for ai_safety:")
        for alert in alerts:
            print(f"     {alert}")
    
    print("\n" + "="*50 + "\n")

def example_scalable_system():
    """Example 8: Scalable System"""
    print("=== Scalable System Examples ===\n")
    
    # Generate large batch of truth vectors
    np.random.seed(42)
    n_vectors = 1000
    truth_vectors = np.random.random((n_vectors, 4))
    
    print(f"1. Batch Processing {n_vectors} vectors:")
    
    scalable_system = ScalableIQLSystem()
    start_time = time.time()
    results = scalable_system.batch_process_vectors(truth_vectors)
    processing_time = time.time() - start_time
    
    print(f"   Processing time: {processing_time:.4f} seconds")
    print(f"   Average time per vector: {processing_time/n_vectors*1000:.2f} ms")
    
    # Display results
    ri_stats = results['ri_distribution']
    print(f"\n2. Results:")
    print(f"   Mean RI: {ri_stats['mean']:.3f}")
    print(f"   Std RI: {ri_stats['std']:.3f}")
    print(f"   Max RI: {ri_stats['max']:.3f}")
    print(f"   High-risk vectors: {ri_stats['high_risk_count']} ({ri_stats['high_risk_count']/n_vectors*100:.1f}%)")
    
    # Distributed optimization
    print(f"\n3. Distributed Optimization:")
    initial_states = [
        TruthVector(0.9, 0.2, 0.8, 0.1),
        TruthVector(0.6, 0.95, 0.3, 0.2),
        TruthVector(0.7, 0.98, 0.6, 0.4)
    ]
    
    interventions = scalable_system.distributed_optimization(initial_states, target_ri=0.2)
    
    for i, intervention_list in enumerate(interventions):
        print(f"   System {i+1}: {len(intervention_list)} interventions")
    
    print("\n" + "="*50 + "\n")

def example_performance_comparison():
    """Example 9: Performance Comparison"""
    print("=== Performance Comparison Examples ===\n")
    
    # Generate test data
    np.random.seed(42)
    test_vectors = np.random.random((1000, 4))
    
    print("1. RI Computation Performance:")
    
    # Standard implementation
    start_time = time.time()
    standard_ri = []
    for vector in test_vectors:
        tv = TruthVector(*vector)
        standard_ri.append(tv.reflexive_imbalance())
    standard_time = time.time() - start_time
    
    # Optimized implementation
    start_time = time.time()
    optimized_ri = fast_ri_computation(test_vectors)
    optimized_time = time.time() - start_time
    
    print(f"   Standard: {standard_time:.4f} seconds")
    print(f"   Optimized: {optimized_time:.4f} seconds")
    print(f"   Speedup: {standard_time/optimized_time:.1f}x")
    
    # Verify accuracy
    max_diff = np.max(np.abs(np.array(standard_ri) - optimized_ri))
    print(f"   Accuracy: {'✅ Match' if max_diff < 1e-10 else '❌ Mismatch'}")
    
    print(f"\n2. Tensor Norm Performance:")
    test_vector = test_vectors[0]
    
    # Standard tensor norm
    start_time = time.time()
    tv = TruthVector(*test_vector)
    tensor = BayesTensor(tv)
    standard_norm = tensor.max_norm()
    standard_norm_time = time.time() - start_time
    
    # Optimized tensor norm
    start_time = time.time()
    optimized_norm = fast_tensor_norm(test_vector)
    optimized_norm_time = time.time() - start_time
    
    print(f"   Standard: {standard_norm_time*1000:.2f} ms")
    print(f"   Optimized: {optimized_norm_time*1000:.2f} ms")
    print(f"   Speedup: {standard_norm_time/optimized_norm_time:.1f}x")
    print(f"   Accuracy: {'✅ Match' if abs(standard_norm - optimized_norm) < 1e-10 else '❌ Mismatch'}")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE IQL SYSTEM EXAMPLES")
    print("=" * 60)
    
    try:
        example_core_library()
        example_bayes_tensor()
        example_operators()
        example_system_evolution()
        example_optimal_intervention()
        example_crisis_simulation()
        example_dashboard_monitoring()
        example_scalable_system()
        example_performance_comparison()
        
        print("✅ All examples completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  • Core IQL library with TruthVector and BayesTensor")
        print("  • Four epistemic operators (α, β, γ, δ) and reflexivity cycle")
        print("  • System evolution and optimal intervention strategies")
        print("  • Crisis simulation for financial, AI safety, and climate scenarios")
        print("  • Real-time dashboard monitoring with alerts")
        print("  • Scalable batch processing and distributed optimization")
        print("  • Performance optimizations with Numba JIT compilation")
        
    except Exception as e:
        print(f"❌ Error during examples: {e}")
        import traceback
        traceback.print_exc() 