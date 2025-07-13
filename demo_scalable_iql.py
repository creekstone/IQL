#!/usr/bin/env python3
"""
Demonstration of the Scalable IQL System
Showcases batch processing, real-time monitoring, and performance optimizations.
"""

import numpy as np
import time
from iql_system import (
    TruthVector, IQLSystem, ScalableIQLSystem, IQLDashboard,
    fast_ri_computation, fast_tensor_norm, BayesTensor
)

def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("=== Batch Processing Demonstration ===\n")
    
    # Generate 1000 random truth vectors
    np.random.seed(42)  # For reproducible results
    n_vectors = 1000
    truth_vectors = np.random.random((n_vectors, 4))
    
    print(f"Processing {n_vectors} truth vectors...")
    
    # Time the batch processing
    start_time = time.time()
    scalable_system = ScalableIQLSystem()
    results = scalable_system.batch_process_vectors(truth_vectors)
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.4f} seconds")
    print(f"Average time per vector: {processing_time/n_vectors*1000:.2f} ms")
    
    # Display results
    ri_stats = results['ri_distribution']
    print(f"\nRI Distribution:")
    print(f"  Mean: {ri_stats['mean']:.3f}")
    print(f"  Std:  {ri_stats['std']:.3f}")
    print(f"  Max:  {ri_stats['max']:.3f}")
    print(f"  High-risk vectors: {ri_stats['high_risk_count']} ({ri_stats['high_risk_count']/n_vectors*100:.1f}%)")
    
    if len(results['high_risk_indices']) > 0:
        print(f"\nHigh-risk tensor norms (first 5):")
        for i, norm in enumerate(results['tensor_norms'][:5]):
            print(f"  Vector {results['high_risk_indices'][i]}: {norm:.3f}")
    
    print("\n" + "="*50 + "\n")

def demo_dashboard_monitoring():
    """Demonstrate real-time dashboard monitoring."""
    print("=== Real-time Dashboard Demonstration ===\n")
    
    # Create dashboard
    dashboard = IQLDashboard()
    
    # Register multiple systems
    systems = {
        "financial_market": TruthVector(0.8, 0.2, 0.9, 0.95),
        "ai_safety": TruthVector(0.6, 0.95, 0.3, 0.2),
        "climate_system": TruthVector(0.7, 0.98, 0.6, 0.4),
        "stable_system": TruthVector(0.5, 0.48, 0.52, 0.49)
    }
    
    for name, state in systems.items():
        dashboard.register_system(name, state)
        print(f"Registered system: {name}")
    
    print("\nInitial system statuses:")
    for name in systems.keys():
        status = dashboard.get_system_status(name)
        print(f"\n{name.upper()}:")
        print(f"  RI: {status['ri']:.3f}")
        print(f"  Tensor norm: {status['tensor_norm']:.3f}")
        print(f"  Weakest quadrant: {status['weakest_quadrant']}")
        print(f"  Recommendation: {status['recommended_action']}")
    
    # Simulate system updates
    print("\n" + "-"*30)
    print("Simulating system updates...")
    
    # Update financial market (crisis scenario)
    crisis_state = TruthVector(0.9, 0.1, 0.95, 0.98)  # High imbalance
    alerts = dashboard.update_system("financial_market", crisis_state)
    
    if alerts:
        print(f"\n🚨 ALERTS for financial_market:")
        for alert in alerts:
            print(f"  {alert}")
    
    # Update AI safety (improvement)
    improved_state = TruthVector(0.7, 0.8, 0.6, 0.5)  # Better balance
    alerts = dashboard.update_system("ai_safety", improved_state)
    
    if alerts:
        print(f"\n🚨 ALERTS for ai_safety:")
        for alert in alerts:
            print(f"  {alert}")
    
    print("\nUpdated system statuses:")
    for name in systems.keys():
        status = dashboard.get_system_status(name)
        print(f"\n{name.upper()}:")
        print(f"  RI: {status['ri']:.3f}")
        print(f"  Status: {'🟢 Stable' if status['ri'] < 0.3 else '🟡 Warning' if status['ri'] < 0.5 else '🔴 Critical'}")
        print(f"  Recent alerts: {len(status['recent_alerts'])}")
        if status['recent_alerts']:
            for alert in status['recent_alerts'][-2:]:  # Show last 2 alerts
                print(f"    - {alert}")
    
    print("\n" + "="*50 + "\n")

def demo_performance_comparison():
    """Compare performance of optimized vs standard implementations."""
    print("=== Performance Comparison ===\n")
    
    # Generate test data
    np.random.seed(42)
    test_vectors = np.random.random((1000, 4))
    
    # Test standard RI computation
    print("Testing standard RI computation...")
    start_time = time.time()
    standard_ri = []
    for vector in test_vectors:
        tv = TruthVector(*vector)
        standard_ri.append(tv.reflexive_imbalance())
    standard_time = time.time() - start_time
    
    # Test optimized RI computation
    print("Testing optimized RI computation...")
    start_time = time.time()
    optimized_ri = fast_ri_computation(test_vectors)
    optimized_time = time.time() - start_time
    
    # Compare results
    print(f"\nPerformance Results:")
    print(f"  Standard implementation: {standard_time:.4f} seconds")
    print(f"  Optimized implementation: {optimized_time:.4f} seconds")
    print(f"  Speedup: {standard_time/optimized_time:.1f}x")
    
    # Verify accuracy
    max_diff = np.max(np.abs(np.array(standard_ri) - optimized_ri))
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Results match: {'✅' if max_diff < 1e-10 else '❌'}")
    
    # Test tensor norm computation
    print(f"\nTesting tensor norm computation...")
    test_vector = test_vectors[0]
    
    start_time = time.time()
    tv = TruthVector(*test_vector)
    tensor = BayesTensor(tv)
    standard_norm = tensor.max_norm()
    standard_norm_time = time.time() - start_time
    
    start_time = time.time()
    optimized_norm = fast_tensor_norm(test_vector)
    optimized_norm_time = time.time() - start_time
    
    print(f"  Standard tensor norm: {standard_norm:.6f} ({standard_norm_time*1000:.2f} ms)")
    print(f"  Optimized tensor norm: {optimized_norm:.6f} ({optimized_norm_time*1000:.2f} ms)")
    print(f"  Speedup: {standard_norm_time/optimized_norm_time:.1f}x")
    print(f"  Results match: {'✅' if abs(standard_norm - optimized_norm) < 1e-10 else '❌'}")
    
    print("\n" + "="*50 + "\n")

def demo_distributed_optimization():
    """Demonstrate distributed optimization capabilities."""
    print("=== Distributed Optimization Demonstration ===\n")
    
    # Create multiple initial states with different characteristics
    initial_states = [
        TruthVector(0.9, 0.2, 0.8, 0.1),  # High imbalance
        TruthVector(0.6, 0.95, 0.3, 0.2),  # AI safety scenario
        TruthVector(0.7, 0.98, 0.6, 0.4),  # Climate scenario
        TruthVector(0.5, 0.5, 0.5, 0.5),   # Balanced
        TruthVector(0.1, 0.9, 0.2, 0.8),   # Another high imbalance
    ]
    
    print(f"Optimizing {len(initial_states)} systems to target RI = 0.2")
    
    scalable_system = ScalableIQLSystem()
    start_time = time.time()
    interventions = scalable_system.distributed_optimization(initial_states, target_ri=0.2)
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.4f} seconds")
    
    for i, (initial_state, intervention_list) in enumerate(zip(initial_states, interventions)):
        initial_ri = initial_state.reflexive_imbalance()
        
        # Apply interventions to get final state
        system = IQLSystem(initial_state)
        for intervention in intervention_list:
            system.apply_operator(intervention)
        
        final_ri = system.current_state.reflexive_imbalance()
        
        print(f"\nSystem {i+1}:")
        print(f"  Initial RI: {initial_ri:.3f}")
        print(f"  Final RI: {final_ri:.3f}")
        print(f"  Interventions: {len(intervention_list)}")
        print(f"  Sequence: {intervention_list[:5]}{'...' if len(intervention_list) > 5 else ''}")
        print(f"  Improvement: {initial_ri - final_ri:.3f}")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("🚀 SCALABLE IQL SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    try:
        demo_batch_processing()
        demo_dashboard_monitoring()
        demo_performance_comparison()
        demo_distributed_optimization()
        
        print("✅ All demonstrations completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  • Batch processing of 1000+ truth vectors")
        print("  • Real-time system monitoring with alerts")
        print("  • Performance optimizations with Numba JIT")
        print("  • Distributed optimization across multiple systems")
        print("  • Comprehensive dashboard with recommendations")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc() 