import numpy as np
import zlib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import re

def generate_mock_tensor(seed=42, size=4):
    np.random.seed(seed)
    t = np.random.uniform(0.1, 0.9, size)
    B = np.log(np.outer(1/t, t))
    return B

def load_from_csv(file_path, quadrant_cols):
    df = pd.read_csv(file_path)
    t = []
    for col in quadrant_cols:
        avg = df[col].mean() if col in df else 0.5
        t.append(np.clip(avg / df[col].max(), 0.1, 0.9) if df[col].max() else 0.5)
    t = np.array(t)
    B = np.log(np.outer(1/t, t))
    return B

def discretize_tensor(B, thresholds=[-0.5, 0.5]):
    disc = np.zeros_like(B, dtype=int)
    disc[B < thresholds[0]] = -1
    disc[(B >= thresholds[0]) & (B < thresholds[1])] = 0
    disc[B >= thresholds[1]] = 1
    return disc

def apply_ca_rule(grid, rule='rule30', randomness=0.1):
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for i in range(rows):
        for j in range(cols):
            left = grid[i, (j-1) % cols]
            center = grid[i, j]
            right = grid[i, (j+1) % cols]
            bin_val = (left + 1) * 4 + (center + 1) * 2 + (right + 1)
            new_val = (30 >> bin_val) & 1 - 1
            if np.random.rand() < randomness:
                new_val = np.random.choice([-1, 0, 1])
            new_grid[i, j] = new_val
    return new_grid

def calculate_half_life(epsilon):
    """Compute half-life tau_1/2 from epsilon, per user's formula."""
    if epsilon <= 0 or epsilon > 1:
        return float('inf')
    return 0.5 / np.log(1 / (1 - epsilon)**2)  # Approximate from lambda = -ln((1-ε)^2)

def simulate_operators(grid, operator, epsilon=0.1):
    if operator == 'alpha':
        grid[0, :] = -grid[0, :]
    elif operator == 'beta':
        grid[1, :] = (grid[1, :] * 0.5).astype(int)
        grid[1, :] = np.clip(grid[1, :], -1, 1)
    elif operator == 'gamma':
        grid[2, :] = np.mean(grid[[1,2,3], :], axis=0).astype(int)
    elif operator == 'delta':
        # Revised δ with epsilon: Convex combo on row 3 (t_Its)
        bar_t = np.mean(grid[:3, :], axis=0)  # Mean of first three rows
        grid[3, :] = ((1 - epsilon) * grid[3, :] + epsilon * bar_t).astype(int)
        grid[3, :] = np.clip(grid[3, :], -1, 1)
    return grid

def compute_entropy(states):
    flattened = [tuple(s.flatten()) for s in states]
    unique = len(set(flattened))
    data = ''.join(str(x) for s in states for x in s.flatten())
    compressed = zlib.compress(data.encode())
    comp_ratio = len(compressed) / len(data.encode()) if data else 0
    return unique, comp_ratio

def compute_hysteresis(states):
    if len(states) < 2:
        return 0
    diff = np.linalg.norm(states[-1] - states[0])
    return diff / np.linalg.norm(states[0])

def plot_evolution(states):
    fig, axs = plt.subplots(1, len(states), figsize=(3*len(states), 3))
    for i, grid in enumerate(states):
        axs[i].imshow(grid, cmap='RdBu', vmin=-1, vmax=1)
        axs[i].set_title(f'Step {i}')
        axs[i].axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def run_simulation(steps=5, operator_sequence=[], size=4, randomness=0.1, epsilon=0.1, data_file=None, quadrant_cols=None):
    if data_file:
        B = load_from_csv(data_file, quadrant_cols or ['col1', 'col2', 'col3', 'col4'])
    else:
        B = generate_mock_tensor(size=size)
    grid = discretize_tensor(B)
    states = [grid.copy()]
    for step in range(steps):
        for op in operator_sequence:
            grid = simulate_operators(grid, op, epsilon=epsilon)  # Pass epsilon to ops
        grid = apply_ca_rule(grid, randomness=randomness)
        states.append(grid.copy())
    ent_unique, ent_comp = compute_entropy(states)
    hyst = compute_hysteresis(states)
    plot_b64 = plot_evolution(states)
    half_life = calculate_half_life(epsilon)
    return B, states[-1], ent_unique, ent_comp, hyst, plot_b64, half_life

# Example run with epsilon=0.13 (5-cycle calibration)
initial_B, final_grid, ent_unique, ent_comp, hyst, plot, hl = run_simulation(steps=5, operator_sequence=['alpha', 'delta'], size=4, randomness=0.05, epsilon=0.13)
print("Initial B:\n", initial_B)
print("Final Grid:\n", final_grid)
print("Entropy (Unique Patterns):", ent_unique)
print("Entropy (Compression Ratio):", ent_comp)
print("Hysteresis Norm:", hyst)
print("Half-Life (tau_1/2):", hl)
print("Evolution Plot (Base64 - paste to decoder):\n", plot[:100] + '...')

# --- Save the base64 string and decode to PNG ---
with open('evolution_plot_base64.txt', 'w') as f:
    f.write(plot)

with open('evolution_plot.png', 'wb') as f:
    f.write(base64.b64decode(plot))

print('Saved evolution_plot_base64.txt and evolution_plot.png')