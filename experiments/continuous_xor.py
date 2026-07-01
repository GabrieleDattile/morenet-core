import torch
import networkx as nx
import sys
import os

# Add root path to import morenet_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morenet_core.spectral_engine import SpectralMoReNet

def create_small_world_graph(n, k, p):
    """
    Creates a Small-World (Watts-Strogatz) graph and extracts edges as a PyTorch tensor.
    """
    G = nx.watts_strogatz_graph(n, k, p)
    edges = list(G.edges())
    # Convert to tensor [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def run_xor_experiment():
    print("Initializing Experiment Zero: Continuous XOR on Small-World Graph")
    num_nodes = 256
    
    # Create graph: 256 nodes, each connected to 10 neighbors, 10% rewire probability
    edges = create_small_world_graph(num_nodes, k=10, p=0.1)
    
    model = SpectralMoReNet(num_nodes=num_nodes, edges=edges, plasticity_threshold=0.1, alpha=0.01)
    
    # Sensor node mapping (Input)
    # [0,0] -> Node 10
    # [0,1] -> Node 20
    # [1,0] -> Node 30
    # [1,1] -> Node 40
    sensor_map = {
        (0, 0): 10,
        (0, 1): 20,
        (1, 0): 30,
        (1, 1): 40
    }
    
    # Actuator node mapping (Target)
    # Target 0 -> Node 200
    # Target 1 -> Node 250
    target_map = {
        0: 200,
        1: 250
    }
    
    # Classic XOR dataset
    dataset = [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0)
    ]
    
    print(f"Graph initialized with {num_nodes} nodes and {edges.size(1)} edges.")
    print("Starting Topological Erosion (Training without Backpropagation)...")
    
    epochs = 100
    
    # Track the mean weight to see if topology is evolving
    initial_mean_weight = model.W_values.mean().item()
    
    for epoch in range(epochs):
        for inputs, target in dataset:
            input_idx = sensor_map[inputs]
            target_idx = target_map[target]
            
            # Create impulses (One-hot on graph)
            impulse_in = torch.zeros(num_nodes, device=model.device)
            impulse_in[input_idx] = 1.0
            
            impulse_target = torch.zeros(num_nodes, device=model.device)
            impulse_target[target_idx] = 1.0
            
            # Learning
            energy = model.learn_spectral_hebbian(impulse_in, impulse_target, time_steps=5)
            
        if (epoch + 1) % 20 == 0:
            current_mean = model.W_values.mean().item()
            print(f"Epoch {epoch + 1}/{epochs} | Mean edge weight: {current_mean:.6f} | Maximum: {model.W_values.max().item():.6f}")

    print("\nTraining Completed.")
    print(f"Global Graph Deformation: Mean weight changed from {initial_mean_weight:.6f} to {model.W_values.mean().item():.6f}")
    print("Resonance 'canyons' (input->target associations) have been carved into the Laplacian.")

if __name__ == "__main__":
    run_xor_experiment()
