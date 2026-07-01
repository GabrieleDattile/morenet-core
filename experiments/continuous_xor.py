import torch
import networkx as nx
import sys
import os

# Aggiunge il percorso radice per importare morenet_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morenet_core.spectral_engine import SpectralMoReNet

def create_small_world_graph(n, k, p):
    """
    Crea un grafo Small-World (Watts-Strogatz) e ne estrae gli archi come tensore PyTorch.
    """
    G = nx.watts_strogatz_graph(n, k, p)
    edges = list(G.edges())
    # Converte in tensore [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def run_xor_experiment():
    print("Inizializzazione Esperimento Zero: Continuous XOR su Grafo Small-World")
    num_nodes = 256
    
    # Crea grafo: 256 nodi, ogni nodo connesso a 10 vicini, probabilità di rewire 10%
    edges = create_small_world_graph(num_nodes, k=10, p=0.1)
    
    model = SpectralMoReNet(num_nodes=num_nodes, edges=edges, plasticity_threshold=0.1, alpha=0.01)
    
    # Mappatura nodi sensori (Input)
    # [0,0] -> Nodo 10
    # [0,1] -> Nodo 20
    # [1,0] -> Nodo 30
    # [1,1] -> Nodo 40
    sensor_map = {
        (0, 0): 10,
        (0, 1): 20,
        (1, 0): 30,
        (1, 1): 40
    }
    
    # Mappatura nodi attuatori (Target)
    # Target 0 -> Nodo 200
    # Target 1 -> Nodo 250
    target_map = {
        0: 200,
        1: 250
    }
    
    # Dataset XOR classico
    dataset = [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0)
    ]
    
    print(f"Grafo inizializzato con {num_nodes} nodi e {edges.size(1)} archi.")
    print("Avvio dell'Erosione Topologica (Training senza Backpropagation)...")
    
    epochs = 100
    
    # Tracciamo il peso medio per vedere se la topologia sta evolvendo
    initial_mean_weight = model.W_values.mean().item()
    
    for epoch in range(epochs):
        for inputs, target in dataset:
            input_idx = sensor_map[inputs]
            target_idx = target_map[target]
            
            # Crea impulsi (One-hot su grafo)
            impulse_in = torch.zeros(num_nodes, device=model.device)
            impulse_in[input_idx] = 1.0
            
            impulse_target = torch.zeros(num_nodes, device=model.device)
            impulse_target[target_idx] = 1.0
            
            # Apprendimento
            energy = model.learn_spectral_hebbian(impulse_in, impulse_target, time_steps=5)
            
        if (epoch + 1) % 20 == 0:
            current_mean = model.W_values.mean().item()
            print(f"Epoca {epoch + 1}/{epochs} | Peso medio archi: {current_mean:.6f} | Massimo: {model.W_values.max().item():.6f}")

    print("\nTraining Completato.")
    print(f"Deformazione Globale del Grafo: Peso medio passato da {initial_mean_weight:.6f} a {model.W_values.mean().item():.6f}")
    print("I 'canyon' di risonanza (associazioni input->target) sono stati scolpiti nel Laplaciano.")

if __name__ == "__main__":
    run_xor_experiment()
