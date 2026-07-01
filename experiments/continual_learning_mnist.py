import torch
import networkx as nx
import torchvision
import torchvision.transforms as transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morenet_core.spectral_engine import SpectralMoReNet

def create_small_world_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def filter_dataset(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, indices)

def run_split_mnist():
    print("Inizializzazione Benchmark 1: Split-MNIST su MoReNet (2048 Nodi)")
    num_nodes = 2048
    
    # 1. Inizializzazione Grafo Small-World
    edges = create_small_world_graph(num_nodes, k=10, p=0.1)
    model = SpectralMoReNet(num_nodes=num_nodes, edges=edges, plasticity_threshold=0.05, alpha=0.01)
    
    # 2. Matrice di Proiezione Ortogonale P
    # Mappa 784 pixel -> 2048 nodi in modo ortogonale per evitare collisioni spaziali a priori
    rand_mat = torch.randn(2048, 784, device=model.device)
    P, _ = torch.linalg.qr(rand_mat) # P shape: [2048, 784]
    
    # 3. Mappatura Target (Nodi attuatori)
    target_nodes = {
        0: 100,
        1: 500,
        2: 900,
        3: 1300
    }
    
    trainset, testset = get_mnist_data()
    
    def train_task(task_classes, epochs=1):
        print(f"\n--- Inizio Training Task: Cifre {task_classes} ---")
        task_data = filter_dataset(trainset, task_classes)
        # Limitiamo il training set per velocità sperimentale
        subset_indices = torch.randperm(len(task_data))[:500] 
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(task_data, subset_indices), batch_size=1, shuffle=True)
        
        for epoch in range(epochs):
            for i, (img, label) in enumerate(loader):
                label = label.item()
                img_flat = img.view(784).to(model.device)
                
                # Proiezione ortogonale per generare l'impulso sensoriale
                impulse_in = torch.matmul(P, img_flat)
                # NORMALIZZAZIONE CRITICA: previene l'esplosione dell'energia e la saturazione dei pesi
                impulse_in = impulse_in / (torch.norm(impulse_in) + 1e-8)
                
                # Impulso target (one-hot sul nodo attuatore)
                impulse_target = torch.zeros(num_nodes, device=model.device)
                impulse_target[target_nodes[label]] = 1.0
                
                # Apprendimento (Erosione Topologica)
                model.learn_spectral_hebbian(impulse_in, impulse_target, time_steps=5)
                
                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch+1} | Step {i+1} | W_mean: {model.W_values.mean():.6f} | W_max: {model.W_values.max():.6f}")

    def evaluate(task_classes):
        test_data = filter_dataset(testset, task_classes)
        subset_indices = torch.randperm(len(test_data))[:200]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_data, subset_indices), batch_size=1)
        
        correct = 0
        total = 0
        for img, label in loader:
            label = label.item()
            img_flat = img.view(784).to(model.device)
            impulse_in = torch.matmul(P, img_flat)
            impulse_in = impulse_in / (torch.norm(impulse_in) + 1e-8)
            
            # Propagazione onda in inferenza
            wave_out = model.propagate_wave_chebyshev(impulse_in, time_steps=5)
            
            # Trova il nodo target con la massima energia tra quelli permessi per questo task
            # (o tra tutti i task visti finora)
            energies = {cls: torch.abs(wave_out[target_nodes[cls]]).item() for cls in task_classes}
            prediction = max(energies, key=energies.get)
            
            if prediction == label:
                correct += 1
            total += 1
            
        accuracy = 100 * correct / total
        print(f"Accuratezza sulle cifre {task_classes}: {accuracy:.2f}%")
        return accuracy

    # Esecuzione Continual Learning
    # Task 1
    train_task([0, 1])
    print("\nValutazione dopo Task 1:")
    evaluate([0, 1])
    
    # Task 2
    train_task([2, 3])
    print("\nValutazione dopo Task 2 (Test Invarianza Temporale):")
    evaluate([2, 3])
    print("Verifica Oblio Catastrofico sul Task 1:")
    evaluate([0, 1])

if __name__ == "__main__":
    run_split_mnist()
