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
    print("Initializing Benchmark 1: Split-MNIST on MoReNet (2048 Nodes)")
    num_nodes = 2048
    
    # 1. Small-World Graph Initialization
    edges = create_small_world_graph(num_nodes, k=10, p=0.1)
    model = SpectralMoReNet(num_nodes=num_nodes, edges=edges, plasticity_threshold=0.05, alpha=0.01)
    
    # 2. Orthogonal Projection Matrix P
    # Maps 784 pixels -> 2048 nodes orthogonally to prevent spatial collisions apriori
    rand_mat = torch.randn(2048, 784, device=model.device)
    P, _ = torch.linalg.qr(rand_mat) # P shape: [2048, 784]
    
    # 3. Target Mapping (Actuator nodes)
    target_nodes = {
        0: 100,
        1: 500,
        2: 900,
        3: 1300
    }
    
    trainset, testset = get_mnist_data()
    
    def train_task(task_classes, epochs=1):
        print(f"\n--- Starting Training Task: Digits {task_classes} ---")
        task_data = filter_dataset(trainset, task_classes)
        # Limit training set for experimental speed
        subset_indices = torch.randperm(len(task_data))[:500] 
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(task_data, subset_indices), batch_size=1, shuffle=True)
        
        for epoch in range(epochs):
            for i, (img, label) in enumerate(loader):
                label = label.item()
                img_flat = img.view(784).to(model.device)
                
                # Orthogonal projection to generate sensory impulse
                impulse_in = torch.matmul(P, img_flat)
                # CRITICAL NORMALIZATION: prevents energy explosion and weight saturation
                impulse_in = impulse_in / (torch.norm(impulse_in) + 1e-8)
                
                # Target impulse (one-hot on actuator node)
                impulse_target = torch.zeros(num_nodes, device=model.device)
                impulse_target[target_nodes[label]] = 1.0
                
                # Learning (Topological Erosion)
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
            
            # Wave propagation during inference
            wave_out = model.propagate_wave_chebyshev(impulse_in, time_steps=5)
            
            # Find target node with max energy among allowed tasks
            energies = {cls: torch.abs(wave_out[target_nodes[cls]]).item() for cls in task_classes}
            prediction = max(energies, key=energies.get)
            
            if prediction == label:
                correct += 1
            total += 1
            
        accuracy = 100 * correct / total
        print(f"Accuracy on digits {task_classes}: {accuracy:.2f}%")
        return accuracy

    # Continual Learning Execution
    # Task 1
    train_task([0, 1])
    print("\nEvaluation after Task 1:")
    evaluate([0, 1])
    
    # Task 2
    train_task([2, 3])
    print("\nEvaluation after Task 2 (Temporal Invariance Test):")
    evaluate([2, 3])
    print("Checking Catastrophic Forgetting on Task 1:")
    evaluate([0, 1])

if __name__ == "__main__":
    run_split_mnist()
