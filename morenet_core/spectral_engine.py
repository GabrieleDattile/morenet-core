import torch
import torch.nn.functional as F

class SpectralMoReNet:
    def __init__(self, num_nodes, edges, plasticity_threshold=0.6, alpha=0.05, lambda_max=2.0):
        """
        Initializes the Spectral Morphogenetic Resonance Network.
        
        :param num_nodes: Total number of nodes in the semantic graph.
        :param edges: Tensor of shape (2, num_edges) containing initial edge indices.
        :param plasticity_threshold: Energy threshold required to carve a "canyon".
        :param alpha: Learning rate / topological erosion rate.
        :param lambda_max: Maximum eigenvalue estimate to scale Chebyshev approximation.
        """
        self.num_nodes = num_nodes
        self.plasticity = plasticity_threshold
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize edge weights sparsely with small values (initial tension)
        # Ensure edges are symmetric (undirected graph)
        edges = torch.cat([edges, edges[[1, 0]]], dim=1).unique(dim=1)
        num_edges = edges.size(1)
        
        # Initial values (e.g., 0.1 for all edges)
        values = torch.full((num_edges,), 0.1, dtype=torch.float32, device=self.device)
        self.edge_indices = edges.to(self.device)
        self.W_values = values

    def compute_laplacian_sparse(self):
        """
        Computes the sparse Laplacian matrix L = D - W
        """
        W_sparse = torch.sparse_coo_tensor(self.edge_indices, self.W_values, (self.num_nodes, self.num_nodes))
        
        # Node degrees (sum of weights for each row)
        degrees = torch.sparse.sum(W_sparse, dim=1).to_dense()
        
        # Build sparse diagonal matrix D
        indices_D = torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        D_sparse = torch.sparse_coo_tensor(indices_D, degrees, (self.num_nodes, self.num_nodes))
        
        # L = D - W
        L = D_sparse - W_sparse
        return L

    def propagate_wave_chebyshev(self, impulse, time_steps, order=5, c=1.0):
        """
        Solves wave propagation using truncated Chebyshev polynomials, 
        avoiding explicit eigenvector decomposition. O(K * |E|).
        """
        L = self.compute_laplacian_sparse()
        
        # Scale L so eigenvalues are in [-1, 1]
        I_indices = torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        I_values = torch.ones(self.num_nodes, device=self.device)
        I_sparse = torch.sparse_coo_tensor(I_indices, I_values, (self.num_nodes, self.num_nodes))
        
        L_scaled = torch.sparse_coo_tensor(
            L._indices(), 
            (2.0 / self.lambda_max) * L._values(), 
            (self.num_nodes, self.num_nodes)
        ) - I_sparse
        
        # Prepare Chebyshev coefficients for the function cos(sqrt(lambda) * t)
        # Note: In a rigorous physical implementation, coefficients should be 
        # calculated by projecting the cosine function onto the Chebyshev basis (using Bessel functions).
        # Here we use simplified empirical coefficients for demonstration.
        coeffs = [1.0, 0.5, -0.25, -0.1, 0.05, 0.01] # Toy approximation
        
        # Chebyshev recurrence (column vector)
        impulse = impulse.unsqueeze(1) if impulse.dim() == 1 else impulse
        
        T_prev = impulse
        # Sparse matrix - dense vector multiplication
        T_curr = torch.sparse.mm(L_scaled, impulse)
        
        wave = coeffs[0] * T_prev + coeffs[1] * T_curr
        
        for k in range(2, order):
            # T_k = 2 * x * T_{k-1} - T_{k-2}
            T_next = 2.0 * torch.sparse.mm(L_scaled, T_curr) - T_prev
            wave += coeffs[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            
        return wave.squeeze()

    def learn_spectral_hebbian(self, impulse_in, impulse_target, time_steps=5):
        """
        Updates the graph topology based on wave interference.
        """
        wave_in = self.propagate_wave_chebyshev(impulse_in, time_steps)
        wave_target = self.propagate_wave_chebyshev(impulse_target, time_steps)
        
        # Local constructive/destructive interference
        wave_tot = wave_in + wave_target
        
        # Calculate energy on nodes
        energy = torch.abs(wave_tot)
        
        # To update edges, we check joint energy at the ends of each edge
        src, dst = self.edge_indices[0], self.edge_indices[1]
        
        # Joint edge energy (product of energy of the two nodes)
        edge_energy = energy[src] * energy[dst]
        
        # Erosion mask (Heaviside step)
        erosion_mask = (edge_energy > self.plasticity).float()
        
        # Weight update: increment W_values where energy exceeds threshold
        delta_W = self.alpha * (edge_energy - self.plasticity) * erosion_mask
        self.W_values = self.W_values + delta_W
        
        # Clamp for stability (no weight > 1.0)
        self.W_values = torch.clamp(self.W_values, max=1.0)
        
        return energy
