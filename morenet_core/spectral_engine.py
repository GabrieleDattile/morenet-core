import torch
import torch.nn.functional as F

class SpectralMoReNet:
    def __init__(self, num_nodes, edges, plasticity_threshold=0.6, alpha=0.05, lambda_max=2.0):
        """
        Inizializza la Spectral Morphogenetic Resonance Network.
        
        :param num_nodes: Numero totale di nodi nel grafo semantico.
        :param edges: Tensore di shape (2, num_edges) contenente gli indici degli archi iniziali.
        :param plasticity_threshold: Soglia di energia necessaria per scavare il "canyon".
        :param alpha: Learning rate / tasso di erosione topologica.
        :param lambda_max: Stima dell'autovalore massimo per scalare l'approssimazione di Chebyshev.
        """
        self.num_nodes = num_nodes
        self.plasticity = plasticity_threshold
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inizializza i pesi degli archi in modo sparso con piccoli valori (tensione iniziale)
        # Assicuriamo che gli archi siano simmetrici (grafo non orientato)
        edges = torch.cat([edges, edges[[1, 0]]], dim=1).unique(dim=1)
        num_edges = edges.size(1)
        
        # Valori iniziali (es. 0.1 per tutti gli archi)
        values = torch.full((num_edges,), 0.1, dtype=torch.float32, device=self.device)
        self.edge_indices = edges.to(self.device)
        self.W_values = values

    def compute_laplacian_sparse(self):
        """
        Calcola la matrice Laplaciana sparsa L = D - W
        """
        W_sparse = torch.sparse_coo_tensor(self.edge_indices, self.W_values, (self.num_nodes, self.num_nodes))
        
        # Gradi dei nodi (somma dei pesi per ogni riga)
        degrees = torch.sparse.sum(W_sparse, dim=1).to_dense()
        
        # Costruisce la matrice diagonale D sparsa
        indices_D = torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        D_sparse = torch.sparse_coo_tensor(indices_D, degrees, (self.num_nodes, self.num_nodes))
        
        # L = D - W
        L = D_sparse - W_sparse
        return L

    def propagate_wave_chebyshev(self, impulse, time_steps, order=5, c=1.0):
        """
        Risolve la propagazione d'onda usando i polinomi di Chebyshev troncati, 
        evitando la decomposizione autovettoriale esplicita. O(K * |E|).
        """
        L = self.compute_laplacian_sparse()
        
        # Scala L affinché gli autovalori siano in [-1, 1]
        # L_scaled = (2.0 / lambda_max) * L - I
        I_indices = torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        I_values = torch.ones(self.num_nodes, device=self.device)
        I_sparse = torch.sparse_coo_tensor(I_indices, I_values, (self.num_nodes, self.num_nodes))
        
        L_scaled = torch.sparse_coo_tensor(
            L._indices(), 
            (2.0 / self.lambda_max) * L._values(), 
            (self.num_nodes, self.num_nodes)
        ) - I_sparse
        
        # Prepariamo i coefficienti di Chebyshev per la funzione cos(sqrt(lambda) * t)
        # Nota: In un'implementazione fisica rigorosa, i coefficienti andrebbero 
        # calcolati proiettando la funzione coseno sulla base di Chebyshev.
        # Qui usiamo coefficienti empirici semplificati per dimostrazione.
        coeffs = [1.0, 0.5, -0.25, -0.1, 0.05, 0.01] # Approssimazione toy
        
        # Ricorrenza di Chebyshev (vettore colonna)
        impulse = impulse.unsqueeze(1) if impulse.dim() == 1 else impulse
        
        T_prev = impulse
        # Moltiplicazione matrice sparsa - vettore denso
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
        Aggiorna la topologia del grafo in base all'interferenza tra onde.
        """
        wave_in = self.propagate_wave_chebyshev(impulse_in, time_steps)
        wave_target = self.propagate_wave_chebyshev(impulse_target, time_steps)
        
        # Interferenza costruttiva/distruttiva locale
        wave_tot = wave_in + wave_target
        
        # Calcolo energia sui nodi
        energy = torch.abs(wave_tot)
        
        # Per aggiornare gli archi, verifichiamo l'energia congiunta ai capi di ogni arco
        # Estraiamo i nodi sorgente e destinazione per ogni arco
        src, dst = self.edge_indices[0], self.edge_indices[1]
        
        # Energia congiunta sull'arco (prodotto dell'energia dei due nodi)
        edge_energy = energy[src] * energy[dst]
        
        # Maschera di erosione (Heaviside step)
        erosion_mask = (edge_energy > self.plasticity).float()
        
        # Aggiornamento pesi: incrementiamo W_values dove l'energia supera la soglia
        delta_W = self.alpha * (edge_energy - self.plasticity) * erosion_mask
        self.W_values = self.W_values + delta_W
        
        # Clamp per stabilità (nessun peso maggiore di 1.0)
        self.W_values = torch.clamp(self.W_values, max=1.0)
        
        return energy
