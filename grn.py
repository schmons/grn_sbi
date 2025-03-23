import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class GeneRegulatoryNetwork:
    """
    A simple gene regulatory network model with tunable parameters.
    
    Parameters:
    -----------
    n_genes : int
        Number of genes in the network
    interaction_matrix : numpy.ndarray or None
        Matrix defining interactions between genes (activating/repressing)
    hill_coefficients : numpy.ndarray or None
        Hill coefficients for each interaction
    activation_thresholds : numpy.ndarray or None
        Activation thresholds for each interaction
    basal_rates : numpy.ndarray or None
        Basal expression rates for each gene
    degradation_rates : numpy.ndarray or None
        Degradation rates for each gene product
    """
    
    def __init__(self, n_genes=3, interaction_matrix=None, hill_coefficients=None, 
                 activation_thresholds=None, basal_rates=None, degradation_rates=None):
        self.n_genes = n_genes
        
        # Initialize parameters with default values if not provided
        if interaction_matrix is None:
            # Default: gene 0 activates gene 1, gene 1 represses gene 2, gene 2 represses gene 0
            self.interaction_matrix = np.array([
                [0, 1, -0.5],
                [0, 0, 1],
                [-0.8, 0, 0]
            ])
        else:
            self.interaction_matrix = interaction_matrix
            
        if hill_coefficients is None:
            # Default Hill coefficients (cooperativity)
            self.hill_coefficients = np.ones((n_genes, n_genes)) * 2
        else:
            self.hill_coefficients = hill_coefficients
            
        if activation_thresholds is None:
            # Default activation thresholds
            self.activation_thresholds = np.ones((n_genes, n_genes)) * 0.5
        else:
            self.activation_thresholds = activation_thresholds
            
        if basal_rates is None:
            # Default basal expression rates
            self.basal_rates = np.ones(n_genes) * 0.1
        else:
            self.basal_rates = basal_rates
            
        if degradation_rates is None:
            # Default degradation rates
            self.degradation_rates = np.ones(n_genes) * 0.2
        else:
            self.degradation_rates = degradation_rates
    
    def regulation_function(self, x, i, j):
        """
        Compute the regulatory effect of gene j on gene i.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current gene expression levels
        i : int
            Target gene index
        j : int
            Regulator gene index
            
        Returns:
        --------
        float
            Regulatory effect
        """
        if self.interaction_matrix[i, j] == 0:
            return 0
        
        hill_coef = self.hill_coefficients[i, j]
        threshold = self.activation_thresholds[i, j]
        
        if self.interaction_matrix[i, j] > 0:  # Activation
            return self.interaction_matrix[i, j] * (x[j]**hill_coef) / (threshold**hill_coef + x[j]**hill_coef)
        else:  # Repression
            return self.interaction_matrix[i, j] * (threshold**hill_coef) / (threshold**hill_coef + x[j]**hill_coef)
    
    def dynamics(self, t, x):
        """
        Define the system dynamics dx/dt.
        
        Parameters:
        -----------
        t : float
            Current time point
        x : numpy.ndarray
            Current gene expression levels
            
        Returns:
        --------
        numpy.ndarray
            Rate of change for each gene
        """
        dx_dt = np.zeros(self.n_genes)
        
        for i in range(self.n_genes):
            # Basal expression
            dx_dt[i] = self.basal_rates[i]
            
            # Add regulatory effects from all genes
            for j in range(self.n_genes):
                dx_dt[i] += self.regulation_function(x, i, j)
            
            # Degradation
            dx_dt[i] -= self.degradation_rates[i] * x[i]
        
        return dx_dt
    
    def simulate(self, initial_conditions=None, t_span=(0, 100), t_points=1000):
        """
        Simulate the gene regulatory network dynamics.
        
        Parameters:
        -----------
        initial_conditions : numpy.ndarray or None
            Initial gene expression levels
        t_span : tuple
            Time span for simulation (start, end)
        t_points : int
            Number of time points to evaluate
            
        Returns:
        --------
        tuple
            (time_points, gene_expression_trajectories)
        """
        if initial_conditions is None:
            initial_conditions = np.random.rand(self.n_genes) * 0.1
        
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        
        # Solve the ODE system
        solution = solve_ivp(
            self.dynamics,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method='RK45'
        )
        
        return solution.t, solution.y
    
    def plot_simulation(self, t, y, title="Gene Regulatory Network Dynamics"):
        """
        Plot the simulation results.
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time points
        y : numpy.ndarray
            Gene expression trajectories
        title : str
            Plot title
        """
        plt.figure(figsize=(10, 6))
        
        for i in range(self.n_genes):
            plt.plot(t, y[i], label=f"Gene {i}")
        
        plt.xlabel("Time")
        plt.ylabel("Expression Level")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create a GRN with default parameters
    grn = GeneRegulatoryNetwork()
    
    # Simulate the network
    t, y = grn.simulate()
    
    # Plot the results
    grn.plot_simulation(t, y)
    
    # Example with custom parameters
    custom_interaction = np.array([
        [0, 1.2, -0.3],
        [-0.8, 0, 0.9],
        [0.5, -0.4, 0]
    ])
    
    custom_hill = np.ones((3, 3)) * 2.5
    custom_thresholds = np.ones((3, 3)) * 0.3
    custom_basal = np.array([0.05, 0.08, 0.12])
    custom_degradation = np.array([0.1, 0.15, 0.2])
    
    custom_grn = GeneRegulatoryNetwork(
        n_genes=3,
        interaction_matrix=custom_interaction,
        hill_coefficients=custom_hill,
        activation_thresholds=custom_thresholds,
        basal_rates=custom_basal,
        degradation_rates=custom_degradation
    )
    
    # Simulate with custom parameters
    t_custom, y_custom = custom_grn.simulate(t_span=(0, 200))
    
    # Plot custom simulation
    custom_grn.plot_simulation(t_custom, y_custom, "Custom GRN Dynamics")
