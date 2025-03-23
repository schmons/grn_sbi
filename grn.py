import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class GeneRegulatoryNetwork:
    """
    A simplified repressilator model based on the classic paper by Elowitz and Leibler.
    
    This model implements a 3-gene repressilator with parameters specifically tuned
    to produce oscillations.
    """
    
    def __init__(self, alpha=216, alpha0=0.216, beta=5, n=2, K=40):
        """
        Initialize the repressilator model with parameters from literature.
        
        Parameters:
        -----------
        alpha : float
            Maximum transcription rate in the absence of repressor
        alpha0 : float
            Leakiness of promoter (basal transcription rate)
        beta : float
            Ratio of protein decay rate to mRNA decay rate
        n : float
            Hill coefficient (cooperativity)
        K : float
            Repression coefficient
        """
        self.alpha = alpha
        self.alpha0 = alpha0
        self.beta = beta
        self.n = n
        self.K = K
        self.n_genes = 3  # Fixed for repressilator
    
    def dynamics(self, t, state):
        """
        Define the system dynamics.
        
        Parameters:
        -----------
        t : float
            Current time point
        state : numpy.ndarray
            Current state [m1, m2, m3, p1, p2, p3] where m is mRNA and p is protein
            
        Returns:
        --------
        numpy.ndarray
            Rate of change for each state variable
        """
        # Unpack state variables
        m1, m2, m3, p1, p2, p3 = state
        
        # Ensure non-negative values
        m1, m2, m3 = max(0, m1), max(0, m2), max(0, m3)
        p1, p2, p3 = max(0, p1), max(0, p2), max(0, p3)
        
        # Calculate derivatives
        dm1_dt = -m1 + self.alpha/(1 + (p3/self.K)**self.n) + self.alpha0
        dm2_dt = -m2 + self.alpha/(1 + (p1/self.K)**self.n) + self.alpha0
        dm3_dt = -m3 + self.alpha/(1 + (p2/self.K)**self.n) + self.alpha0
        
        dp1_dt = -self.beta * (p1 - m1)
        dp2_dt = -self.beta * (p2 - m2)
        dp3_dt = -self.beta * (p3 - m3)
        
        return [dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt]
    
    def simulate(self, initial_conditions=None, t_span=(0, 100), t_points=1000):
        """
        Simulate the repressilator dynamics.
        
        Parameters:
        -----------
        initial_conditions : numpy.ndarray or None
            Initial state [m1, m2, m3, p1, p2, p3]
        t_span : tuple
            Time span for simulation (start, end)
        t_points : int
            Number of time points to evaluate
            
        Returns:
        --------
        tuple
            (time_points, state_trajectories)
        """
        if initial_conditions is None:
            # Asymmetric initial conditions to break symmetry
            initial_conditions = np.array([0, 0, 0, 2, 1, 3])
        
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
    
    def plot_simulation(self, t, y, title="Repressilator Dynamics"):
        """
        Plot the simulation results.
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time points
        y : numpy.ndarray
            State trajectories
        title : str
            Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Plot mRNA levels
        plt.subplot(2, 1, 1)
        for i in range(3):
            plt.plot(t, y[i], label=f"mRNA {i+1}")
        plt.xlabel("Time")
        plt.ylabel("mRNA Concentration")
        plt.title(f"{title} - mRNA Levels")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot protein levels
        plt.subplot(2, 1, 2)
        for i in range(3):
            plt.plot(t, y[i+3], label=f"Protein {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Protein Concentration")
        plt.title(f"{title} - Protein Levels")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create a repressilator with parameters known to produce oscillations
    grn = GeneRegulatoryNetwork(alpha=216, alpha0=0.216, beta=5, n=2, K=40)
    
    # Simulate the network with lower initial conditions
    initial_conditions = np.array([60, 0, 0, 0.5, 0.2, 0.8])
    t, y = grn.simulate(
        initial_conditions=initial_conditions,
        t_span=(0, 200), 
        t_points=2000
    )
    
    # Plot the results
    grn.plot_simulation(t, y, "Repressilator")
    
    # Try a different parameter set with stronger oscillations
    grn2 = GeneRegulatoryNetwork(alpha=300, alpha0=0.1, beta=5, n=4, K=20)
    
    # Simulate with different initial conditions
    initial_conditions2 = np.array([60, 0, 0, 0.1, 0.3, 0.5])
    t2, y2 = grn2.simulate(
        initial_conditions=initial_conditions2,
        t_span=(0, 200),
        t_points=2000
    )
    
    # Plot the second simulation
    grn2.plot_simulation(t2, y2, "Repressilator with Stronger Oscillations")
