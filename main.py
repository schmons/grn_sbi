import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE
from grn import GeneRegulatoryNetwork

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Starting parameter inference for Gene Regulatory Network...")

# Define the true parameters for generating synthetic data
true_params = {
    'alpha': 250.0,
    'alpha0': 0.2,
    'beta': 5.5,
    'n': 3.0,
    'K': 35.0
}

# Define prior distributions for each parameter
prior_min = torch.tensor([100.0, 0.01, 2.0, 1.0, 10.0])
prior_max = torch.tensor([400.0, 1.0, 10.0, 5.0, 100.0])
prior = utils.BoxUniform(prior_min, prior_max)

# Define a simulator function that takes parameters and returns summary statistics
def simulator(params):
    # Convert parameters from tensor to numpy array
    params_np = params.numpy()
    
    # Create GRN with the given parameters
    grn = GeneRegulatoryNetwork(
        alpha=params_np[0],
        alpha0=params_np[1],
        beta=params_np[2],
        n=params_np[3],
        K=params_np[4]
    )
    
    # Run simulation
    initial_conditions = np.array([0, 0, 0, 2, 1, 3])
    t, y = grn.simulate(
        initial_conditions=initial_conditions,
        t_span=(0, 100),
        t_points=500
    )
    
    # Extract summary statistics (we'll use protein levels at specific time points)
    # For simplicity, we'll sample 20 evenly spaced time points from the simulation
    indices = np.linspace(0, len(t)-1, 20, dtype=int)
    
    # Extract protein levels (last 3 rows of y) at the selected time points
    protein_levels = y[3:6, indices].flatten()
    
    # Convert to tensor
    return torch.tensor(protein_levels, dtype=torch.float32)

# Generate synthetic observed data with the true parameters
print("Generating synthetic observed data with true parameters...")
start_time = time.time()
true_params_tensor = torch.tensor([
    true_params['alpha'],
    true_params['alpha0'],
    true_params['beta'],
    true_params['n'],
    true_params['K']
], dtype=torch.float32)

x_o = simulator(true_params_tensor)
print(f"Generated observed data in {time.time() - start_time:.2f} seconds")

# Prepare simulator for SBI
print("Preparing simulator for SBI...")
# The simulator_wrapper function doesn't exist in the current version
# Let's just use the simulator directly
simulator_wrapper = simulator

# Run simulations for training the neural network
num_simulations = 1000
print(f"Running {num_simulations} simulations for training...")
start_time = time.time()

# Manual simulation approach
theta = prior.sample((num_simulations,))
x = torch.zeros((num_simulations, 60))  # 60 = 3 proteins × 20 time points

for i in tqdm(range(num_simulations), desc="Simulating"):
    x[i] = simulator(theta[i])

print(f"Completed {num_simulations} simulations in {time.time() - start_time:.2f} seconds")

# Train the neural network with SNPE (Sequential Neural Posterior Estimation)
print("Training neural density estimator...")
start_time = time.time()
inference = SNPE(prior=prior)
density_estimator = inference.append_simulations(theta, x).train(show_train_summary=True)
print(f"Completed training in {time.time() - start_time:.2f} seconds")

print("Building posterior...")
posterior = inference.build_posterior(density_estimator)

# Sample from the posterior given the observed data
num_samples = 10000
print(f"Sampling {num_samples} samples from posterior...")
start_time = time.time()
posterior_samples = posterior.sample((num_samples,), x=x_o)
print(f"Completed sampling in {time.time() - start_time:.2f} seconds")

# Calculate summary statistics of the posterior
posterior_means = posterior_samples.mean(dim=0)
posterior_stds = posterior_samples.std(dim=0)

# Print results
param_names = ['alpha', 'alpha0', 'beta', 'n', 'K']
print("\nParameter Inference Results:")
print("-" * 50)
print(f"{'Parameter':<10} {'True Value':<15} {'Inferred Mean':<15} {'Inferred Std':<15}")
print("-" * 50)
for i, name in enumerate(param_names):
    print(f"{name:<10} {true_params[name]:<15.4f} {posterior_means[i].item():<15.4f} {posterior_stds[i].item():<15.4f}")

# Plot posterior distributions
print("Generating posterior plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, name in enumerate(param_names):
    if i < len(axes):
        print(f"Plotting posterior for {name}...")
        # Create a histogram of the posterior samples for this parameter
        axes[i].hist(posterior_samples[:, i].numpy(), bins=30, density=True, alpha=0.7)
        axes[i].axvline(true_params[name], color='red', linestyle='--', label='True Value')
        axes[i].set_title(f"{name} (true: {true_params[name]:.2f})")
        axes[i].set_xlabel(name)
        axes[i].set_ylabel('Density')
        axes[i].legend()

if len(param_names) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
print("Saving parameter posteriors plot...")
plt.savefig('parameter_posteriors.png')
plt.show()

# Plot pairwise posteriors for selected parameters
print("Generating pairwise posterior plot...")
# Create a figure with a grid of subplots for pairwise relationships
fig, axes = plt.subplots(len(param_names), len(param_names), figsize=(15, 15))

# Plot pairwise relationships
for i in range(len(param_names)):
    for j in range(len(param_names)):
        ax = axes[i, j]
        
        if i == j:  # Diagonal: histogram of single parameter
            ax.hist(posterior_samples[:, i].numpy(), bins=30, density=True)
            ax.axvline(true_params[param_names[i]], color='red', linestyle='--')
            ax.set_title(param_names[i])
        elif i < j:  # Upper triangle: scatter plot
            ax.scatter(posterior_samples[:, j].numpy(), posterior_samples[:, i].numpy(), 
                      alpha=0.1, s=1)
            ax.scatter(true_params[param_names[j]], true_params[param_names[i]], 
                      color='red', marker='x', s=100)
        else:  # Lower triangle: 2D histogram
            h = ax.hist2d(posterior_samples[:, j].numpy(), posterior_samples[:, i].numpy(), 
                         bins=30, cmap='Blues')
            ax.scatter(true_params[param_names[j]], true_params[param_names[i]], 
                      color='red', marker='x', s=100)
        
        if i == len(param_names) - 1:  # Bottom row
            ax.set_xlabel(param_names[j])
        if j == 0:  # Leftmost column
            ax.set_ylabel(param_names[i])

plt.tight_layout()
print("Saving pairwise posteriors plot...")
plt.savefig('pairwise_posteriors.png')
plt.show()

# Validate the inference by simulating with inferred parameters
print("\nValidating inference by simulating with inferred parameters...")
start_time = time.time()

# Create GRN with true parameters
grn_true = GeneRegulatoryNetwork(
    alpha=true_params['alpha'],
    alpha0=true_params['alpha0'],
    beta=true_params['beta'],
    n=true_params['n'],
    K=true_params['K']
)

# Create GRN with inferred parameters (using posterior means)
grn_inferred = GeneRegulatoryNetwork(
    alpha=posterior_means[0].item(),
    alpha0=posterior_means[1].item(),
    beta=posterior_means[2].item(),
    n=posterior_means[3].item(),
    K=posterior_means[4].item()
)

# Run simulations
initial_conditions = np.array([0, 0, 0, 2, 1, 3])
print("Running simulation with true parameters...")
t_true, y_true = grn_true.simulate(
    initial_conditions=initial_conditions,
    t_span=(0, 100),
    t_points=500
)

print("Running simulation with inferred parameters...")
t_inferred, y_inferred = grn_inferred.simulate(
    initial_conditions=initial_conditions,
    t_span=(0, 100),
    t_points=500
)
print(f"Completed validation simulations in {time.time() - start_time:.2f} seconds")

# Plot comparison
print("Generating validation plot...")
plt.figure(figsize=(12, 8))

# Plot protein levels
for i in range(3):
    plt.plot(t_true, y_true[i+3], label=f"True Protein {i+1}", linestyle='-')
    plt.plot(t_inferred, y_inferred[i+3], label=f"Inferred Protein {i+1}", linestyle='--')

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Comparison of True vs. Inferred Dynamics")
plt.legend()
plt.grid(True, alpha=0.3)
print("Saving validation plot...")
plt.savefig('validation_plot.png')
plt.show()

print("Parameter inference process completed!")
