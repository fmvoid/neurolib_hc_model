import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from neurolib.models.aln import ALNModel
import neurolib.utils.functions as func

from pyswarms.utils.plotters import plot_cost_history
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.backend.topology.von_neumann import VonNeumann

class ParameterFitter:
    def __init__(self, subject_id, session, data_dir, param_config='param_config.json', output_dir='output_dir', sim_duration=30000):
        """
        Subject ID needs to be formatted as XXX (eg. 001, 002, 003, etc.)
        Session is either 1 or 2
        """
        self.sub_id = subject_id
        self.session =  session
        self.data_dir = data_dir    
        self.param_config = self.load_param_config(param_config)
        self.output_dir = output_dir
        self.sim_duration = sim_duration

        # Load empirical data
        self.empirical_fc, self.normalized_weights, self.normalized_length = self.load_empirical_data()

        # Initialize the model with the loaded data
        self.model = ALNModel(Cmat=self.normalized_weights, Dmat=self.normalized_length)
        self.model.params['duration'] = self.sim_duration  # 30 seconds by default

    def load_empirical_data(self):
        """
        Loads and normalizes the empirical data for the connectome.

        Returns:
            Tuple: A tuple containing the empirical functional connectivity matrix (empirical_fc),
            the normalized weights of the connectome (normalized_weights), and the normalized lengths
            of the connectome (normalized_length).
        """
        self.empirical_fc = np.loadtxt(f"{self.data_dir}/FC/0{self.sub_id}_{self.session}_RestEmpCorrFC.csv", delimiter=',')
        self.weights = np.loadtxt(f"{self.data_dir}/SC/0{self.sub_id}_{self.session}_Counts.csv", delimiter=',')
        self.length = np.loadtxt(f"{self.data_dir}/SC/0{self.sub_id}_{self.session}_Lengths.csv", delimiter=',')
        self.normalized_weights = self.weights / np.max(self.weights)
        self.normalized_length = self.length / np.max(self.length)

        return self.empirical_fc, self.normalized_weights, self.normalized_length

    def load_param_config(self, param_config_file):
        with open(param_config_file, 'r') as file:
            return json.load(file)

    def get_params_bounds(self):
        lower_bounds = []
        upper_bounds = []

        for param in self.param_config['parameters']:
            lower_bounds.append(param['lower_bound'])
            upper_bounds.append(param['upper_bound'])

        return lower_bounds, upper_bounds
    
    def simulate_fc(self, params):
        # Unpack parameters
        # print(f"params: {params}")  # Debugging line to print the params

        param_names = [param['name'] for param in self.param_config['parameters']]
        param_dict = dict(zip(param_names, params))

        # Set model parameters
        for name, value in param_dict.items():
            self.model.params[name] = value

        # Stage 1: Short simulation to check for activity
        self.model.params['duration'] = 5 * 1000 # 5 seconds
        self.model.run()

        # Check if stage 1 was successful
        if np.max(self.model.output[:, self.model.t > 500]) > 100 or np.max(self.model.output[:, self.model.t > 500]) < 0.1:
            return None

        # Stage 2: Full simulation
        self.model.params['duration'] = self.sim_duration
        self.model.run(chunkwise=True, bold = True)
        
        # Compute FC from BOLD signal
        simulated_fc = func.fc(self.model.BOLD.BOLD[:, 5:])
        return simulated_fc

    def objective_function(self, params):
        """
        Calculates the objective function value for a given set of parameters.

        Parameters:
        - self: The instance of the class.
        - params: A list of parameter sets.

        Returns:
        - results: An array of objective function values for each parameter set.

        Description:
        This function calculates the objective function value for each parameter set in the given list.
        It simulates the functional connectivity (FC) using the parameter set and calculates the correlation
        between the empirical FC and the simulated FC. The objective function value is defined as 1 minus
        the correlation, as we aim to maximize the correlation.

        Note:
        - The empirical FC is accessed using `self.empirical_fc`.
        - The simulated FC is obtained by calling the `simulate_fc` method with each parameter set.
        - The `pearsonr` function from the `scipy.stats` module is used to calculate the correlation.
        """
        results = []
        for param_set in params:
            simulated_fc = self.simulate_fc(param_set)
            if simulated_fc is None:
                results.append(np.inf) # Penalize invalid parameter sets
                # print("Invalid parameter set")
            else:
                correlation, _ = pearsonr(self.empirical_fc.flatten(), simulated_fc.flatten())
                results.append(1 - correlation)  # We minimize 1-correlation to maximize correlation
        return np.array(results)

    def fit(self):
        """
        Fits the model parameters using Particle Swarm Optimization (PSO) algorithm.

        Parameters:
        - n_particles (int): The number of particles in the PSO algorithm. Default is 5.
        - n_iterations (int): The maximum number of iterations for the PSO algorithm. Default is 5.

        Returns:
        - best_pos (list): The best position found by the PSO algorithm.
        - best_cost (float): The cost associated with the best position found by the PSO algorithm.
        """
        # Define bounds
        lower_bounds, upper_bounds = self.get_params_bounds()
        bounds = (lower_bounds, upper_bounds)

        # Initialize PSO optimizer
        optimizer = GlobalBestPSO(
            n_particles= self.param_config['n_particles'], 
            dimensions= self.param_config['dimensions'], 
            options={'w': self.param_config['options']['w'], 
                     'c1': self.param_config['options']["c1"],
                     'c2': self.param_config['options']["c2"]},
            bounds=bounds)

        # Run optimization
        best_cost, best_pos = optimizer.optimize(
            self.objective_function, 
            iters=self.param_config['n_iterations'], 
            n_processes=self.param_config['n_processes'])

        return best_pos, best_cost, optimizer
    
    def plot_cost_history(self, optimizer):
        cost_history = optimizer.cost_history
        plot_cost_history(cost_history)
        plt.savefig(f'{self.output_dir}/cost_history_plot_sub-{self.sub_id}.png', dpi=300)
        print("Cost history plot saved as 'cost_history_plot.png'")

    def print_output_results(self):
        print("Best parameters:")
        print(f"Correlation: {1 - best_fit:.3f}")

    def plot_best_fc(self, best_params):
        simulated_fc = self.simulate_fc(best_params)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.empirical_fc, cmap='viridis', origin='lower')
        plt.title('Empirical FC')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(simulated_fc, cmap='viridis', origin='lower')
        plt.title('Simulated FC')
        plt.colorbar()
        plt.savefig(f'{self.output_dir}/fc_comparison_plot_sub-{self.sub_id}.png', dpi=300)

if __name__ == '__main__':
    # Instantiate the ParameterFitter class
    fitter = ParameterFitter(subject_id='009', session=1, data_dir='/Users/fdjim/Downloads/096-HarvardOxfordMaxProbThr0',
                            param_config='/Users/fdjim/Desktop/Projects/neurolib/src/param_config.json',
                            output_dir='/Users/fdjim/Desktop/Projects/neurolib/output_dir')

    best_params, best_fit, optimizer = fitter.fit()
    fitter.plot_cost_history(optimizer)
    fitter.print_output_results()
    fitter.plot_best_fc(best_params)