import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import ParameterGrid
import time
from neurolib.models.aln import ALNModel
from joblib import Parallel, delayed
import multiprocessing
import itertools
import math

class ParameterExplorer:
    def __init__(
        self,
        model=None,
        parameterSpace=None,
        evalFunction=None,
        filename=None,
        saveAllModelOutputs=False,
        ncores=None,
    ):
        """
        Initialize the ParameterExplorer with a model, parameter space, and optional evaluation function.

        :param model: Model to run for each parameter, defaults to None
        :type model: `neurolib.models.model.Model`, optional
        :param parameterSpace: Parameter space to explore, defaults to None
        :type parameterSpace: dict, optional
        :param evalFunction: Evaluation function to call for each run, defaults to None
        :type evalFunction: function, optional
        :param filename: HDF5 storage file name, defaults to "exploration.hdf"
        :type filename: str, optional
        :param saveAllModelOutputs: If True, save all outputs of the model, defaults to False
        :type saveAllModelOutputs: bool, optional
        :param ncores: Number of cores to simulate on, defaults to None
        :type ncores: int, optional
        """
        self.model = model
        self.parameterSpace = parameterSpace
        self.evalFunction = evalFunction
        self.filename = filename if filename else f"exploration_{time.strftime('%Y%m%d-%H%M%S')}.h5"
        self.saveAllModelOutputs = saveAllModelOutputs
        self.ncores = ncores
        self.results = []

    def run_single_parameter_set(self, **params):
        # Set model parameters
        for param, value in params.items():
            self.model.params[param] = value

        # Run the model
        self.model.run(chunkwise=True, chunksize=5000, bold=False)  # Removed BOLD
        # self.model.run()

        if self.saveAllModelOutputs:
            outputs = self.model.outputs
        else:
            outputs = {
                'rates_exc': self.model.outputs.rate_exc,
                'rates_inh': self.model.outputs.rate_inh,
                'IA': self.model.outputs.IA,
            }

        return {
            'params': params,
            'outputs': outputs
        }

    def evaluate_model(self, params):
        if self.evalFunction:
            return self.evalFunction(self.model, params)
        else:
            return self.run_single_parameter_set(**params)


    def explore_parameter_space(self):
        param_grid = list(ParameterGrid(self.parameterSpace))
        num_cores_avail = multiprocessing.cpu_count()  # Get the number of available cores
        num_cores = num_cores_avail - 1 if self.ncores is None else self.ncores  # Use all but 2 cores by default
        print(f"Exploring {len(param_grid)} parameter sets using {num_cores} cores...")
        results = Parallel(n_jobs=num_cores)(delayed(self.evaluate_model)(params) for params in param_grid)
        self.results = [result for result in results if result]
        self.save_results_to_hdf5()


    def save_results_to_hdf5(self):
        try:
            with h5py.File(self.filename, 'a') as hf:  # Open in append mode
                for i, result in enumerate(self.results):
                    group_name = f"result_{i + len(hf.keys())}"  # Ensure unique group names
                    group = hf.create_group(group_name)
                    param_group = group.create_group('params')
                    
                    for param_key, param_value in result['params'].items():
                        param_group.create_dataset(param_key, data=param_value)

                    if self.saveAllModelOutputs:
                        for key, value in result['outputs'].items():
                            # Check if the value can be converted to a numpy array
                            try:
                                value = np.asarray(value)
                                group.create_dataset(key, data=value)
                            except Exception as e:
                                print(f"Skipping key '{key}' due to an error: {e}")
                    else:
                        # Only save selected outputs
                        for key, value in result['outputs'].items():
                            if value is not None:
                                try:
                                    value = np.asarray(value)
                                    group.create_dataset(key, data=value)
                                except Exception as e:
                                    print(f"Skipping key '{key}' due to an error: {e}")
            print(f"Results saved to {self.filename}")
        except IOError as e:
            print(f"An error occurred while saving to HDF5: {e}")
            
    def plot_parameter_space(self):
        """
        Plot the results of a multi-parameter exploration.
        Creates a matrix of plots for each output, showing how it changes across 2D slices of the parameter space.
        """
        # Load results from HDF5 file
        with h5py.File(self.filename, 'r') as hf:
            results = []
            for group_name in hf.keys():
                result = {}
                group = hf[group_name]
                result['params'] = {k: v[()] for k, v in group['params'].items()}
                result['outputs'] = {k: v[()] for k, v in group.items() if k != 'params'}
                results.append(result)

        # Determine unique output names and parameter names
        output_names = set()
        param_names = set()
        for result in results:
            output_names.update(result['outputs'].keys())
            param_names.update(result['params'].keys())

        # Generate all possible 2D parameter combinations
        param_combinations = list(itertools.combinations(param_names, 2))

        # Create a plot for each output and parameter combination
        for output_name in output_names:
            # Calculate the number of subplots needed
            n_plots = len(param_combinations)
            n_cols = min(3, n_plots)  # Max 3 columns
            n_rows = math.ceil(n_plots / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_plots == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for i, (param1, param2) in enumerate(param_combinations):
                ax = axes[i]
                
                # Create a 2D grid for the output values
                param_values1 = sorted(set(result['params'][param1] for result in results))
                param_values2 = sorted(set(result['params'][param2] for result in results))
                grid = np.full((len(param_values1), len(param_values2)), np.nan)
                
                for result in results:
                    if output_name in result['outputs']:
                        i1 = param_values1.index(result['params'][param1])
                        i2 = param_values2.index(result['params'][param2])
                        # Use the mean of the output if it's a time series
                        grid[i1, i2] = np.mean(result['outputs'][output_name])

                # Create the heatmap
                im = ax.imshow(grid, origin='lower', aspect='auto', interpolation='nearest')
                plt.colorbar(im, ax=ax)

                # Set tick labels
                ax.set_xticks(np.arange(len(param_values2))[::len(param_values2)//5])
                ax.set_yticks(np.arange(len(param_values1))[::len(param_values1)//5])
                ax.set_xticklabels([f'{v:.2f}' for v in param_values2[::len(param_values2)//5]], rotation=45)
                ax.set_yticklabels([f'{v:.2f}' for v in param_values1[::len(param_values1)//5]])

                # Set labels
                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
                ax.set_title(f'{output_name}: {param1} vs {param2}')

            # Remove any unused subplots
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig(f'{output_name}_parameter_space.png')
            plt.close()

        print(f"Plots saved as PNG files in the current directory.")


# Usage example
weights_file = "/Users/fdjim/Desktop/Projects/neurolib/data/sub-0001/sc/0001_1_Counts.csv"
length_file = "/Users/fdjim/Desktop/Projects/neurolib/data/sub-0001/sc/0001_1_Lengths.csv" 

weights = np.loadtxt(weights_file, delimiter=',')
length = np.loadtxt(length_file, delimiter=',')

# Normalize each connectome matrix
normalized_weights = weights / np.max(weights)
normalized_length = length / np.max(length)

model = ALNModel(Cmat=normalized_weights, Dmat=normalized_length)

model.params['duration'] = 1000 * 10  # ms
# model.params.sigma_ou = 0.2
model.params.b = 24
model.params.a = 28

# model.params.mue_ext_mean = 1.6  # mV/ms (default is 0.4)
# model.params.mui_ext_mean = 0.05 # mV/ms (default is 0.3)

parameterSpace = {
    'mue_ext_mean': np.linspace(0.0, 4, 10),
    'mui_ext_mean': np.linspace(0.0, 4, 10),
    'sigma_ou':np.linspace(0.0, 0.3, 6)
}

explorer = ParameterExplorer(model=model, parameterSpace=parameterSpace, saveAllModelOutputs=True)
explorer.explore_parameter_space()
explorer.plot_parameter_space()