# global imports
import json
import numpy as np
import os
import subprocess
import time
import csv

import scipy.stats.qmc as qmc
import matplotlib.pyplot as plt

from plotter import plot, plot_plate, data_analysis

n_generations = 15        # Number of iterations to find the best layout
stress_limit = 275.0     # Constraint from project description
nSamples = 10

def getConfig():
    # The dsParameterBounds names need to match the NX parameter names in the part file as well as the journal file!
    config_file = os.path.join(os.getcwd(), 'config.json')
    try:
        # Open and read the JSON file
        with open(config_file, 'r') as file:
            config = json.load(file)
    except Exception as e:
        print("Configuration file not found. Make sure it is in the same folder as the main.py script")
    return config

def generateSamples(config, nSamples):
    """
    Generates Latin Hypercube samples for given parameters and saves each sample as a separate JSON file.
    
    Parameters:
    - parameters: dict, keys are parameter names and values are [min, max] lists
    - numSamples: int, number of samples to generate
    """
    # create one dictionary containing all parameters that have ranges
    parameters = config["dsParameterBounds"]

    # Generate input files for latin hypercube samples
    sampler = qmc.LatinHypercube(d=len(parameters)) 
    samples = sampler.random(nSamples)
    scaled_samples = qmc.scale(samples, [v[0] for v in parameters.values()], [v[1] for v in parameters.values()])
    sampleLst = []
    for i in range(nSamples):
        param_dict = {k:float(v) for k,v in zip(parameters.keys(), scaled_samples[i])}
        sampleLst.append(param_dict)
    return sampleLst

def build_command(sample, journalFile, freeCadpath):
    freecad_exec_path = os.path.join(freeCadpath,  r"FreeCAD.app/Contents/Resources/bin/freecadcmd")
    cmd = [freecad_exec_path, journalFile, json.dumps(sample),] 
    return cmd

def calculate_objective(sample, freeCAD_journal, freeCADpath):
    # create necessary folders
    # try processing the current sample geometry
    res = None
    try:
        # run freecad journal, that updates the geometry with current parameters
        # and solves the fe simulation and returns the deformation energy
        cmd = build_command(sample, freeCAD_journal , freeCADpath)
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)# creationflags=subprocess.CREATE_NO_WINDOW)
        assert res.returncode == 0 , "freeCAD script routine failed"
        # retrieve the absorbed energy from the logged string
        absorbed_energy = float(res.stdout.split('\n')[-3])
        max_stress = float(res.stdout.split('\n')[-2])
    # catch exceptions and log failed samples. DB entry at index (sampleId) is invalid
    except Exception as e:
        print(f"Sample processing failed due to error: {e}")
        if res is not None:
            print("STDOUT:")
            print(res.stdout)
            print("STDERR:")
            print(res.stderr)
        time.sleep(1)
        return None

    # return binary files for optional backward conversion to geometries
    return absorbed_energy, max_stress

def save_to_csv(results, filename="data.csv"):
    if not results:
        return
    field = sorted({k for row in results for k in row.keys()})
    with open(filename, mode="a", newline="") as f:
        write = csv.DictWriter(f, fieldnames=field)
        write.writeheader()
        for row in results:
            write.writerow(row)
    
def gen_algo(verbose=False, mutation_rate=0.2, n_population=10):
    """
    In this example, a latin hypercube sampling is performed on the parameter bounds
    and the geometries are evaluated for there objective of energy absorbtion. 
    TODO:
    Your task is to implement an optimizer to find the optimal geometry maximizing the
    energy absorption capacity. This can include better sampling strategies, choosing 
    a suitable optimizer and handling parameter combinations that yield invalid geometries.

    Do not forget to fulfil the maximum stress constrain as well as the geometry 
    constraints listed in section 2 of the project description.

    """
    # TODO: set the freecad path your the installation location of freeCAD
    freeCAD_path = r'/Applications/'

    # freecad scripts for geometry manipulation and fem simulation
    '''
    if the script does not work initially, copy the full_journal_fc.py file
    to the bin folder in the FreeCAD 1.0 directory,  i.e for windows default
    'C:/Program Files/FreeCAD 1.0/bin/full_journal_fc.py' and change the 
    freeCAD_journal variable below to this path.
    After running like that once, you should be able to copy the file back and
    run normally.
    '''
    freeCAD_journal = os.path.join(os.getcwd(), "full_journal_fc.py")
    config = getConfig()
    parameters = config["dsParameterBounds"]

    # Optimization Settings
    penalty_factor = 2000.0

    best_energy_history = []

    # Generate initial random guesses for genetic algorithm
    print("Generating initial population via Latin Hypercube Sampling...")
    population = generateSamples(config, n_population)

    # Optimization Loop
    for generation in range(n_generations):
        print(f"\n================ Generation {generation + 1} / {n_generations} ================")
        results = []

        # Evaluate population
        for sample in population:
            result = calculate_objective(sample, freeCAD_journal, freeCAD_path)
            
            if result is None:
                # Set penalized score so optimizer ignores failed results
                sample['penalized_score'] = 1e5
                sample['energy_objective'] = 0.0
                sample['stress_constraint'] = 999.0
                results.append(sample)
                continue
            
            absorbed_energy, max_stress = result
            sample['energy_objective'] = absorbed_energy
            sample['stress_constraint'] = max_stress

            # Calculate Penalty if stress exceeds 275 MPa
            stress_violation = max(0.0, max_stress - stress_limit)
            penalty = penalty_factor * (stress_violation ** 2)

            # Maximise energy by minimizing score
            sample['penalized_score'] = -absorbed_energy + penalty
            results.append(sample)

        # Sort population based on best (lowest) penalized score
        results.sort(key=lambda x: x['penalized_score'])

        valid_designs = [d for d in results if d['stress_constraint'] <= stress_limit and d['energy_objective'] > 0]
        if valid_designs:
            best_valid_energy = valid_designs[0]['energy_objective']
            print(f"-> Best Feasible Energy in Generation {generation + 1}: {best_valid_energy:.4f}")
        else:
            best_valid_energy = 0.0
            print("-> Warning: No valid designs found in this generation (all violated stress or broke geometry).")
        
        best_energy_history.append(best_valid_energy)

        # Setup next generation, keep top 3 performers as parents
        parents = results[:3]
        new_population = []
        new_population.append({k: parents[0][k] for k in parameters.keys()})

        # Fill up the rest of the population with mutated versions of the parents
        while len(new_population) < n_population:
            # Pick random parent
            parent = np.random.choice(parents)
            child = {}
            for k, bounds in parameters.items():
                # Add random shift to parent's coordinate
                bound_range = bounds[1] - bounds[0]
                mutation = np.random.normal(0, mutation_rate * bound_range)
                new_value = parent[k] + mutation
                # Ensure new value doesn't clip outside the boundaries defined in config.json
                child[k] = float(np.clip(new_value, bounds[0], bounds[1]))
            new_population.append(child)
        # As children as new population
        population = new_population

    # Optimization complete: Display final results
    if verbose:
        print("\n================ OPTIMIZATION COMPLETE ================")
        print("Best parameters found:")
        print(json.dumps({k: results[0][k] for k in parameters.keys()}, indent=4))
        print(f"Achieved Energy: {results[0]['energy_objective']:.4f}")
        print(f"Max Stress: {results[0]['stress_constraint']:.2f} MPa")

    # Save results into csv file
    save_to_csv(results)

    # Generate Convergence Plot
    if verbose:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, n_generations + 1), best_energy_history, marker='o', color='b')
        plt.title('Optimizer Convergence')
        plt.xlabel('Generation / Iteration')
        plt.ylabel('Best Feasible Absorbed Energy')
        plt.grid(True)
        # add text box for statistics
        best = results[0]
        stats = (
            f'$x_1$ = {best["x1"]:.2f}  (range: {parameters["x1"]})\n'
            f'$y_1$ = {best["y1"]:.2f}  (range: {parameters["y1"]})\n'
            f'$x_2$ = {best["x2"]:.2f}  (range: {parameters["x2"]})\n'
            f'$y_2$ = {best["y2"]:.2f}  (range: {parameters["y2"]})\n'
            f'$x_3$ = {best["x3"]:.2f}  (range: {parameters["x3"]})\n'
            f'$y_3$ = {best["y3"]:.2f}  (range: {parameters["y3"]})\n'
            f'$angle$ = {best["angle"]:.2f}°  (range: {parameters["angle"]})\n'
        )
        bbox = dict(boxstyle='round', fc='white', ec='black', alpha=1)
        ax = plt.gca()
        plt.text(0.95, 0.07, stats, fontsize=9, bbox=bbox,
                transform=ax.transAxes, horizontalalignment='right')
        plt.savefig('optimizer_convergence.png')
        plt.show()

    best_design = results[0] if results else None
    return best_energy_history, best_design

def base(verbose=False):
    freeCAD_path = r'/Applications/'
    freeCAD_journal = os.path.join(os.getcwd(), "full_journal_fc.py")
    config = getConfig()
    samples = generateSamples(config, nSamples)
    best_energy_history = []
    best_design = None
    for generation in range(n_generations):
        print(f"\n================ Iteration {generation + 1} / {n_generations} ================")
        results = []
        for sample in samples:
            geom_keys = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'angle']
            clean_sample = {k: sample[k] for k in geom_keys if k in sample}
            result = calculate_objective(clean_sample, freeCAD_journal, freeCAD_path)
            if result is None:
                print(f"Skipping failed sample.")
                continue
            absorbed_energy, max_stress = result
            if absorbed_energy is not None:
                sample['energy_objective'] = absorbed_energy
                sample['stress_constraint'] = max_stress
                results.append(sample)
        valid_designs = [d for d in results if d['stress_constraint'] <= stress_limit and d['energy_objective'] > 0]
        if valid_designs:
            # Sort descending by energy objective to find the maximum
            valid_designs.sort(key=lambda x: x['energy_objective'], reverse=True)
            best_valid_energy = valid_designs[0]['energy_objective']
            best_design = valid_designs[0]
        else:
            best_valid_energy = 0.0
            if results:
                best_design = results[0]

        best_energy_history.append(best_valid_energy)

        if verbose and len(results) > 0:
            print(results)
            idx = 1
            print('objective value (total plastic deformation energy):', results[idx]['energy_objective'])
            print('maximum stress value:', results[idx]['stress_constraint'])

    return best_energy_history, best_design

if __name__ == "__main__":
    result_0, base_best = base()
    result_1, gen_best_1 = gen_algo()
    result_2, gen_best_2 = gen_algo(n_population=20)
    result_3, gen_best_3 = gen_algo(mutation_rate=0.5)
    result_4, gen_best_4 = gen_algo(mutation_rate=0.7, n_population=30)

    results_list = [
        (result_0, "Latin Hypercube Baseline"),
        (result_1, "GA (Pop=10, Mut=0.2)"),
        (result_2, "GA (Pop=20, Mut=0.2)"),
        (result_3, "GA (Pop=10, Mut=0.5)"),
        (result_4, "GA (Pop=30, Mut=0.7)")]

    plot(n_generations, results_list)
    plot_plate(base_best, "_base_best")
    plot_plate(gen_best_2, "_gen_best")
    data_analysis()