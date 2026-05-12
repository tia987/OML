# global imports
import json
import math
import numpy as np
import os
import subprocess
import scipy.stats.qmc as qmc
import time


def getConfig():
    # The dsParameterBounds names need to match the NX parameter names in the part file as well as the journal file!
    config_file = os.path.join(os.getcwd(), 'config.json')
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config
    except Exception as e:
        print("Configuration file not found. Make sure it is in the same folder as the main.py script")
        print(f"Error: {e}")
        return None


def generateSamples(config, nSamples):
    """
    Generates Latin Hypercube samples for given parameters and saves each sample as a separate JSON file.

    Parameters:
    - parameters: dict, keys are parameter names and values are [min, max] lists
    - numSamples: int, number of samples to generate
    """
    parameters = config["dsParameterBounds"]

    sampler = qmc.LatinHypercube(d=len(parameters))
    samples = sampler.random(nSamples)
    scaled_samples = qmc.scale(
        samples,
        [v[0] for v in parameters.values()],
        [v[1] for v in parameters.values()]
    )

    sampleLst = []
    for i in range(nSamples):
        param_dict = {k: float(v) for k, v in zip(parameters.keys(), scaled_samples[i])}
        sampleLst.append(param_dict)

    return sampleLst


def build_command(sample, journalFile, freeCadpath):
    freecad_exec_path = os.path.join(freeCadpath, r"FreeCAD.app/Contents/MacOS/FreeCAD")
    cmd = [freecad_exec_path, journalFile, json.dumps(sample)]
    return cmd


def calculate_objective(sample, freeCAD_journal, freeCADpath):
    """
    Runs the FreeCAD journal for one sample and returns:
    (absorbed_energy, max_stress)
    """
    res = None
    try:
        cmd = build_command(sample, freeCAD_journal, freeCADpath)
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if res.returncode != 0:
            print("FreeCAD returned a non-zero exit code.")
            print("STDOUT:")
            print(res.stdout)
            print("STDERR:")
            print(res.stderr)
            return None

        stdout_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if len(stdout_lines) < 2:
            print("Unexpected FreeCAD output. Not enough lines to parse results.")
            print("STDOUT:")
            print(res.stdout)
            print("STDERR:")
            print(res.stderr)
            return None

        absorbed_energy = float(stdout_lines[-2])
        max_stress = float(stdout_lines[-1])

        print(".")
        return absorbed_energy, max_stress

    except subprocess.CalledProcessError as e:
        print(f"Sample processing failed: FreeCAD crashed or returned an error (exit code {e.returncode}).")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        time.sleep(1)
        return None

    except Exception as e:
        print(f"Sample processing failed due to error: {e}")
        if res is not None:
            print("STDOUT:")
            print(res.stdout)
            print("STDERR:")
            print(res.stderr)
        time.sleep(1)
        return None


def main():
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
    freeCAD_journal = os.path.join(os.getcwd(), "full_journal_fc.py")

    nSamples = 3
    config = getConfig()
    if config is None:
        return

    samples = generateSamples(config, nSamples)

    results = []
    for sample in samples:
        objective = calculate_objective(sample, freeCAD_journal, freeCAD_path)
        if objective is not None:
            absorbed_energy, max_stress = objective
            sample['energy_objective'] = absorbed_energy
            sample['stress_constraint'] = max_stress
            results.append(sample)

    print(results)

    if len(results) == 0:
        print("No valid samples were evaluated successfully.")
        return

    idx = min(1, len(results) - 1)
    print('objective value (total plastic deformation energy):', results[idx]['energy_objective'])
    print('maximum stress value:', results[idx]['stress_constraint'])


if __name__ == "__main__":
    main()