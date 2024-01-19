import os
import time
import re
import random
from tqdm import tqdm

from src.problem import MultiODProblem
from src.solution import MultiODSolution
from src.utils import read_instance_data
from src.gurobi.gurobi_formulation import formulation
from src.gurobi.utils import display_gurobi_result
from src.ortools.ortools_formulation import ortools_formulation, ortools_pd_formulation_2D
from src.ortools.utils import display_ortools_result, generate_paths_from_ortools_result, display_pd_ortools_result

num_O = '1000'
resample_k = 1

def run_experiment(instance):
    report_str = ""
    locations = read_instance_data(instance)
    p = MultiODProblem(locations=locations, ignore_to_dummy_cost=False, ignore_from_dummy_cost=False)
    start_time = time.time()
    solution = ortools_pd_formulation_2D(p)
    end_time = time.time()
    results = display_pd_ortools_result(p, solution)
    report_str += f"{results[0]}\n"
    report_str += f"Time taken: {end_time-start_time} seconds"
    return report_str, results[1]

instance_dir = os.path.join('data', 'tsppdlib', 'instances', 'random-uniform')
instances = [i for i in os.listdir(instance_dir) if i.endswith('.tsp')]

sub_instances = [i for i in instances if '-' + num_O + '-' in i]
sub_instances = random.sample(sub_instances, k=resample_k)

selected_files = sub_instances
for selected_file in tqdm(selected_files, desc="Processing files", unit="file"):
    instance = os.path.join(instance_dir, selected_file)
    results = run_experiment(instance)
    file_name = selected_file.replace(".tsp", "")
    file_path = f"../tmp/ortools/{file_name}.{results[1]}.txt"
    with open(file_path, 'w+') as file:
        file.write(f"The result for {selected_file}\n")
        file.write(f"{results[0]}\n")