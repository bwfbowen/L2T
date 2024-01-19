from src.problem import MultiODProblemV2

problem = MultiODProblemV2()
new_info = problem.info._replace(sequence={1:2})
new_sequence = new_info.sequence
new_sequence[1] = 3
paths = [[node for item in problem.info.od_pairing.items() if problem.info.od_type[item[0]] == 0 for node in item]]
print(paths)
print(new_info.sequence)
print(new_info._replace(sequence=new_sequence).sequence)