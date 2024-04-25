from typing import Dict, NamedTuple, List, Any
import numpy as np
import dm_env


Path = List[int]

class SequenceInfo(NamedTuple):
    sequence: int = 0
    index: int = 0


class ProblemInfo(NamedTuple):
    locations: np.ndarray = None 
    distance_matrix: np.ndarray = None 
    od_pairing: Dict[int, int] = {} 
    od_type: Dict[int, int] = {} # -1 for depot, 0 for O, 1 for D
    sequence: Dict[int, SequenceInfo] = {}


class Improvement(NamedTuple):
    delta: float = 0.
    delta_best: float = 0.
    no_improvement: int = 0


class BestSol(NamedTuple):
    paths: Path = None
    best_cost: float = np.inf 
    best_step: int = 0


class EnvState(NamedTuple):
    problem: ProblemInfo = None
    action_history: np.ndarray = None   
    improvement: Improvement = None  


class ObservationExtras(NamedTuple):
    node_features: np.ndarray
    operator_features: np.ndarray