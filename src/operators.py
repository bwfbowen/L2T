from typing import Union 
import random
import copy
from collections import defaultdict
import numpy as np

from . import solution
from . import utils 
from src import types

EPSILON = 1e-5

Solution = solution.Solution
MultiODSolution = solution.MultiODSolution
MultiODPath = solution.MultiODPath
Node = solution.Node
SliceableDeque = utils.SliceableDeque


class Operator:
    def __init__(self, operator_type: str):
        self.operator_type = operator_type
    
    def __call__(self, solution: Solution):
        improved_solution, delta, label = solution, 0, None 
        return improved_solution, delta, label
          

class TwoOptOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='in-block')
    
    def __call__(self, solution: MultiODSolution, block_id: int, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        block = path.get_by_block_id(block_id)
        n = len(block) - 1
        label = None
        for first in range(n):
            node1 = block[first]
            prev1 = node1.prev_node.node_id if node1.prev_node is not None else 0
            for second in range(first + 1, n + 1):
                node2 = block[second]
                next2 = node2.next_node.node_id if node2.next_node is not None else 0
                before = (
                    path.get_distance_by_node_ids(prev1, node1.node_id)
                    + path.get_distance_by_node_ids(node2.node_id, next2)
                    )
                after = (
                    path.get_distance_by_node_ids(prev1, node2.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next2)
                )
                
                delta = after - before
                if delta < min_delta:
                    min_delta = delta
                    label = node1.seq_id, node2.seq_id
        if label is None:
            return None, None, None
        else:
            improved_path = path
            first, second = label[0], label[1]
            improved_path = solution.reverse_within_path(first, second, path=improved_path)
            return improved_path, min_delta, label
        

class SegmentTwoOptOperator(Operator):
    def __init__(self, include_taxi_node: bool = True):
        super().__init__(operator_type='path')
        self.include_taxi_node = include_taxi_node
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        _start = self._start
        segment_starts, temp_Ds = SliceableDeque([_start]), set([path.OD_mapping[path.get_by_seq_id(_start).node_id]])
        for seq_id in range(_start + 1, n):
            node_id = path.get_by_seq_id(seq_id).node_id
            if node_id in path.OD_mapping:
                temp_Ds.add(path.OD_mapping[node_id])
            else:
                if node_id in temp_Ds:
                    segment_starts.append(seq_id)
                    temp_Ds = set([node_id])
        segment_starts.append(n)
        label = None
        for i in range(0, len(segment_starts) - 1):
            start, end = segment_starts[i], segment_starts[i + 1]
            for first in range(start, end - 1):
                node1 = path.get_by_seq_id(first)
                prev1 = node1.prev_node.node_id if node1.prev_node is not None else 0
                for second in range(first + 1, end):
                    node2 = path.get_by_seq_id(second)
                    next2 = node2.next_node.node_id if node2.next_node is not None else 0
                    before = (
                        path.get_distance_by_node_ids(prev1, node1.node_id)
                        + path.get_distance_by_node_ids(node2.node_id, next2)
                        )
                    after = (
                        path.get_distance_by_node_ids(prev1, node2.node_id)
                        + path.get_distance_by_node_ids(node1.node_id, next2)
                        )
                    delta = after - before
                    if delta < min_delta:
                        min_delta = delta
                        label = node1.seq_id, node2.seq_id
        if label is None:
            return None, None, None
        else:
            improved_path = path
            first, second = label[0], label[1]
            improved_path = solution.reverse_within_path(first, second, path=improved_path)
            return improved_path, min_delta, label
        

class SegmentTwoOptOperatorV2(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: solution.MultiODSolutionV2, path_id: int = 0, min_delta=-EPSILON):
        path: types.Path = solution.paths[path_id]
        info = solution.info
        n = len(path) - 1

        starts_from = 1
        segment_starts, temp_Ds = SliceableDeque([starts_from]), set([info.od_pairing[path[starts_from]]])
        for seq_id in range(starts_from + 1, n):
            node_id = path[seq_id]
            if info.od_type[node_id] == 0:
                temp_Ds.add(info.od_pairing[node_id])
            else:
                if node_id in temp_Ds:
                    segment_starts.append(seq_id)
                    temp_Ds = set([node_id])
        segment_starts.append(n)
        label = None
        for i in range(0, len(segment_starts) - 1):
            start, end = segment_starts[i], segment_starts[i + 1]
            for first in range(start, end - 1):
                node1 = path[first]
                prev1 = path[first - 1] 
                for second in range(first + 1, end):
                    node2 = path[second]
                    next2 = path[second + 1]
                    before = (
                        info.distance_matrix[prev1, node1] 
                        + info.distance_matrix[node2, next2])
                    after = (
                        info.distance_matrix[prev1, node2]
                        + info.distance_matrix[node1, next2])
                    delta = after - before
                    if delta < min_delta:
                        min_delta = delta
                        label = first, second
        if label is None:
            return None, None, None
        else:
            first, second = label[0], label[1]
            path[first: second + 1] = path[first: second + 1][::-1]
            for idx, n in enumerate(path):
                info.sequence[n] = types.SequenceInfo(sequence=path_id, index=idx)
            return solution, min_delta, label        
        

class TwoKOptOperator(Operator):
    def __init__(self, include_taxi_node: bool = True):
        super().__init__(operator_type='path')
        self.include_taxi_node = include_taxi_node
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        F, R = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))
        F_label, R_label = defaultdict(list), defaultdict(list)
        _start = self._start
        for seq_id1 in range(n - 1, _start, -1):
            node1: Node = path.get_by_seq_id(seq_id1)
            next1 = node1.next_node.node_id if node1.next_node is not None else 0 
            for seq_id2 in range(seq_id1 + 1, n):
                node2: Node = path.get_by_seq_id(seq_id2)
                prev2 = node2.prev_node.node_id
                before = (
                        path.get_distance_by_node_ids(node1.node_id, next1)
                        + path.get_distance_by_node_ids(prev2, node2.node_id)
                        )
                after = (
                    path.get_distance_by_node_ids(prev2, node1.node_id)
                    + path.get_distance_by_node_ids(node2.node_id, next1)
                    )
                delta = after - before
                if seq_id2 <= seq_id1 + 1:
                    F[seq_id1, seq_id2] = 0
                else:
                    _cases = [F[seq_id1 + 1, seq_id2], F[seq_id1, seq_id2 - 1], delta + R[seq_id1 + 1, seq_id2 - 1]]
                    min_idx = np.argmin(_cases)
                    F[seq_id1, seq_id2] = _cases[min_idx]
                    if min_idx == 0:
                        F_label[(seq_id1, seq_id2)] = F_label[(seq_id1 + 1, seq_id2)]
                    elif min_idx == 1:
                        F_label[(seq_id1, seq_id2)] = F_label[(seq_id1, seq_id2 - 1)]
                    else:
                        F_label[(seq_id1, seq_id2)] = [(seq_id1, seq_id2)] + R_label[(seq_id1 + 1, seq_id2 - 1)]

                if ((node1.OD_type == 0 and seq_id1 + 1 <= path.get_by_node_id(path.OD_mapping[node1.node_id]).seq_id <= seq_id2)
                    or
                    (node2.OD_type == 1 and seq_id1 <= path.get_by_node_id(path.DO_mapping[node2.node_id]).seq_id <= seq_id2 - 1)):
                    R[seq_id1, seq_id2] = np.inf
                elif seq_id1 == seq_id2:
                    R[seq_id1, seq_id2] = 0
                elif seq_id2 == seq_id1 + 1 and not (is_od := node1.OD_type == 0 and path.get_by_node_id(path.OD_mapping[node1.node_id]).node_id == node2.node_id):
                    R[seq_id1, seq_id2] = 0
                elif seq_id2 == seq_id1 + 1 and is_od:
                    R[seq_id1, seq_id2] = np.inf 
                else:
                    _reverse_cases = [R[seq_id1 + 1, seq_id2], R[seq_id1, seq_id2 - 1], delta + F[seq_id1 + 1, seq_id2 - 1]]
                    rev_min_idx = np.argmin(_reverse_cases)
                    R[seq_id1, seq_id2] = _reverse_cases[rev_min_idx]
                    if rev_min_idx == 0:
                        R_label[(seq_id1, seq_id2)] = R_label[(seq_id1 + 1, seq_id2)]
                    elif rev_min_idx == 1:
                        R_label[(seq_id1, seq_id2)] = R_label[(seq_id1, seq_id2 - 1)]
                    else:
                        R_label[(seq_id1, seq_id2)] = [(seq_id1, seq_id2)] + F_label[(seq_id1 + 1, seq_id2 - 1)]
        min_index = np.unravel_index(np.argmin(F, axis=None), F.shape)
        labels = F_label[min_index]
        if not labels:
            return None, None, None 
        else:
            min_delta = F[min_index]
            improved_path = path
            for label in reversed(labels):
                first, second = label 
                improved_path = solution.reverse_within_path(first + 1, second - 1, path=improved_path)
            return improved_path, min_delta, labels  


class TwoKOptOperatorV2(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: solution.MultiODSolutionV2, path_id: int = 0, min_delta=-EPSILON):
        path: types.Path = solution.paths[path_id]
        info = solution.info
        n = len(path) - 1
        F, R = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))
        F_label, R_label = defaultdict(list), defaultdict(list)
        starts_from = 1
        for seq_id1 in range(n - 1, starts_from, -1):
            node1 = path[seq_id1]
            next1 = path[seq_id1 + 1]
            for seq_id2 in range(seq_id1 + 1, n):
                node2 = path[seq_id2]
                prev2 = path[seq_id2 - 1]
                before = (
                    info.distance_matrix[node1, next1]
                    + info.distance_matrix[prev2, node2])
                after = (
                    info.distance_matrix[prev2, node1]
                    + info.distance_matrix[node2, next1])
                delta = after - before
                if seq_id2 <= seq_id1 + 1:
                    F[seq_id1, seq_id2] = 0
                else:
                    _cases = [F[seq_id1 + 1, seq_id2], F[seq_id1, seq_id2 - 1], delta + R[seq_id1 + 1, seq_id2 - 1]]
                    min_idx = np.argmin(_cases)
                    F[seq_id1, seq_id2] = _cases[min_idx]
                    if min_idx == 0:
                        F_label[(seq_id1, seq_id2)] = F_label[(seq_id1 + 1, seq_id2)]
                    elif min_idx == 1:
                        F_label[(seq_id1, seq_id2)] = F_label[(seq_id1, seq_id2 - 1)]
                    else:
                        F_label[(seq_id1, seq_id2)] = [(seq_id1, seq_id2)] + R_label[(seq_id1 + 1, seq_id2 - 1)]

                if ((info.od_type[node1] == 0 and seq_id1 + 1 <= info.sequence[info.od_pairing[node1]].index <= seq_id2)
                    or
                    (info.od_type[node2] == 1 and seq_id1 <= info.sequence[info.od_pairing[node2]].index <= seq_id2 - 1)):
                    R[seq_id1, seq_id2] = np.inf
                elif seq_id1 == seq_id2:
                    R[seq_id1, seq_id2] = 0
                elif seq_id2 == seq_id1 + 1 and not (is_od := info.od_type[node1] == 0 and info.od_pairing[node1] == node2):
                    R[seq_id1, seq_id2] = 0
                elif seq_id2 == seq_id1 + 1 and is_od:
                    R[seq_id1, seq_id2] = np.inf 
                else:
                    _reverse_cases = [R[seq_id1 + 1, seq_id2], R[seq_id1, seq_id2 - 1], delta + F[seq_id1 + 1, seq_id2 - 1]]
                    rev_min_idx = np.argmin(_reverse_cases)
                    R[seq_id1, seq_id2] = _reverse_cases[rev_min_idx]
                    if rev_min_idx == 0:
                        R_label[(seq_id1, seq_id2)] = R_label[(seq_id1 + 1, seq_id2)]
                    elif rev_min_idx == 1:
                        R_label[(seq_id1, seq_id2)] = R_label[(seq_id1, seq_id2 - 1)]
                    else:
                        R_label[(seq_id1, seq_id2)] = [(seq_id1, seq_id2)] + F_label[(seq_id1 + 1, seq_id2 - 1)]
        min_index = np.unravel_index(np.argmin(F, axis=None), F.shape)
        labels = F_label[min_index]
        if not labels:
            return None, None, None 
        else:
            min_delta = F[min_index]
            improved_path = path
            for label in reversed(labels):
                first, second = label 
                improved_path[first + 1: second] = improved_path[first + 1: second][::-1]
            for idx, n in enumerate(improved_path):
                info.sequence[n] = types.SequenceInfo(sequence=path_id, index=idx)
            return solution, min_delta, labels                                   


class ExchangeOperator(Operator):
    def __init__(self, include_taxi_node: bool = True):
        super().__init__(operator_type='path')
        self.include_taxi_node = include_taxi_node
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        label = None
        _start = self._start
        for first in range(_start, n - 1):
            node1: Node = path.get_by_seq_id(first)
            od1 = node1.OD_type
            if od1 == 0:
                d: Node = path.get_by_node_id(path.OD_mapping[node1.node_id])
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, first+1, d.seq_id, min_delta)
            else:
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, first+1, n, min_delta)
            if inner_min_delta < min_delta:
                min_delta = inner_min_delta
                label = inner_label
        if label is None:
            return None, None, None
        else:
            improved_path = solution.exchange_nodes_within_path(label[0], label[1], path=path) 
            return improved_path, min_delta, label  
    
    def _inner_loop(self, first: int, node1: Node, path: MultiODPath, start, end, min_delta=-EPSILON):
        label = None
        for second in range(start, end):
            node2: Node = path.get_by_seq_id(second)
            od2 = node2.OD_type
            if od2 == 1 and (o2 := path.get_by_node_id(path.DO_mapping[node2.node_id])).seq_id > node1.seq_id:
                continue  
            prev1 = node1.prev_node.node_id if node1.prev_node is not None else 0
            next1 = node1.next_node.node_id
            prev2 = node2.prev_node.node_id
            next2 = node2.next_node.node_id if node2.next_node is not None else 0
            if second == first + 1:
                before = (
                    path.get_distance_by_node_ids(prev1, node1.node_id)
                    + path.get_distance_by_node_ids(node2.node_id, next2)
                    )
                after = (
                    path.get_distance_by_node_ids(prev1, node2.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next2)
                )
            else:
                before = (
                    path.get_distance_by_node_ids(prev1, node1.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next1)
                    + path.get_distance_by_node_ids(prev2, node2.node_id)
                    + path.get_distance_by_node_ids(node2.node_id, next2)
                    )
                after = (
                    path.get_distance_by_node_ids(prev1, node2.node_id)
                    + path.get_distance_by_node_ids(node2.node_id, next1)
                    + path.get_distance_by_node_ids(prev2, node1.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next2)
                )
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = node1.node_id, node2.node_id
        return min_delta, label 


class ExchangeOperatorV2(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: solution.MultiODSolutionV2, path_id: int = 0, min_delta=-EPSILON):
        path: types.Path = solution.paths[path_id]
        info = solution.info
        n = len(path) - 1
        label = None
        starts_from = 1
        for first in range(starts_from, n - 1):
            node1 = path[first]
            od1 = info.od_type[node1]
            if od1 == 0:
                d = info.od_pairing[node1]
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, info, first+1, info.sequence[d].index, min_delta)
            else:
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, info, first+1, n, min_delta)
            if inner_min_delta < min_delta:
                min_delta = inner_min_delta
                label = inner_label
        if label is None:
            return None, None, None
        else:
            improved_path = path
            n1, n2 = improved_path[label[0]], improved_path[label[1]]
            improved_path[label[0]], improved_path[label[1]] = n2, n1
            info.sequence[n1] = types.SequenceInfo(sequence=path_id, index=label[1])
            info.sequence[n2] = types.SequenceInfo(sequence=path_id, index=label[0])
            return solution, min_delta, label  
    
    def _inner_loop(self, first: int, node1: int, path: types.Path, info: types.ProblemInfo, start, end, min_delta=-EPSILON):
        label = None
        for second in range(start, end):
            node2 = path[second]
            od2 = info.od_type[node2]
            if od2 == 1 and info.sequence[info.od_pairing[node2]].index > info.sequence[node1].index:
                continue  
            prev1 = path[info.sequence[node1].index - 1]
            next1 = path[info.sequence[node1].index + 1]
            prev2 = path[info.sequence[node2].index - 1]
            next2 = path[info.sequence[node2].index + 1]
            if second == first + 1:
                before = (
                    info.distance_matrix[prev1, node1]
                    + info.distance_matrix[node2, next2])
                after = (
                    info.distance_matrix[prev1, node2]
                    + info.distance_matrix[node1, next2]) 
            else:
                before = (
                    info.distance_matrix[prev1, node1]
                    + info.distance_matrix[node1, next1]
                    + info.distance_matrix[prev2, node2]
                    + info.distance_matrix[node2, next2])
                after = (
                    info.distance_matrix[prev1, node2]
                    + info.distance_matrix[node2, next1]
                    + info.distance_matrix[prev2, node1]
                    + info.distance_matrix[node1, next2])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = node1, node2
        return min_delta, label 
    

class InsertOperator(Operator):
    def __init__(self, include_taxi_node: bool = True):
        super().__init__(operator_type='path')
        self.include_taxi_node = include_taxi_node 
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1

    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        label = None
        _start = self._start
        for first in range(_start, n):
            node1: Node = path.get_by_seq_id(first)
            od1 = node1.OD_type
            if od1 == 0:
                d: Node = path.get_by_node_id(path.OD_mapping[node1.node_id])
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, _start, d.seq_id, min_delta)
            else:
                o: Node = path.get_by_node_id(path.DO_mapping[node1.node_id])
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, o.seq_id + 1, n, min_delta)
            if inner_min_delta < min_delta:
                min_delta = inner_min_delta
                label = inner_label
        if label is None:
            return None, None, None
        else:
            improved_path = solution.insert_within_path(label[0], label[1], path=path)
            return improved_path, min_delta, label 
    
    def _inner_loop(self, first: int, node1: Node, path: MultiODPath, start, end, min_delta=-EPSILON):
        label, delta = None, 0.
        for second in range(start, end):
            node2: Node = path.get_by_seq_id(second)
            prev1 = node1.prev_node.node_id if node1.prev_node is not None else 0
            next1 = node1.next_node.node_id if node1.next_node is not None else 0
            prev2 = node2.prev_node.node_id if node2.prev_node is not None else 0
            next2 = node2.next_node.node_id if node2.next_node is not None else 0
            if first < second:
                before = (
                    path.get_distance_by_node_ids(prev1, node1.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next1)
                    + path.get_distance_by_node_ids(node2.node_id, next2)
                )
                after = (
                    path.get_distance_by_node_ids(prev1, next1)
                    + path.get_distance_by_node_ids(node1.node_id, node2.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next2)
                )
                delta = after - before
            elif first > second:
                before = (
                    path.get_distance_by_node_ids(prev1, node1.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, next1)
                    + path.get_distance_by_node_ids(prev2, node2.node_id)
                )
                after = (
                    path.get_distance_by_node_ids(prev1, next1)
                    + path.get_distance_by_node_ids(node1.node_id, node2.node_id)
                    + path.get_distance_by_node_ids(node1.node_id, prev2)
                )
                delta = after - before
            if delta < min_delta:
                min_delta = delta 
                label = node1.node_id, second
        return min_delta, label 


class InsertOperatorV2(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: solution.MultiODSolutionV2, path_id: int = 0, min_delta=-EPSILON):
        path: types.Path = solution.paths[path_id]
        info = solution.info
        n = len(path) - 1
        label = None
        starts_from = 1
        for first in range(starts_from, n):
            node1 = path[first]
            od1 = info.od_type[node1]
            if od1 == 0:
                d = info.od_pairing[node1]
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, info, starts_from, info.sequence[d].index, min_delta)
            else:
                o = info.od_pairing[node1]
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, info, info.sequence[o].index + 1, n, min_delta)
            if inner_min_delta < min_delta:
                min_delta = inner_min_delta
                label = inner_label
        if label is None:
            return None, None, None
        else:
            improved_path = path 
            node = improved_path.pop(label[0])
            improved_path.insert(label[1], node)
            for seq_idx, n in improved_path:
                info.sequence[n] = types.SequenceInfo(sequence=path_id, index=seq_idx)
            return solution, min_delta, label 
    
    def _inner_loop(self, first: int, node1: int, path: types.Path, info: types.ProblemInfo, start, end, min_delta=-EPSILON):
        label, delta = None, 0.
        for second in range(start, end):
            node2 = path[second]
            prev1 = path[info.sequence[node1].index - 1]
            next1 = path[info.sequence[node1].index + 1] 
            prev2 = path[info.sequence[node2].index - 1]
            next2 = path[info.sequence[node2].index + 1] 
            if first < second:
                before = (
                    info.distance_matrix[prev1, node1]
                    + info.distance_matrix[node1, next1]
                    + info.distance_matrix[node2, next2])
                after = (
                    info.distance_matrix[prev1, next1]
                    + info.distance_matrix[node1, node2]
                    + info.distance_matrix[node1, next2])
                delta = after - before
            elif first > second:
                before = (
                    info.distance_matrix[prev1, node1]
                    + info.distance_matrix[node1, next1]
                    + info.distance_matrix[prev2, node2])
                after = (
                    info.distance_matrix[prev1, next1]
                    + info.distance_matrix[node1, node2]
                    + info.distance_matrix[node1, prev2])
                delta = after - before
            if delta < min_delta:
                min_delta = delta 
                label = info.sequence[node1].index, second
        return min_delta, label         
            

class OForwardOperator(Operator):
    def __init__(self, length: int = 1, include_taxi_node: bool = True):
        super().__init__(operator_type='path')
        self.length = length
        self.include_taxi_node = include_taxi_node 
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        label = None
        _start = self._start
        for O_block_id in path.O_blocks:
            for in_block_seq_id in range(len(path.block_dict[O_block_id]) - self.length + 1):
                node1, node1_tail = path.block_dict[O_block_id][in_block_seq_id], path.block_dict[O_block_id][in_block_seq_id + self.length - 1]
                inner_min_delta, inner_label = self._inner_loop(node1, node1_tail, path, _start, node1.seq_id, min_delta)
                if inner_min_delta < min_delta:
                    min_delta = inner_min_delta
                    label = inner_label
        if label is None:
            return None, None, None
        else:
            target_seq_id = label[1]
            for seq_id in range(label[0], label[0] + self.length):
                node_id = path.get_by_seq_id(seq_id).node_id
                improved_path = solution.insert_within_path(node_id, target_seq_id, path=path)  # Path modified in-place
                target_seq_id += 1
            return improved_path, min_delta, label 
    
    def _inner_loop(self, node1: Node, node1_tail: Node, path: MultiODPath, start: int, end: int, min_delta=-EPSILON):
        label, delta = None, 0.
        for second in range(start, end):
            node2: Node = path.get_by_seq_id(second)
            prev1 = node1.prev_node.node_id if node1.prev_node is not None else 0
            next1_tail = node1_tail.next_node.node_id if node1_tail.next_node is not None else 0
            prev2 = node2.prev_node.node_id if node2.prev_node is not None else 0
            before = (
                path.get_distance_by_node_ids(prev1, node1.node_id)
                + path.get_distance_by_node_ids(node1_tail.node_id, next1_tail)
                + path.get_distance_by_node_ids(prev2, node2.node_id)
            )
            after = (
                path.get_distance_by_node_ids(prev1, next1_tail)
                + path.get_distance_by_node_ids(node1_tail.node_id, node2.node_id)
                + path.get_distance_by_node_ids(node1.node_id, prev2)
            )
            delta = after - before
            if delta < min_delta:
                min_delta = delta 
                label = node1.seq_id, second
        return min_delta, label


class DBackwardOperator(Operator):
    def __init__(self, length: int = 1):
        super().__init__(operator_type='path')
        self.length = length 
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        label = None
        for D_block_id in path.D_blocks:
            for in_block_seq_id in range(len(path.block_dict[D_block_id]) - self.length + 1):
                node1, node1_tail = path.block_dict[D_block_id][in_block_seq_id], path.block_dict[D_block_id][in_block_seq_id + self.length - 1]
                inner_min_delta, inner_label = self._inner_loop(node1, node1_tail, path, node1_tail.seq_id + 1, n, min_delta)
                if inner_min_delta < min_delta:
                    min_delta = inner_min_delta
                    label = inner_label
        if label is None:
            return None, None, None
        else:            
            target_seq_id = label[1]
            for seq_id in range(label[0] + self.length - 1, label[0] - 1, -1):
                node_id = path.get_by_seq_id(seq_id).node_id
                improved_path = solution.insert_within_path(node_id, target_seq_id, path=path)
                target_seq_id -= 1
            return improved_path, min_delta, label 
    
    def _inner_loop(self, node1: Node, node1_tail: Node, path: MultiODPath, start: int, end: int, min_delta=-EPSILON):
        label, delta = None, 0.
        for second in range(start, end):
            node2: Node = path.get_by_seq_id(second)
            prev1 = node1.prev_node.node_id
            next1_tail = node1_tail.next_node.node_id if node1_tail.next_node is not None else 0
            next2 = node2.next_node.node_id if node2.next_node is not None else 0
            before = (
                path.get_distance_by_node_ids(prev1, node1.node_id)
                + path.get_distance_by_node_ids(node1_tail.node_id, next1_tail)
                + path.get_distance_by_node_ids(node2.node_id, next2)
            )
            after = (
                path.get_distance_by_node_ids(prev1, next1_tail)
                + path.get_distance_by_node_ids(node1.node_id, node2.node_id)
                + path.get_distance_by_node_ids(node1_tail.node_id, next2)
            )
            delta = after - before
            if delta < min_delta:
                min_delta = delta 
                label = node1.seq_id, second
        return min_delta, label


class RandomOForwardOperator(Operator):
    def __init__(self, change_percentage: Union[int, float] = 0.1, include_taxi_node: bool = True):
        super().__init__(operator_type='path-random')
        self.change = change_percentage 
        self.include_taxi_node = include_taxi_node 
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1

    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        O_list = list(path.OD_mapping.keys())
        np.random.shuffle(O_list)
        if type(self.change) == float:
            num_Os = max(int(len(O_list)*self.change), 1)
        elif type(self.change) == int:
            num_Os = max(self.change, 1)
        selected = O_list[:num_Os]

        delta = 0.
        _start = self._start
        for node_id in selected:
            node = path.get_by_node_id(node_id)
            d = path.get_by_node_id(path.OD_mapping[node.node_id])
            target_seq_id = random.randint(_start, d.seq_id - 1)
            inner_delta = self.compute_delta(node, target_seq_id, path)
            delta += inner_delta
            improved_path = solution.insert_within_path(node_id, target_seq_id, path=path) # Path modified in-place
        
        return improved_path, delta, True

    def compute_delta(self, node1: Node, target_seq_id: int, path: MultiODPath):
        node2: Node = path.get_by_seq_id(target_seq_id)
        prev1 = node1.prev_node.node_id if node1.prev_node is not None else 0
        next1 = node1.next_node.node_id if node1.next_node is not None else 0
        prev2 = node2.prev_node.node_id if node2.prev_node is not None else 0
        before = (
            path.get_distance_by_node_ids(prev1, node1.node_id)
            + path.get_distance_by_node_ids(node1.node_id, next1)
            + path.get_distance_by_node_ids(prev2, node2.node_id)
        )
        after = (
            path.get_distance_by_node_ids(prev1, next1)
            + path.get_distance_by_node_ids(node1.node_id, node2.node_id)
            + path.get_distance_by_node_ids(node1.node_id, prev2)
        )
        delta = after - before
        return delta


class RandomDBackwardOperator(Operator):
    def __init__(self, change_percentage: Union[int, float] = 0.1):
        super().__init__(operator_type='path-random')
        self.change = change_percentage
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        D_list = list(path.DO_mapping.keys())
        np.random.shuffle(D_list)
        if type(self.change) is float:
            num_Ds = max(int(len(D_list)*self.change), 1)
        else:
            num_Ds = max(self.change, 1)
        selected = D_list[:num_Ds]

        delta = 0.
        for node_id in selected:
            node = path.get_by_node_id(node_id)
            o = path.get_by_node_id(path.DO_mapping[node.node_id])
            target_seq_id = random.randint(o.seq_id + 1, n - 1)
            inner_delta = self.compute_delta(node, target_seq_id, path)
            delta += inner_delta
            improved_path = solution.insert_within_path(node_id, target_seq_id, path=path) # Path modified in-place
        
        return improved_path, delta, True
    
    def compute_delta(self, node1: Node, target_seq_id: int, path: MultiODPath):
        node2: Node = path.get_by_seq_id(target_seq_id)
        prev1 = node1.prev_node.node_id
        next1 = node1.next_node.node_id if node1.next_node is not None else 0
        next2 = node2.next_node.node_id if node2.next_node is not None else 0
        before = (
            path.get_distance_by_node_ids(prev1, node1.node_id)
            + path.get_distance_by_node_ids(node1.node_id, next1)
            + path.get_distance_by_node_ids(node2.node_id, next2)
        )
        after = (
            path.get_distance_by_node_ids(prev1, next1)
            + path.get_distance_by_node_ids(node1.node_id, node2.node_id)
            + path.get_distance_by_node_ids(node1.node_id, next2)
        )
        delta = after - before
        return delta


class ODPairsExchangeOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='path')

    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        label = None
        O_list = list(path.OD_mapping.keys())
        for i in range(len(O_list)):
            O1_id = O_list[i]
            for j in range(i+1, len(O_list)):
                O2_id = O_list[j]
                inner_min_delta, inner_label = self.compute_delta(path.get_by_node_id(O1_id), path.get_by_node_id(O2_id), path)
                if inner_min_delta < min_delta:
                    min_delta = inner_min_delta
                    label = inner_label
        if label is None:
            return None, None, None
        else:
            improved_path = solution.exchange_nodes_within_path(label[0], label[1], path_id, path)
            improved_path = solution.exchange_nodes_within_path(path.OD_mapping[label[0]], path.OD_mapping[label[1]], path_id, improved_path)
            return improved_path, min_delta, label

    def compute_delta(self, O1: Node, O2: Node, path: MultiODPath):
        delta, label = _compute_delta_pair_exchange(O1, O2, path)
        return delta, label


class RandomODPairsExchangeOperator(Operator):
    """"""
    def __init__(self, change_percentage: Union[int, float] = 0.1):
        super().__init__(operator_type='path-random')
        self.change = change_percentage

    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        O_list = list(path.OD_mapping.keys())
        np.random.shuffle(O_list)
        if type(self.change) is float:
            num_Os = max(int(len(O_list)*self.change), 2)
        else:
            num_Os = max(self.change, 2)
        if num_Os % 2 != 0:
            num_Os -= 1
        selected = O_list[:num_Os]
        if len(selected) < 2:
            return None, 0., False
        delta = 0.
        # pairwise exchange
        for i in range(0, len(selected), 2):
            node1_id, node2_id = selected[i], selected[i + 1]
            inner_delta, _ = _compute_delta_pair_exchange(path.get_by_node_id(node1_id), path.get_by_node_id(node2_id), path)
            delta += inner_delta
            improved_path = solution.exchange_nodes_within_path(node1_id, node2_id, path_id, path) # Path modified in-place
            improved_path = solution.exchange_nodes_within_path(path.OD_mapping[node1_id], path.OD_mapping[node2_id], path_id, improved_path)
        return improved_path, delta, True
    

class SameBlockExchangeOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='in-block')
    
    def __call__(self, solution: MultiODSolution, block_id: int, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        block = path.get_by_block_id(block_id)
        n = len(block)
        if n < 1: return None, None, None 
        label = None
        head, tail = block[0], block[-1]
        pb = head.prev_node.node_id if head.prev_node is not None else 0
        nb = tail.next_node.node_id if tail.next_node is not None else 0
        for first in range(1, n):
            node1 = block[first]
            prev1 = node1.prev_node.node_id
            before = (
                path.get_distance_by_node_ids(pb, head.node_id)
                + path.get_distance_by_node_ids(node1.node_id, prev1)
                + path.get_distance_by_node_ids(tail.node_id, nb)
                )
            after = (
                path.get_distance_by_node_ids(pb, node1.node_id)
                + path.get_distance_by_node_ids(head.node_id, tail.node_id)
                + path.get_distance_by_node_ids(nb, prev1)
            )
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = [first]
        if label is None:
            return None, None, None
        else:
            improved_path = path
            first = label[0]
            improved_path = solution.exhange_sequence_within_block(first, block_id, path=improved_path)
            return improved_path, min_delta, label


class MixedBlockExchangeOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        label = None
        # print(path.block_dict)
        for O_id in path.O_blocks:
            for D_id in path.D_blocks:
                O, D = path.block_dict[O_id], path.block_dict[D_id]
                if len(O) < 1 or len(D) < 1 or not O[0].seq_id > D[0].seq_id: continue
                dp, dn = D[0].prev_node.node_id, D[-1].next_node.node_id
                op, on = O[0].prev_node.node_id, O[-1].next_node.node_id
                if dn == O[0].node_id:
                    before = (
                        path.get_distance_by_node_ids(dp, D[0].node_id)
                        + path.get_distance_by_node_ids(D[-1].node_id, O[0].node_id)
                        + path.get_distance_by_node_ids(O[-1].node_id, on)
                    )
                    after = (
                        path.get_distance_by_node_ids(dp, O[0].node_id)
                        + path.get_distance_by_node_ids(O[-1].node_id, D[0].node_id)
                        + path.get_distance_by_node_ids(D[-1].node_id, on)
                    )
                else:
                    before = (
                        path.get_distance_by_node_ids(dp, D[0].node_id)
                        + path.get_distance_by_node_ids(dn, D[-1].node_id)
                        + path.get_distance_by_node_ids(op, O[0].node_id)
                        + path.get_distance_by_node_ids(O[-1].node_id, on)
                    )
                    after = (
                        path.get_distance_by_node_ids(dp, O[0].node_id)
                        + path.get_distance_by_node_ids(dn, O[-1].node_id)
                        + path.get_distance_by_node_ids(op, D[0].node_id)
                        + path.get_distance_by_node_ids(D[-1].node_id, on)
                    )
                inner_min_delta = after - before 
                if inner_min_delta < min_delta:
                    min_delta = inner_min_delta
                    label = [O_id, D_id]
        if label is None:
            return None, None, None
        else:
            block_id1, block_id2 = label
            improved_path = solution.exchange_blocks_within_path(block_id1, block_id2, path=path)
            return improved_path, min_delta, label
        

class RandomSameBlockExchangeOperator(Operator):
    def __init__(self, change_percentage: Union[int, float] = 0.1):
        super().__init__(operator_type='path-random')
        self.change = change_percentage
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        O_list = copy.deepcopy(path.O_blocks)
        np.random.shuffle(O_list)
        if type(self.change) == float:
            num_Os = max(int(len(O_list)*self.change), 1)
        elif type(self.change) == int:
            num_Os = max(self.change, 1)
        selected = O_list[:num_Os]

        improved_path = path
        delta = 0.
        for block_id in selected:
            block = path.block_dict[block_id]
            head, tail = block[0], block[-1]
            pb = head.prev_node.node_id if head.prev_node is not None else 0
            nb = tail.next_node.node_id if tail.next_node is not None else 0
            break_in_block_seq_id = random.randint(0, len(block) - 1)
            node1 = block[break_in_block_seq_id]
            prev1 = node1.prev_node.node_id
            before = (
                path.get_distance_by_node_ids(pb, head.node_id)
                + path.get_distance_by_node_ids(node1.node_id, prev1)
                + path.get_distance_by_node_ids(tail.node_id, nb)
                )
            after = (
                path.get_distance_by_node_ids(pb, node1.node_id)
                + path.get_distance_by_node_ids(head.node_id, tail.node_id)
                + path.get_distance_by_node_ids(nb, prev1)
            )
            inner_delta = after - before
            delta += inner_delta
            improved_path = solution.exhange_sequence_within_block(break_in_block_seq_id, block_id, path=improved_path)
        
        return improved_path, delta, True
    

class RandomMixedBlockExchangeOperator(Operator):
    def __init__(self, change_percentage: Union[int, float] = 0.1):
        super().__init__(operator_type='path-random')
        self.change = change_percentage
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        O_list = copy.deepcopy(path.O_blocks)
        np.random.shuffle(O_list)
        if type(self.change) == float:
            num_Os = max(int(len(O_list)*self.change), 1)
        elif type(self.change) == int:
            num_Os = max(self.change, 1)
        selected = O_list[:num_Os]
        
        improved_path = path
        delta = 0.
        for block_id in selected:
            # in-loop modify the path, therefore the block_id could be removed from path because of merge.
            if block_id not in path.block_dict: continue 
            block = path.block_dict[block_id]
            head, tail = block[0], block[-1]
            pb = head.prev_node.node_id if head.prev_node is not None else 0
            nb = tail.next_node.node_id if tail.next_node is not None else 0
            feasible_Ds = [D_id for D_id in path.D_blocks if path.block_dict[D_id][0].seq_id < block[0].seq_id]
            if len(feasible_Ds) < 1: continue
            O = block
            D_id = random.sample(feasible_Ds, k=1)[0]
            D = path.block_dict[D_id]
            dp, dn = D[0].prev_node.node_id, D[-1].next_node.node_id
            op, on = O[0].prev_node.node_id, O[-1].next_node.node_id
            if dn == O[0].node_id:
                before = (
                    path.get_distance_by_node_ids(dp, D[0].node_id)
                    + path.get_distance_by_node_ids(D[-1].node_id, O[0].node_id)
                    + path.get_distance_by_node_ids(O[-1].node_id, on)
                )
                after = (
                    path.get_distance_by_node_ids(dp, O[0].node_id)
                    + path.get_distance_by_node_ids(O[-1].node_id, D[0].node_id)
                    + path.get_distance_by_node_ids(D[-1].node_id, on)
                )
            else:
                before = (
                    path.get_distance_by_node_ids(dp, D[0].node_id)
                    + path.get_distance_by_node_ids(dn, D[-1].node_id)
                    + path.get_distance_by_node_ids(op, O[0].node_id)
                    + path.get_distance_by_node_ids(O[-1].node_id, on)
                )
                after = (
                    path.get_distance_by_node_ids(dp, O[0].node_id)
                    + path.get_distance_by_node_ids(dn, O[-1].node_id)
                    + path.get_distance_by_node_ids(op, D[0].node_id)
                    + path.get_distance_by_node_ids(D[-1].node_id, on)
                )
            inner_delta = after - before
            delta += inner_delta
            improved_path = solution.exchange_blocks_within_path(block_id, D_id, path=improved_path)
        
        return improved_path, delta, True
    

class ODPairsExchangeMultiVehicles(Operator):
    def __init__(self):
        super().__init__(operator_type='multi-paths')
    
    def __call__(self, solution: MultiODSolution, path_id1: int = 0, path_id2: int = 0, min_delta=-EPSILON):
        path1: MultiODPath = solution.paths[path_id1]
        path2: MultiODPath = solution.paths[path_id2]
        cumcap1 = np.sum(path1.capacities)
        cumcap2 = np.sum(path2.capacities)
        O1s = path1.OD_mapping.keys()
        O2s = path2.OD_mapping.keys()
        
        label = None
        for o1_id in O1s:
            for o2_id in O2s:
                o1: Node = path1.get_by_node_id(o1_id)
                d1: Node = path1.get_by_node_id(path1.OD_mapping[o1_id])
                o2: Node = path2.get_by_node_id(o2_id)
                d2: Node = path2.get_by_node_id(path2.OD_mapping[o2_id])
                # Check capacity
                _item_cap1, _item_cap2 = path1.capacities[o1.seq_id] + path1.capacities[d1.seq_id], path2.capacities[o2.seq_id] + path2.capacities[d2.seq_id]
                cap_diff = _item_cap1 - _item_cap2
                if cumcap1 - cap_diff <= path1.capacity and cumcap2 + cap_diff <= path2.capacity:
                    delta, inner_label = self.compute_delta(o1, o2, d1, d2, path1, path2)
                    if delta <= min_delta:
                        label = inner_label
                        min_delta = delta  
                else:
                    continue
        
        if label is None:
            return None, None, None 
        else:
            # print(label, path_id1, path_id2)
            o_pair, d_pair = label
            improved_path1, improved_path2 = solution.exchange_od_pair_across_paths(o_pair[0], o_pair[1], d_pair[0], d_pair[1], path1=path1, path2=path2)
            improved_paths = (improved_path1, improved_path2)
            return improved_paths, min_delta, label
    
    def compute_delta(self, o1: Node, o2: Node, d1: Node, d2: Node, path1: MultiODPath, path2: MultiODPath):
        delta, label = _compute_delta_pair_exchange_across_paths(o1, o2, d1, d2, path1, path2)
        return delta, label


class ODPairsInsertMultiVehicles(Operator):
    def __init__(self, include_taxi_node: bool = False):
        super().__init__(operator_type='multi-paths')
        self.include_taxi_node = include_taxi_node 
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1

    def __call__(self, solution: MultiODSolution, path_id1: int = 0, path_id2: int = 0, min_delta=-EPSILON):
        path1: MultiODPath = solution.paths[path_id1]
        path2: MultiODPath = solution.paths[path_id2]
        cumcap2 = np.sum(path2.capacities)
        cap_left = path2.capacity - cumcap2 
        end = len(path2)
        O1s = path1.OD_mapping.keys()
        label = None
        for o1_id in O1s:
            o1: Node = path1.get_by_node_id(o1_id)
            d1: Node = path1.get_by_node_id(path1.OD_mapping[o1_id])
            _item_cap = path1.capacities[o1.seq_id] + path1.capacities[d1.seq_id]
            if _item_cap <= cap_left:
                for t1 in range(self._start, end):  #
                    for t2 in range(t1, end):
                        inner_delta = self._compute_delta(o1, d1, t1, t2, path1, path2)
                        if inner_delta <= min_delta:
                            min_delta = inner_delta
                            label = [o1_id, t1, t2]
        
        if label is None:
            return None, None, None 
        else:
            # print(label, path_id1, path_id2)
            o_id, t1, t2 = label 
            improved_path1, improved_path2 = solution.insert_od_pair_across_paths(o_id, t1, t2, path1=path1, path2=path2)
            improved_paths = (improved_path1, improved_path2)
            return improved_paths, min_delta, label
        
    def _compute_delta(self, o: Node, d: Node, t1: int, t2: int, path1: MultiODPath, path2: MultiODPath):
        delta = _compute_delta_pair_insert_across_paths(o=o, d=d, t1=t1, t2=t2, path1=path1, path2=path2) 
        return delta 


class RandomODPairsInsertMultiVehicles(Operator):
    def __init__(self, change_percentage: Union[int, float] = 0.1, include_taxi_node: bool = False):
        super().__init__(operator_type='path-random')
        self.change = change_percentage
        self.include_taxi_node = include_taxi_node 
        if self.include_taxi_node:
            self._start = 2
        else:
            self._start = 1
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        O_list = list(path.OD_mapping.keys())
        np.random.shuffle(O_list)
        if type(self.change) is float:
            num_Os = max(int(len(O_list)*self.change), 1)
        else:
            num_Os = max(self.change, 1)
        selected = O_list[:num_Os] 
        if not selected:
            return None, 0., False
        to_paths = [i for i in range(len(solution.paths)) if i != path_id]
        to_paths = np.random.choice(to_paths, size=len(selected))

        delta = 0.
        for i in range(len(selected)):
            o_id = selected[i]
            o: Node = path.get_by_node_id(o_id)
            d: Node = path.get_by_node_id(path.OD_mapping[o_id])
            to_path = solution.paths[to_paths[i]]
            end = len(to_path)
            t1 = random.randint(self._start, end)
            t2 = random.randint(t1, end)
            inner_delta = self._compute_delta(o, d, t1, t2, path1=path, path2=to_path)
            delta += inner_delta
            # print(o_id, d.node_id, t1, t2, path_id, to_paths[i])
            improved_path = solution.insert_od_pair_across_paths(o_id, t1, t2, path1=path, path2=to_path)
        return improved_path, delta, True 
    
    def _compute_delta(self, o: Node, d: Node, t1: int, t2: int, path1: MultiODPath, path2: MultiODPath):
        delta = _compute_delta_pair_insert_across_paths(o=o, d=d, t1=t1, t2=t2, path1=path1, path2=path2) 
        return delta 
    

def _compute_delta_pair_exchange(o1: Node, o2: Node, path: MultiODPath):
    label, delta = None, 0.

    o1_id = o1.node_id
    o2_id = o2.node_id
    d1 = path.get_by_node_id(path.OD_mapping[o1_id])
    d2 = path.get_by_node_id(path.OD_mapping[o2_id])
    if o1.seq_id > o2.seq_id:
        o_f, o_s = o2, o1 
    else:
        o_f, o_s = o1, o2 
    if d1.seq_id > d2.seq_id:
        d_f, d_s = d2, d1 
    else:
        d_f, d_s = d1, d2 
    o_f_prev, o_f_next = o_f.prev_node.node_id if o_f.prev_node is not None else 0, o_f.next_node.node_id
    o_s_prev, o_s_next = o_s.prev_node.node_id, o_s.next_node.node_id 
    d_f_prev, d_f_next = d_f.prev_node.node_id, d_f.next_node.node_id
    d_s_prev, d_s_next = d_s.prev_node.node_id, d_s.next_node.node_id if d_s.next_node is not None else 0
    
    o_f_nseq, o_s_nseq, d_f_nseq = o_f.seq_id + 1, o_s.seq_id + 1, d_f.seq_id + 1
    o_fs_is_neighbor, od_sf_is_neighbor, d_sf_is_neighbor = o_f_nseq == o_s.seq_id, o_s_nseq == d_f.seq_id, d_f_nseq == d_s.seq_id
    od_f_is_neighbor, do_fs_is_neighbor, od_s_is_neighbor = o_f_nseq == d_f.seq_id, d_f_nseq == o_s.seq_id, o_s_nseq == d_s.seq_id

    if o_s.seq_id < d_f.seq_id:
        # Of,Os,...,Df,...,Ds
        if o_fs_is_neighbor and not od_sf_is_neighbor and not d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            )
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )
        
        # Of,Os,...,Df,Ds
        elif o_fs_is_neighbor and not od_sf_is_neighbor and d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            )
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,Os,Df,Ds
        elif o_fs_is_neighbor and od_sf_is_neighbor and d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,Os,Df,...,Ds
        elif o_fs_is_neighbor and od_sf_is_neighbor and not d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,...,Os,...,Df,...,Ds
        elif not o_fs_is_neighbor and not od_sf_is_neighbor and not d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,...,Os,...,Df,Ds
        elif not o_fs_is_neighbor and not od_sf_is_neighbor and d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            ) 

        # Of,...,Os,Df,Ds
        elif not o_fs_is_neighbor and od_sf_is_neighbor and d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,...,Os,Df,...,Ds
        elif not o_fs_is_neighbor and od_sf_is_neighbor and not d_sf_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            ) 
    else:
        # Of,Df,Os,Ds
        if od_f_is_neighbor and do_fs_is_neighbor and od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,...,Df,Os,Ds
        elif not od_f_is_neighbor and do_fs_is_neighbor and od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_f.node_id, o_s.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_f.node_id, o_f.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,...,Df,...,Os,Ds
        elif not od_f_is_neighbor and not do_fs_is_neighbor and od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )  

        # Of,...,Df,...,Os,...,Ds
        elif not od_f_is_neighbor and not do_fs_is_neighbor and not od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,Df,...,Os,Ds
        elif od_f_is_neighbor and not do_fs_is_neighbor and od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            )
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,Df,...,Os,...,Ds
        elif od_f_is_neighbor and not do_fs_is_neighbor and not od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_f.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            )
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(d_f_next, d_s.node_id)
                + path.get_distance_by_node_ids(o_s_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,Df,Os,...,Ds
        elif od_f_is_neighbor and do_fs_is_neighbor and not od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            )

        # Of,...,Df,Os,...,Ds
        elif not od_f_is_neighbor and do_fs_is_neighbor and not od_s_is_neighbor:
            before = (
                path.get_distance_by_node_ids(o_f_prev, o_f.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_f.node_id)
                + path.get_distance_by_node_ids(o_s.node_id, d_f.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_s.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_s.node_id)
            ) 
            after = (
                path.get_distance_by_node_ids(o_f_prev, o_s.node_id)
                + path.get_distance_by_node_ids(o_f_next, o_s.node_id)
                + path.get_distance_by_node_ids(d_f_prev, d_s.node_id)
                + path.get_distance_by_node_ids(o_f.node_id, d_s.node_id)
                + path.get_distance_by_node_ids(o_s_next, o_f.node_id)
                + path.get_distance_by_node_ids(d_s_prev, d_f.node_id)
                + path.get_distance_by_node_ids(d_s_next, d_f.node_id)
            ) 

    delta = after - before
    label = o1.node_id, o2.node_id
    return delta, label


def _compute_delta_pair_exchange_across_paths(o1: Node, o2: Node, d1: Node, d2: Node, path1: MultiODPath, path2: MultiODPath):
    label, delta = None, 0.
    prev_o1, next_o1 = o1.prev_node.node_id if o1.prev_node is not None else 0, o1.next_node.node_id
    prev_d1, next_d1 = d1.prev_node.node_id, d1.next_node.node_id if d1.next_node is not None else 0
    prev_o2, next_o2 = o2.prev_node.node_id if o2.prev_node is not None else 0, o2.next_node.node_id
    prev_d2, next_d2 = d2.prev_node.node_id, d2.next_node.node_id if d2.next_node is not None else 0

    od1_is_neighbor, od2_is_neighbor = o1.seq_id + 1 == d1.seq_id, o2.seq_id + 1 == d2.seq_id
    # o1,d1; o2,d2
    if od1_is_neighbor and od2_is_neighbor:
        before = (
            path1.get_distance_by_node_ids(prev_o1, o1.node_id)
            + path1.get_distance_by_node_ids(d1.node_id, next_d1)
            + path2.get_distance_by_node_ids(prev_o2, o2.node_id)
            + path2.get_distance_by_node_ids(d2.node_id, next_d2)
            )
        after = (
            path1.get_distance_by_node_ids(prev_o1, o2.node_id)
            + path1.get_distance_by_node_ids(d2.node_id, next_d1)
            + path2.get_distance_by_node_ids(prev_o2, o1.node_id)
            + path2.get_distance_by_node_ids(d1.node_id, next_d2)
        )
    # o1,d1; o2,...,d2
    elif od1_is_neighbor and not od2_is_neighbor:
        before = (
            path1.get_distance_by_node_ids(prev_o1, o1.node_id)
            + path1.get_distance_by_node_ids(o1.node_id, d1.node_id)
            + path1.get_distance_by_node_ids(d1.node_id, next_d1)
            + path2.get_distance_by_node_ids(prev_o2, o2.node_id)
            + path2.get_distance_by_node_ids(o2.node_id, next_o2)
            + path2.get_distance_by_node_ids(prev_d2, d2.node_id)
            + path2.get_distance_by_node_ids(d2.node_id, next_d2)
        )
        after = (
            path1.get_distance_by_node_ids(prev_o1, o2.node_id)
            + path1.get_distance_by_node_ids(o2.node_id, d2.node_id)
            + path1.get_distance_by_node_ids(d2.node_id, next_d1)
            + path2.get_distance_by_node_ids(prev_o2, o1.node_id)
            + path2.get_distance_by_node_ids(next_o2, o1.node_id)
            + path2.get_distance_by_node_ids(prev_d2, d1.node_id)
            + path2.get_distance_by_node_ids(d1.node_id, next_d2)
        )
    # o1,...,d1; o2,d2
    elif not od1_is_neighbor and od2_is_neighbor:
        before = (
            path1.get_distance_by_node_ids(prev_o1, o1.node_id)
            + path1.get_distance_by_node_ids(o1.node_id, next_o1)
            + path1.get_distance_by_node_ids(prev_d1, d1.node_id)
            + path1.get_distance_by_node_ids(next_d1, d1.node_id)
            + path2.get_distance_by_node_ids(prev_o2, o2.node_id)
            + path2.get_distance_by_node_ids(o2.node_id, d2.node_id)
            +path2.get_distance_by_node_ids(d2.node_id, next_d2)
        )
        after = (
            path1.get_distance_by_node_ids(prev_o1, o2.node_id)
            + path1.get_distance_by_node_ids(o2.node_id, next_o1)
            + path1.get_distance_by_node_ids(prev_d1, d2.node_id)
            + path1.get_distance_by_node_ids(next_d1, d2.node_id)
            + path2.get_distance_by_node_ids(prev_o2, o1.node_id)
            + path2.get_distance_by_node_ids(o1.node_id, d1.node_id)
            + path2.get_distance_by_node_ids(d1.node_id, next_d2)
        )
    # o1,...,d1; o2,...,d2 
    elif not od1_is_neighbor and not od2_is_neighbor:
        before = (
            path1.get_distance_by_node_ids(prev_o1, o1.node_id)
            + path1.get_distance_by_node_ids(o1.node_id, next_o1)
            + path1.get_distance_by_node_ids(prev_d1, d1.node_id)
            + path1.get_distance_by_node_ids(next_d1, d1.node_id)
            + path2.get_distance_by_node_ids(prev_o2, o2.node_id)
            + path2.get_distance_by_node_ids(o2.node_id, next_o2)
            + path2.get_distance_by_node_ids(prev_d2, d2.node_id)
            + path2.get_distance_by_node_ids(d2.node_id, next_d2)
        )
        after = (
            path1.get_distance_by_node_ids(prev_o1, o2.node_id)
            + path1.get_distance_by_node_ids(o2.node_id, next_o1)
            + path1.get_distance_by_node_ids(prev_d1, d2.node_id)
            + path1.get_distance_by_node_ids(next_d1, d2.node_id)
            + path2.get_distance_by_node_ids(prev_o2, o1.node_id)
            + path2.get_distance_by_node_ids(o1.node_id, next_o2)
            + path2.get_distance_by_node_ids(prev_d2, d1.node_id)
            + path2.get_distance_by_node_ids(d1.node_id, next_d2)
        )
    
    delta = after - before
    label = [(o1.node_id, o2.node_id), (d1.node_id, d2.node_id)]
    return delta, label


def _compute_delta_pair_insert_across_paths(o: Node, d: Node, t1: int, t2: int, path1: MultiODPath, path2: MultiODPath):
    prev_o, next_o = o.prev_node.node_id if o.prev_node is not None else 0, o.next_node.node_id
    prev_d, next_d = d.prev_node.node_id, d.next_node.node_id if d.next_node is not None else 0
    od_is_neighbor = o.seq_id + 1 == d.seq_id

    # o,...,d; o,d
    if not od_is_neighbor:
        before1 = (
            path1.get_distance_by_node_ids(prev_o, o.node_id)
            + path1.get_distance_by_node_ids(o.node_id, next_o)
            + path1.get_distance_by_node_ids(prev_d, d.node_id)
            + path1.get_distance_by_node_ids(d.node_id, next_d)
        )
        after1 = (
            path1.get_distance_by_node_ids(prev_o, next_o)
            + path1.get_distance_by_node_ids(prev_d, next_d)
        )
    else:
        before1 = (
            path1.get_distance_by_node_ids(prev_o, o.node_id)
            + path1.get_distance_by_node_ids(o.node_id, d.node_id)
            + path1.get_distance_by_node_ids(d.node_id, next_d)
        )
        after1 = (
            path1.get_distance_by_node_ids(prev_o, next_d)
        )
    # t1,t2/t1,...,t2; t1=t2; t1/t2>=len(path2)-1
    if t1 < len(path2) - 1 and t2 < len(path2) - 1 and t1 != t2:
        tn1: Node = path2.get_by_seq_id(t1)
        tn2: Node = path2.get_by_seq_id(t2)
        prev_t1 = tn1.prev_node.node_id if tn1.prev_node is not None else 0
        prev_t2 = tn2.prev_node.node_id if tn2.prev_node is not None else 0
        before2 = (
            path2.get_distance_by_node_ids(tn1.node_id, prev_t1)
            + path2.get_distance_by_node_ids(tn2.node_id, prev_t2)
        )
        after2 = (
            path2.get_distance_by_node_ids(prev_t1, o.node_id)
            + path2.get_distance_by_node_ids(o.node_id, tn1.node_id)
            + path2.get_distance_by_node_ids(prev_t2, d.node_id)
            + path2.get_distance_by_node_ids(d.node_id, tn2.node_id)
        )
    elif t1 < len(path2) - 1 and t2 < len(path2) - 1 and t1 == t2:
        tn1: Node = path2.get_by_seq_id(t1)
        prev_t1 = tn1.prev_node.node_id if tn1.prev_node is not None else 0
        before2 = (
            path2.get_distance_by_node_ids(prev_t1, tn1.node_id)
        ) 
        after2 = (
            path2.get_distance_by_node_ids(prev_t1, o.node_id)
            + path2.get_distance_by_node_ids(o.node_id, d.node_id)
            + path2.get_distance_by_node_ids(d.node_id, tn1.node_id)
        )
    elif t1 < len(path2) - 1 and t2 >= len(path2) - 1:
        tn1: Node = path2.get_by_seq_id(t1)
        prev_t1 = tn1.prev_node.node_id if tn1.prev_node is not None else 0
        tail2: Node = path2.get_by_seq_id(-1)
        tail2 = tail2.node_id if tail2 is not None else 0
        depot = 0
        before2 = (
            path2.get_distance_by_node_ids(tn1.node_id, prev_t1)
            + path2.get_distance_by_node_ids(tail2, depot)
        ) 
        after2 = (
            path2.get_distance_by_node_ids(prev_t1, o.node_id)
            + path2.get_distance_by_node_ids(o.node_id, tn1.node_id)
            + path2.get_distance_by_node_ids(tail2, d.node_id)
            + path2.get_distance_by_node_ids(d.node_id, depot)
        )
    elif t1 >= len(path2) - 1:
        tail2: Node = path2.get_by_seq_id(-1) 
        tail2 = tail2.node_id if tail2 is not None else 0
        depot = 0
        before2 = (
            path2.get_distance_by_node_ids(tail2, depot)
        ) 
        after2 = (
            path2.get_distance_by_node_ids(tail2, o.node_id)
            + path2.get_distance_by_node_ids(o.node_id, d.node_id)
            + path2.get_distance_by_node_ids(d.node_id, depot)
        )

    delta = (after1 + after2) - (before1 + before2) 
    return delta  