from typing import Union 
import random
import copy
from collections import defaultdict
import numpy as np

from . import solution
from . import utils 

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
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1

        segment_starts, temp_Ds = SliceableDeque([2]), set([path.OD_mapping[path.get_by_seq_id(2).node_id]])
        for seq_id in range(3, n):
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
                prev1 = node1.prev_node.node_id 
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
        

class TwoKOptOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        F, R = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))
        F_label, R_label = defaultdict(list), defaultdict(list)
        for seq_id1 in range(n - 1, 2, -1):
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


class ExchangeOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        label = None
        for first in range(2, n - 1):
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
            prev1 = node1.prev_node.node_id
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


class InsertOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='path')
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        n = len(path) - 1
        label = None
        for first in range(2, n):
            node1: Node = path.get_by_seq_id(first)
            od1 = node1.OD_type
            if od1 == 0:
                d: Node = path.get_by_node_id(path.OD_mapping[node1.node_id])
                inner_min_delta, inner_label = self._inner_loop(first, node1, path, 2, d.seq_id, min_delta)
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
            prev1 = node1.prev_node.node_id
            next1 = node1.next_node.node_id if node1.next_node is not None else 0
            prev2 = node2.prev_node.node_id
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
            

class OForwardOperator(Operator):
    def __init__(self, length: int = 1):
        super().__init__(operator_type='path')
        self.length = length
    
    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        label = None
        for O_block_id in path.O_blocks:
            for in_block_seq_id in range(len(path.block_dict[O_block_id]) - self.length + 1):
                node1, node1_tail = path.block_dict[O_block_id][in_block_seq_id], path.block_dict[O_block_id][in_block_seq_id + self.length - 1]
                inner_min_delta, inner_label = self._inner_loop(node1, node1_tail, path, 2, node1.seq_id, min_delta)
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
            prev1 = node1.prev_node.node_id
            next1_tail = node1_tail.next_node.node_id if node1_tail.next_node is not None else 0
            prev2 = node2.prev_node.node_id
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
    def __init__(self, change_percentage: Union[int, float] = 0.1):
        super().__init__(operator_type='path-random')
        self.change = change_percentage 

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
        for node_id in selected:
            node = path.get_by_node_id(node_id)
            d = path.get_by_node_id(path.OD_mapping[node.node_id])
            target_seq_id = random.randint(2, d.seq_id - 1)
            inner_delta = self.compute_delta(node, target_seq_id, path)
            delta += inner_delta
            improved_path = solution.insert_within_path(node_id, target_seq_id, path=path) # Path modified in-place
        
        return improved_path, delta, True

    def compute_delta(self, node1: Node, target_seq_id: int, path: MultiODPath):
        node2: Node = path.get_by_seq_id(target_seq_id)
        prev1 = node1.prev_node.node_id
        next1 = node1.next_node.node_id if node1.next_node is not None else 0
        prev2 = node2.prev_node.node_id
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
        
        delta = 0.
        # pairwise exchange
        for i in range(0, len(selected), 2):
            node1_id, node2_id = selected[i], selected[i + 1]
            inner_delta, _ = _compute_delta_pair_exchange(path.get_by_node_id(node1_id), path.get_by_node_id(node2_id), path)
            delta += inner_delta
            improved_path = solution.exchange_nodes_within_path(node1_id, node2_id, path_id, path) # Path modified in-place
            improved_path = solution.exchange_nodes_within_path(path.OD_mapping[node1_id], path.OD_mapping[node2_id], path_id, improved_path)
        return improved_path, delta, True


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
    o_f_prev, o_f_next = o_f.prev_node.node_id, o_f.next_node.node_id
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