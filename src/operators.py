import random

from . import solution

EPSILON = 1e-5

Solution = solution.Solution
MultiODSolution = solution.MultiODSolution
MultiODPath = solution.MultiODPath
Node = solution.Node


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
            while first < second:
                node_id1, node_id2 = path.get_by_seq_id(first).node_id, path.get_by_seq_id(second).node_id
                solution.exchange_nodes_within_path(node_id1, node_id2, path=improved_path) 
                first = first + 1
                second = second - 1
            return improved_path, min_delta, label
  

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
    def __init__(self):
        super().__init__(operator_type='path')


class RandomDBackwardOperator(Operator):
    def __init__(self):
        super().__init__(operator_type='path')

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
                inner_min_delta, inner_label = self.compute_delta(path.get_by_node_id(O1_id), path.get_by_node_id(O2_id), path, min_delta)
                if inner_min_delta < min_delta:
                    min_delta = inner_min_delta
                    label = inner_label
        if label is None:
            return None, None, None
        else:
            improved_path = solution.exchange_nodes_within_path(label[0], label[1], path_id, path)
            improved_path = solution.exchange_nodes_within_path(path.OD_mapping[label[0]], path.OD_mapping[label[1]], path_id, improved_path)
            return improved_path, min_delta, label

    def compute_delta(self, O1: Node, O2: Node, path: MultiODPath, min_delta=-EPSILON):
        label, delta = None, 0.
        O1_id = O1.node_id
        O2_id = O2.node_id
        D1 = path.get_by_node_id(path.OD_mapping[O1_id])
        D2 = path.get_by_node_id(path.OD_mapping[O2_id])
        next1 = D1.next_node.node_id if D1.next_node is not None else 0
        next2 = D2.next_node.node_id if D2.next_node is not None else 0

        before = (
            path.get_distance_by_node_ids(O1.prev_node.node_id, O1.node_id)
            + path.get_distance_by_node_ids(O1.node_id, O1.next_node.node_id)
            + path.get_distance_by_node_ids(O2.prev_node.node_id, O2.node_id)
            + path.get_distance_by_node_ids(O2.node_id, O2.next_node.node_id)
            + path.get_distance_by_node_ids(D1.prev_node.node_id, D1.node_id)
            + path.get_distance_by_node_ids(D1.node_id, next1)
            + path.get_distance_by_node_ids(D2.prev_node.node_id, D2.node_id)
            + path.get_distance_by_node_ids(D2.node_id, next2)
        )

        after = (
            path.get_distance_by_node_ids(O1.prev_node.node_id, O2.node_id)
            + path.get_distance_by_node_ids(O2.node_id, O1.next_node.node_id)
            + path.get_distance_by_node_ids(O2.prev_node.node_id, O1.node_id)
            + path.get_distance_by_node_ids(O1.node_id, O2.next_node.node_id)
            + path.get_distance_by_node_ids(D1.prev_node.node_id, D2.node_id)
            + path.get_distance_by_node_ids(D2.node_id, next1)
            + path.get_distance_by_node_ids(D2.prev_node.node_id, D1.node_id)
            + path.get_distance_by_node_ids(D1.node_id, next2)
        )

        delta = after - before
        if delta < min_delta:
            min_delta = delta
            label = O1.node_id, O2.node_id
        return min_delta, label

class RandomODPairsExchangeOperator(Operator):

    def __init__(self, change_percentage: float):
        super().__init__(operator_type='path')
        self.change = change_percentage

    def __call__(self, solution: MultiODSolution, path_id: int = 0, min_delta=-EPSILON):
        path: MultiODPath = solution.paths[path_id]
        label = None
        O_list = list(path.OD_mapping.keys())
        num_Os = int(len(path)*self.change/2)
        picked_pairs = set()

        for _ in range(num_Os):
            while True:
                random_elements = random.sample(O_list, 2)
                pair = tuple(sorted(random_elements))
                if pair not in picked_pairs:
                    picked_pairs.add(pair)
                    break
        return

