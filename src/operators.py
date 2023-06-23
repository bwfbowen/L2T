
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
   