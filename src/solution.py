from dataclasses import dataclass

from . import problem
from . import utils
    

SliceableDeque = utils.SliceableDeque


class Solution:
    r"""A solution class whose elements are Paths."""
    def __iter__(self):
        raise NotImplementedError
    
    def __getitem__(self, item):
        raise NotImplementedError
        

class Path:
    r"""A Path class."""

    def __init__(self, node_ids):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError 
    
    def __contains__(self, item):
        return item in self.seq_dict
    
    def insert(self, node_id):
        raise NotImplementedError
    
    def indexof(self, node_id: int):
        raise NotImplementedError
    
    def get_by_seq_id(self, seq_id: int):
        raise NotImplementedError


@dataclass
class Node:
    node_id: int
    OD_type: int = None
    seq_id: int = None
    block_id: int = None 
    in_block_seq_id: int = None 
    location=None
    block_OD: int = None
    prev_node=None
    next_node=None


class MultiODPath(Path):
    """
    
    Attributes
    ------
    block_dict: dict, start from 0, keys are integer.
    """
    def __init__(self, node_ids: list, OD_types: dict = None, locations=None, distance_matrix=None):
        self.node_dict: dict = {}
        self.seq_dict: dict = {}
        self.block_dict: dict = {}
        self.O_blocks: list = []
        self.D_blocks: list = []

        self.OD_types = OD_types
        self.locations = locations
        self.distance_matrix = distance_matrix
        self.dummy_first, self.dummy_second = None, None 
        
        self._num_dummy = 0
        self._block_id = -1
        self._prev_node = None

        for node_id in node_ids:
            self.insert(node_id)
    
    def insert(self, node_id):
        if node_id == 0:
            if self.dummy_first is None:
                self.dummy_first = self.len
                self._num_dummy += 1
            else:
                self.dummy_second = self.len
                self._num_dummy += 1
        else:
            if node_id not in self.node_dict:
                node = self.node_dict[node_id] = Node(node_id)
                seq_id = self.len
                self.seq_dict[seq_id] = node 
                node.seq_id = seq_id
            if self.OD_types is not None:
                self._assign_OD_attrs_for_single_node(node)
            if self.locations is not None: 
                node.location = self.locations[node_id]
            
            node.prev_node = self._prev_node
            if self._prev_node is not None:
                self._prev_node.next_node = node 
            self._prev_node = node
    
    def indexof(self, node_id):
        if node_id == 0:
            return [self.dummy_first, self.dummy_second]
        else:
            return self.node_dict[node_id].seq_id
    
    def get_by_seq_id(self, seq_id: int):
        if seq_id == 0: return 0
        return self.seq_dict[seq_id]
    
    def get_by_block_id(self, block_id: int, in_block_seq_id: int = None):
        if in_block_seq_id is None:
            return self.block_dict[block_id]
        else:
            return self.block_dict[block_id][in_block_seq_id]
    
    def get_by_node_id(self, node_id: int):
        if node_id == 0: return 0
        return self.node_dict[node_id]
    
    def assign_OD_attrs(self, OD_types: dict):
        self.OD_types = OD_types
        if self.seq_dict:
            self._block_id = -1
            self._prev_node = None 
            for seq_idx, _ in enumerate(self.seq_dict, start=1):
                node = self.seq_dict[seq_idx]
                self._assign_OD_attrs_for_single_node(node)
                self._prev_node = node 
        return self.node_dict

    def assign_location_attr(self, locations):
        for node_id in self.node_dict:
            node = self.node_dict[node_id]
            node.location = locations[node_id]
        return self.node_dict
    
    def assign_distance_matrix_attr(self, distance_matrix):
        self.distance_matrix = distance_matrix
        return self.distance_matrix
            
    def _assign_OD_attrs_for_single_node(self, node):
        node_id = node.node_id
        if node_id not in self.OD_types: return 
        node.OD_type = self.OD_types[node_id]
        # The condition is:
        # 1. if no previous node
        # 2. or if the previous node is not O/D
        # 3. or the OD type of previous node != OD type of node
        if self._prev_node is None or self._prev_node.node_id not in self.OD_types or self.OD_types[node_id] != self.OD_types[self._prev_node.node_id]:
            self._block_id += 1
            self.block_dict[self._block_id] = SliceableDeque() 
            if self.OD_types[node_id] == 0: self.O_blocks.append(self._block_id)
            elif self.OD_types[node_id] == 1: self.D_blocks.append(self._block_id)
        block_id = self._block_id
        block = self.block_dict[block_id]
        node.in_block_seq_id = len(block)
        block.append(node)
        node.block_OD = self.OD_types[node_id]
        node.block_id = self._block_id

    def __contains__(self, node):
        if node == 0:
            return self.dummy_first is not None and self.dummy_second is not None
        return super().__contains__(node)
        
    def __iter__(self):
        i = 0
        seq_ids = iter(self.seq_dict)
        for i in range(self.len):
            if i == self.dummy_first or i == self.dummy_second:
                yield 0
            else:
                yield self.seq_dict[next(seq_ids)].node_id
    
    def __repr__(self):
        return f'{type(self).__name__}({[*iter(self)]})'
        
    @property
    def len(self):
        return len(self.seq_dict) + self._num_dummy
    
    def __len__(self):
        return self.len         


class MultiODSolution(Solution):

    def __init__(self, paths: list, problem=None):
        self._is_valid: bool = True 
        self.paths = self._validate_list_and_create_paths(paths, problem)
    
    def set_is_valid(self, value, caller):
        if not isinstance(caller, problem.Problem):
            raise ValueError("Only Problem instances can change is_valid")
        self._is_valid = value

    def exchange_nodes_within_path(self, node_id1, node_id2, path_id: int = 0, path: MultiODPath = None):
        """path_id is the index of path in the self.paths. If path is not None, arg path will be used."""
        if path is None:
            path = self.paths[path_id]
        node1, node2 = path.get_by_node_id(node_id1), path.get_by_node_id(node_id2)
        # Ensure seq_id1 < seq_id2
        if node1.seq_id > node2.seq_id:
            node1, node2 = node2, node1 

        # Swap seq_id and update seq_dict
        self._swap_seq_id_within_path(path, node1, node2)

        # Swap prev_node and next_node references
        self._swap_prev_next_within_path(node1, node2)
        
        # Swap in_block_seq_id and update block_dict
        # If nodes have the same OD type
        if node1.OD_type == node2.OD_type:
            node1.block_id, node2.block_id = node2.block_id, node1.block_id
            node1.in_block_seq_id, node2.in_block_seq_id = node2.in_block_seq_id, node1.in_block_seq_id
            path.block_dict[node1.block_id][node1.in_block_seq_id], path.block_dict[node2.block_id][node2.in_block_seq_id] = node1, node2
        else:
            path.block_dict[node1.block_id].remove(node1)
            path.block_dict[node2.block_id].remove(node2)
            self._update_in_block_seq_id(path, node1.block_id)
            self._update_in_block_seq_id(path, node2.block_id)
            if not path.block_dict[node1.block_id]:
                del path.block_dict[node1.block_id]
            if not path.block_dict[node2.block_id]:
                del path.block_dict[node2.block_id]

            # Here swap the in_block_seq_id of node1 and node2 for slicing the block in the `_reassign_block_attrs_after_swap` function.
            node1.in_block_seq_id, node2.in_block_seq_id = node2.in_block_seq_id, node1.in_block_seq_id

            # If the prev and next node both have different OD type compared to the node, 
            # new block is created, original block is split
            # After swap, node2 is ahead of node1
            self._reassign_block_attrs_after_swap_different_OD(path, node2)
            self._reassign_block_attrs_after_swap_different_OD(path, node1)
    
    def _swap_seq_id_within_path(self, path: MultiODPath, node1: Node, node2: Node):
        node1.seq_id, node2.seq_id = node2.seq_id, node1.seq_id
        path.seq_dict[node1.seq_id], path.seq_dict[node2.seq_id] = node1, node2
    
    def _swap_prev_next_within_path(self, node1: Node, node2: Node):
        """Make sure node1.seq_id < node2.seq_id. This function does not have if-statement to check."""
        node1_prev = node1.prev_node
        node1_next = node1.next_node
        node2_prev = node2.prev_node
        node2_next = node2.next_node

        if node1_next == node2: # they are next to each other, node1 is before node2
            node2.prev_node = node1_prev
            node2.next_node = node1 
            node1.prev_node = node2 
            node1.next_node = node2_next
            if node1_prev is not None: node1_prev.next_node = node2 
            if node2_next is not None: node2_next.prev_node = node1 
        else:
            node1.prev_node = node2_prev
            node1.next_node = node2_next
            node2.prev_node = node1_prev
            node2.next_node = node1_next
            if node1_prev is not None: node1_prev.next_node = node2 
            if node1_next is not None: node1_next.prev_node = node2 
            if node2_prev is not None: node2_prev.next_node = node1 
            if node2_next is not None: node2_next.prev_node = node1 
    
    def _reassign_block_attrs_after_swap_different_OD(self, path: MultiODPath, node: Node):
        prev_is_none = node.prev_node is None or node.prev_node.block_id is None  # taxi node
        next_is_none = node.next_node is None or node.next_node.block_id is None 
        prev_is_different = (prev_is_none) or (node.prev_node.OD_type != node.OD_type)
        next_is_different = (next_is_none) or (node.next_node.OD_type != node.OD_type)
        if prev_is_different and next_is_different:
                if not prev_is_none and not next_is_none:
                    original_block = path.block_dict[node.prev_node.block_id]
                    path.block_dict[node.prev_node.block_id] = SliceableDeque(original_block[:node.in_block_seq_id])
                    self._create_new_block_after_swap(path, node)
                    path._block_id += 1
                    path.block_dict[path._block_id] = SliceableDeque(original_block[node.in_block_seq_id:])
                    for idx, node in enumerate(original_block[node.in_block_seq_id:]):
                        node.block_id = path._block_id
                        node.in_block_seq_id = idx 
                else:
                    self._create_new_block_after_swap(path, node)
        elif prev_is_different and not next_is_different:
            # node.next_node has the same OD type as node's if node.next_node is not None
            # insert node at the beginning of node.next_node's block
            node.block_id = node.next_node.block_id
            node.in_block_seq_id = 0
            path.block_dict[node.next_node.block_id].appendleft(node)     
        elif not prev_is_different and next_is_different:
                node.block_id = node.prev_node.block_id
                node.in_block_seq_id = node.prev_node.in_block_seq_id + 1
                path.block_dict[node.prev_node.block_id].append(node)
        else:
            # merge blocks
            merge_block_id = node.prev_node.block_id
            to_merge_block_id = node.next_node.block_id
            
            self._append_node_to_prev_block(path, node, merge_block_id)
            for node in path.block_dict[to_merge_block_id]:
                self._append_node_to_prev_block(path, node, merge_block_id)
            del path.block_dict[to_merge_block_id]
    
    def _update_in_block_seq_id(self, path: MultiODPath, block_id: int):
        block = path.block_dict[block_id]
        for in_seq_id, node in enumerate(block):
            node.in_block_seq_id = in_seq_id

    def _create_new_block_after_swap(self, path: MultiODPath, node: Node):
        path._block_id += 1 
        node.block_id = path._block_id
        node.in_block_seq_id = 0
        path.block_dict[node.block_id] = SliceableDeque([node])
    
    def _append_node_to_prev_block(self, path: MultiODPath, node: Node, merge_block_id: int):
        _in_block_seq_id = len(path.block_dict[merge_block_id])
        node.block_id = merge_block_id
        node.in_block_seq_id = _in_block_seq_id
        path.block_dict[merge_block_id].append(node)
    
    def _create_path(self, path, problem=None):
        """Creates `Path` instance from `list` """
        if not problem is None:
            OD_types = {n: 0 if i < len(problem.O) else 1 for i, n in enumerate(problem.O + problem.D)}
            return MultiODPath(path, OD_types=OD_types, locations=problem.locations, distance_matrix=problem.distance_matrix)
        else:
            return MultiODPath(path)
        
    @staticmethod
    def _validate_paths(paths):
        """To check if the node is unique across all paths and 0 is at the start and the end of all paths."""
        non_zero_nodes = SliceableDeque()
        for path in paths:
            if path[0] != 0 or path[-1] != 0:
                return False 
            non_zero_nodes.extend([node for node in path if node != 0])
        if len(non_zero_nodes) != len(set(non_zero_nodes)):
            return False 
        # If everything is fine
        return True 
    
    def _validate_list_and_create_paths(self, lists, problem=None):
        self._is_valid = self._validate_paths(lists)
        if self._is_valid:
            paths = SliceableDeque()
            for path in lists:
                paths.append(self._create_path(path, problem))
            return paths
        
    def __iter__(self):
        if self._is_valid:
            for path in self.paths:
                yield path

    def __getitem__(self, item):
        return self.paths[item]

    def __repr__(self):
        if self._is_valid:
            return f'{type(self).__name__}({self.paths})'
        else:
            return 'Invalid Solution.'
        


