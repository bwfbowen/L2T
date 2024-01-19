from typing import List, Dict 
from dataclasses import dataclass

from src import types
from src import utils
    

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
    def __init__(self, node_ids: list, OD_types: dict = None, locations=None, distance_matrix=None, OD_mapping=None):
        self.node_dict: dict = {}
        self.seq_dict: dict = {}
        self.block_dict: dict = {}
        self.O_blocks: list = []
        self.D_blocks: list = []

        self.OD_types = OD_types
        self.locations = locations
        self.distance_matrix = distance_matrix
        self.OD_mapping = OD_mapping
        self.DO_mapping = {D: O for O, D in self.OD_mapping.items()}
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
    
    def get_distance_by_node_ids(self, node_id1: int, node_id2: int):
        return self.distance_matrix[node_id1, node_id2]
    
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
    
    def assign_OD_mapping(self, OD_mapping: dict):
        self.OD_mapping = OD_mapping
        self.DO_mapping = {D: O for O, D in self.OD_mapping.items()}
        return self.OD_mapping
            
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

    def _append_id_to_OD_blocks(self, block_id: int, OD_type: int):
        if OD_type == 0: self.O_blocks.append(block_id)
        elif OD_type == 1: self.D_blocks.append(block_id)
    
    def _remove_id_from_OD_blocks(self, block_id: int, OD_type: int):
        if OD_type == 0: self.O_blocks.remove(block_id)
        elif OD_type == 1: self.D_blocks.remove(block_id)

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
        from src import problem
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
            self._remove_and_update_path_block_attrs_after_swap_different_OD(path, node1)
            self._remove_and_update_path_block_attrs_after_swap_different_OD(path, node2)

            # Here swap the in_block_seq_id of node1 and node2 for slicing the block in the `_reassign_block_attrs_after_swap` function.
            node1.in_block_seq_id, node2.in_block_seq_id = node2.in_block_seq_id, node1.in_block_seq_id

            # If the prev and next node both have different OD type compared to the node, 
            # new block is created, original block is split
            # After swap, node2 is ahead of node1
            self._reassign_block_attrs_after_swap_different_OD(path, node2)
            self._reassign_block_attrs_after_swap_different_OD(path, node1)
        
        return path
    
    def insert_within_path(self, node_id: int, target_seq_id: int, path_id: int = 0, path: MultiODPath = None):
        """path_id is the index of path in the self.paths. If path is not None, arg path will be used."""
        if path is None:
            path = self.paths[path_id]
        node: Node = path.get_by_node_id(node_id)
        # store original seq_id
        origin_seq_id = node.seq_id
        if origin_seq_id != target_seq_id:
            self._reassign_block_attrs_before_insert(path, node)
            self._insert_prev_next_within_path(node, target_seq_id, path)
            self._update_seq_attr_insert(node, target_seq_id, path)

            # Handle the block attributes
            self._remove_and_update_path_block_attrs_after_insert(path, node)
            self._reassign_block_attrs_after_insert(path, node)

        return path 

    def reverse_within_path(self, seq_id1: int, seq_id2: int, path_id: int = 0, path: MultiODPath = None):
        if path is None:
            path = self.paths[path_id]
        first, second = seq_id1, seq_id2 
        while first < second:
            node_id1, node_id2 = path.get_by_seq_id(first).node_id, path.get_by_seq_id(second).node_id
            self.exchange_nodes_within_path(node_id1, node_id2, path=path) 
            first = first + 1
            second = second - 1
        return path 
    
    def exchange_blocks_within_path(self, block_id1: int, block_id2: int, path_id: int = 0, path: MultiODPath = None):
        if path is None:
            path = self.paths[path_id]
        block1, block2 = path.block_dict[block_id1], path.block_dict[block_id2]
        if block1[0].seq_id > block2[0].seq_id:
            block1, block2 = block2, block1 
        h1, t1, h2, t2 = block1[0], block1[-1], block2[0], block2[-1]
        hp1, tn1, hp2, tn2 = h1.prev_node, t1.next_node, h2.prev_node, t2.next_node
        
        # sequence id
        # node attributes seq_id, path attributes seq_dict
        if tn1 != h2:
            hp1.next_node, tn1.prev_node = h2, t2 
            hp2.next_node = h1
            if tn2 is not None:
                 tn2.prev_node = t1
            h2.prev_node, t2.next_node = hp1, tn1 
            h1.prev_node, t1.next_node = hp2, tn2 
        else:
            hp1.next_node = h2
            if tn2 is not None:
                tn2.prev_node = t1 
            h2.prev_node, t1.next_node = hp1, tn2 
            t2.next_node, h1.prev_node = h1, t2 

        _node: Node = hp1.next_node
        for _seq_id in range(hp1.seq_id + 1, tn2.seq_id):
            _node.seq_id = _seq_id
            path.seq_dict[_seq_id] = _node 
            _node = _node.next_node
        
        # block merge
        # node attributes block_id, in_block_seq_id, path attributes block_dict

        if h2.prev_node.OD_type == h2.OD_type:
            merge_block_id, to_merge_block_id = h2.prev_node.block_id, h2.block_id
            self._merge_into_prev_block(merge_block_id, to_merge_block_id, path=path)
        if t2.OD_type == t2.next_node.OD_type:
            merge_block_id, to_merge_block_id = t2.block_id, t2.next_node.block_id
            self._merge_into_prev_block(merge_block_id, to_merge_block_id, path=path)
        if h1.prev_node.OD_type == h1.OD_type:
            merge_block_id, to_merge_block_id = h1.prev_node.block_id, h1.block_id
            self._merge_into_prev_block(merge_block_id, to_merge_block_id, path=path)
        if t1.OD_type == t1.next_node.OD_type:
            merge_block_id, to_merge_block_id = t1.block_id, t1.next_node.block_id
            self._merge_into_prev_block(merge_block_id, to_merge_block_id, path=path)
        
        return path 
    
    def exhange_sequence_within_block(self, break_in_seq_id: int, block_id: int, path_id: int = 0, path: MultiODPath = None):
        if path is None:
            path = self.paths[path_id]
        block = path.block_dict[block_id]
        # sequence id
        # node attributes seq_id, path attributes seq_dict
        half1, half2 = block[:break_in_seq_id], block[break_in_seq_id:]
        if len(half1) == 0 or len(half2) == 0: return 
        h1, t1, h2, t2 = half1[0], half1[-1], half2[0], half2[-1]
        hp1, tn2 = h1.prev_node, t2.next_node
        hp1.next_node = h2
        if tn2 is not None:
            tn2.prev_node = t1
        h2.prev_node, t1.next_node = hp1, tn2 
        t2.next_node, h1.prev_node = h1, t2 
        new_block = half2 + half1
        path.block_dict[block_id] = new_block
        _prev = new_block[0].prev_node
        _prev_seq_id = _prev.seq_id
        for _in_block_seq_id, node in enumerate(path.block_dict[block_id]):
            node.in_block_seq_id = _in_block_seq_id
            _prev_seq_id += 1
            node.seq_id = _prev_seq_id
            path.seq_dict[_prev_seq_id] = node
        
        return path

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
    
    def _insert_prev_next_within_path(self, node: Node, target_seq_id: int, path: MultiODPath):
        origin_seq_id = node.seq_id
        if node.prev_node is not None: node.prev_node.next_node = node.next_node
        if node.next_node is not None: node.next_node.prev_node = node.prev_node

        current_target_node = path.get_by_seq_id(target_seq_id)
        if origin_seq_id < target_seq_id:
            if target_seq_id == len(path.seq_dict):
                target_node_next = None 
            else:
                target_node_next = path.get_by_seq_id(target_seq_id + 1)
                target_node_next.prev_node = node 
            current_target_node.next_node = node
            node.prev_node = current_target_node
            node.next_node = target_node_next
        elif origin_seq_id > target_seq_id:
            target_node_prev = path.get_by_seq_id(target_seq_id - 1)
            target_node_prev.next_node = node 
            current_target_node.prev_node = node 
            node.prev_node = target_node_prev
            node.next_node = current_target_node
    
    def _reassign_block_attrs_after_swap_different_OD(self, path: MultiODPath, node: Node):
        prev_is_none = node.prev_node is None or node.prev_node.block_id is None  # taxi node
        next_is_none = node.next_node is None or node.next_node.block_id is None 
        prev_is_different = (prev_is_none) or (node.prev_node.OD_type != node.OD_type)
        next_is_different = (next_is_none) or (node.next_node.OD_type != node.OD_type)
        if prev_is_different and next_is_different:
            self._create_new_block(path, node)
        elif prev_is_different and not next_is_different:
            # node.next_node has the same OD type as node's if node.next_node is not None
            # insert node at the beginning of node.next_node's block 
            self._appendleft_node_to_next_block(path, node, node.next_node.block_id)  
        elif not prev_is_different and next_is_different:
            node.block_id = node.prev_node.block_id
            node.in_block_seq_id = node.prev_node.in_block_seq_id + 1
            path.block_dict[node.prev_node.block_id].append(node)
        else:
            # merge blocks
            merge_block_id = node.prev_node.block_id
            to_merge_block_id = node.next_node.block_id
            path._remove_id_from_OD_blocks(to_merge_block_id, node.next_node.OD_type)
            self._append_node_to_prev_block(path, node, merge_block_id)
            for node in path.block_dict[to_merge_block_id]:
                self._append_node_to_prev_block(path, node, merge_block_id)
            del path.block_dict[to_merge_block_id]
    
    def _reassign_block_attrs_before_insert(self, path: MultiODPath, node: Node):
        prev_next_not_none = not node.prev_node.block_id is None and not node.next_node is None
        prev_next_same_type_diff_from_node = node.prev_node.OD_type != node.OD_type and node.next_node.OD_type != node.OD_type
        if prev_next_not_none and prev_next_same_type_diff_from_node:
            merge_block_id = node.prev_node.block_id
            to_merge_block_id = node.next_node.block_id
            path._remove_id_from_OD_blocks(to_merge_block_id, node.next_node.OD_type)
            for node in path.block_dict[to_merge_block_id]:
                self._append_node_to_prev_block(path, node, merge_block_id)
            del path.block_dict[to_merge_block_id]
    
    def _reassign_block_attrs_after_insert(self, path: MultiODPath, node: Node):
        prev_is_none = node.prev_node is None or node.prev_node.block_id is None  # taxi node
        next_is_none = node.next_node is None or node.next_node.block_id is None 
        prev_is_different = (prev_is_none) or (node.prev_node.OD_type != node.OD_type)
        next_is_different = (next_is_none) or (node.next_node.OD_type != node.OD_type)
        if prev_is_different and next_is_different:
            self._create_new_block(path, node)
            if not prev_is_none and not next_is_none:
                # Divide block
                next_node = node.next_node
                divide_block_id = next_node.block_id
                divide_block = path.block_dict[divide_block_id]
                path.block_dict[divide_block_id] = divide_block[:next_node.in_block_seq_id]
                second_half = divide_block[next_node.in_block_seq_id:]
                self._create_new_block_from_half(path, second_half)
        elif prev_is_different and not next_is_different:
            # node.next_node has the same OD type as node's if node.next_node is not None
            # insert node at the beginning of node.next_node's block 
            self._appendleft_node_to_next_block(path, node, node.next_node.block_id)
        elif not prev_is_different and next_is_different:
            node.block_id = node.prev_node.block_id
            node.in_block_seq_id = node.prev_node.in_block_seq_id + 1
            path.block_dict[node.prev_node.block_id].append(node)
        else:
            # Insert into an existing block of the same type as node
            insert_block_id = node.prev_node.block_id
            insert_in_block_seq_id = node.prev_node.in_block_seq_id + 1
            self._insert_node_to_block(path, node, insert_block_id, insert_in_block_seq_id)
    
    def _update_in_block_seq_id(self, path: MultiODPath, block_id: int):
        block = path.block_dict[block_id]
        for in_seq_id, node in enumerate(block):
            node.in_block_seq_id = in_seq_id
        
    def _update_seq_attr_insert(self, node: Node, target_seq_id: int, path: MultiODPath):
        origin_seq_id = node.seq_id
        if origin_seq_id > target_seq_id:
            # If the node is being moved forward, the nodes in the range (target_seq_id, node.seq_id) 
            # need to be shifted one position to the right
            for i in range(origin_seq_id - 1, target_seq_id - 1, -1):
                shifted_node: Node = path.get_by_seq_id(i)
                shifted_node.seq_id += 1
                path.seq_dict[shifted_node.seq_id] = shifted_node
        elif origin_seq_id < target_seq_id:
            # If the node is being moved backward, the nodes in the range (node.seq_id, target_seq_id) 
            # need to be shifted one position to the left
            for i in range(origin_seq_id + 1, target_seq_id + 1):
                shifted_node = path.get_by_seq_id(i)
                shifted_node.seq_id -= 1
                path.seq_dict[shifted_node.seq_id] = shifted_node
        
        node.seq_id = target_seq_id
        path.seq_dict[target_seq_id] = node 

    def _remove_and_update_path_block_attrs_after_swap_different_OD(self, path: MultiODPath, node: Node):
        original_block = path.block_dict[node.block_id]
        path.block_dict[node.block_id] = original_block[:node.in_block_seq_id]
        self._update_in_block_seq_id(path, node.block_id)
        if not path.block_dict[node.block_id]:
            del path.block_dict[node.block_id]
            path._remove_id_from_OD_blocks(node.block_id, node.OD_type)
        second_half = original_block[node.in_block_seq_id + 1:]
        self._create_new_block_from_half(path, second_half)

    def _remove_and_update_path_block_attrs_after_insert(self, path: MultiODPath, node: Node):
        original_block = path.block_dict[node.block_id]
        path.block_dict[node.block_id] = original_block[:node.in_block_seq_id] + original_block[node.in_block_seq_id + 1:]
        self._update_in_block_seq_id(path, node.block_id)
        if not path.block_dict[node.block_id]:
            del path.block_dict[node.block_id]
            path._remove_id_from_OD_blocks(node.block_id, node.OD_type)

    def _create_new_block(self, path: MultiODPath, node: Node):
        path._block_id += 1 
        node.block_id = path._block_id
        node.in_block_seq_id = 0
        path.block_dict[node.block_id] = SliceableDeque([node])
        path._append_id_to_OD_blocks(node.block_id, node.OD_type)
    
    def _create_new_block_from_half(self, path: MultiODPath, half: SliceableDeque):
        if len(half) > 0:
            path._block_id += 1
            new_block_id = path._block_id
            path.block_dict[new_block_id] = SliceableDeque(half)
            for idx, node in enumerate(half):
                node.block_id = new_block_id
                node.in_block_seq_id = idx 
            path._append_id_to_OD_blocks(new_block_id, node.OD_type)
    
    def _append_node_to_prev_block(self, path: MultiODPath, node: Node, merge_block_id: int):
        _in_block_seq_id = len(path.block_dict[merge_block_id])
        node.block_id = merge_block_id
        node.in_block_seq_id = _in_block_seq_id
        path.block_dict[merge_block_id].append(node)
    
    def _appendleft_node_to_next_block(self, path: MultiODPath, node: Node, merge_block_id: int):
        node.block_id = merge_block_id
        path.block_dict[merge_block_id].appendleft(node)
        for _in_block_seq_id, node in enumerate(path.block_dict[merge_block_id]):
            node.in_block_seq_id = _in_block_seq_id

    def _insert_node_to_block(self, path: MultiODPath, node: Node, insert_block_id: int, insert_in_block_seq_id: int):
        node.block_id = insert_block_id
        path.block_dict[node.block_id].insert(insert_in_block_seq_id, node)
        self._update_in_block_seq_id(path, insert_block_id)
    
    def _merge_into_prev_block(self, merge_block_id: int, to_merge_block_id: int, path: MultiODPath):
        if merge_block_id == to_merge_block_id: return 
        OD_type = path.block_dict[merge_block_id][0].OD_type
        path._remove_id_from_OD_blocks(to_merge_block_id, OD_type)
        for node in path.block_dict[to_merge_block_id]:
            self._append_node_to_prev_block(path, node, merge_block_id)
        del path.block_dict[to_merge_block_id] 
    
    def _create_path(self, path, problem=None):
        """Creates `Path` instance from `list` """
        if not problem is None:
            OD_types = {n: 0 if i < len(problem.O) else 1 for i, n in enumerate(problem.O + problem.D)}
            return MultiODPath(path, OD_types=OD_types, locations=problem.locations, distance_matrix=problem.distance_matrix, OD_mapping=problem.OD_mapping)
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
            return f'Invalid: {type(self).__name__}({self.paths})'
        

class MultiODSolutionV2:
    def __init__(
        self, 
        paths: List[List], 
        problem_info: types.ProblemInfo = None
    ) -> None:
        self._paths = paths 
        self._problem = problem_info._replace(sequence=self._generate_sequence_info(self.paths))

    def _generate_sequence_info(paths: List[List]) -> Dict[int, types.SequenceInfo]:
        sequence_info = {}
        for path_idx, path in enumerate(paths):
            for seq_idx, element in enumerate(path):
                sequence_info[element] = types.SequenceInfo(path_idx, seq_idx)
        return sequence_info

    @property
    def paths(self):
        return self._paths
    
    @property
    def info(self):
        return self._problem
    
    @info.setter
    def info(self, info):
        self._problem = info 
