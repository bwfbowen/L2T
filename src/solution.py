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


class IndexDict:
    def __init__(self):
        self.dict = {}

    def __contains__(self, item):
        return item in self.dict
    
    def __getitem__(self, item):
        return self.dict[item]

    def insert(self, item):
        if item not in self.dict:
            self.dict[item] = len(self.dict)

    def indexof(self, item):
        return self.dict.get(item, -1)

    def replace(self, old_item, new_item):
        if old_item in self.dict:
            index = self.dict[old_item]
            del self.dict[old_item]
            self.dict[new_item] = index
        else:
            raise ValueError(f"Item {old_item} not found in dictionary")
        

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


class MultiODSolution(Solution):

    def __init__(self, paths: list):
        self._is_valid: bool = True 
        self.paths = self._validate_list_and_create_paths(paths)
    
    def set_is_valid(self, value, caller):
        if not isinstance(caller, problem.Problem):
            raise ValueError("Only Problem instances can change is_valid")
        self._is_valid = value
    
    def _create_path(self, path):
        """Creates `Path` instance from `list` """
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
    
    def _validate_list_and_create_paths(self, lists):
        self._is_valid = self._validate_paths(lists)
        if self._is_valid:
            paths = SliceableDeque()
            for path in lists:
                paths.append(self._create_path(path))
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
        

class MultiODPath(Path):
    """
    
    Attributes
    ------
    block_dict: dict, start from 0, keys are integer, even numbers are O blocks, odd numbers are D blocks.
    """
    def __init__(self, node_ids: list, OD_types: dict = None, locations=None):
        self.node_dict: dict = {}
        self.seq_dict: dict = {}
        self.block_dict: dict = {}
        self.OD_types = OD_types
        self.locations = locations
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
            self.block_dict[self._block_id] = []
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
    
