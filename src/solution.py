from . import problem
from . import utils
    

SliceableDeque = utils.SliceableDeque
IndexDict = utils.IndexDict 


class Solution:
    r"""A solution class whose elements are Paths."""
    def __iter__(self):
        raise NotImplementedError
    
    def __getitem__(self, item):
        raise NotImplementedError


class Path(IndexDict):
    r"""A Path class whose elements are integer indexes that represent nodes accompanied by the index in the path."""
    def __iter__(self):
        raise NotImplementedError 
    

class MultiODSolution(Solution):
    
    _is_valid: bool = True 

    def __init__(self, paths: list):
        self.paths = self._validate_list_and_create_paths(paths)
    
    def set_is_valid(self, value, caller):
        if not isinstance(caller, problem.Problem):
            raise ValueError("Only Problem instances can change is_valid")
        self._is_valid = value
    
    def _create_path(self, path):
        """Creates `Path` instance from `list` """
        _path = MultiODPath()
        for node in path:
            _path.insert(node)
        return _path 
    
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
    def __init__(self):
        super().__init__()
        self.dummy_first, self.dummy_second = None, None 
        self._num_dummy = 0
    
    def insert(self, node):
        if node == 0:
            if self.dummy_first is None:
                self.dummy_first = self.len
                self._num_dummy += 1
            else:
                self.dummy_second = self.len
                self._num_dummy += 1
        else:
            if node not in self.dict:
                self.dict[node] = self.len
    
    def indexof(self, node):
        if node == 0:
            return [self.dummy_first, self.dummy_second]
        else:
            return super().indexof(node)

    def __contains__(self, node):
        if node == 0:
            return self.dummy_first is not None and self.dummy_second is not None
        return super().__contains__(node)
        
    def __iter__(self):
        i = 0
        nodes = iter(self.dict)
        for i in range(self.len):
            if i == self.dummy_first or i == self.dummy_second:
                yield 0
            else:
                yield next(nodes) 
    
    def __repr__(self):
        return f'{type(self).__name__}({[*iter(self)]})'
        
    @property
    def len(self):
        return len(self.dict) + self._num_dummy
    
