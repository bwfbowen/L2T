import random
from collections import deque 
from itertools import islice


class SliceableDeque(deque):
    r"""A class implemented slice for collections.deque
    
    Reference: https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
    """
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(islice(self, index.start,
                                               index.stop, index.step))
        return deque.__getitem__(self, index)
    

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
    

def random_split_dict(d, num_splits):
    """Randomly split the dict into `num_splits` parts."""
    keys = list(d.keys())
    random.shuffle(keys)
    split_points = [0] + sorted(random.sample(range(1, len(keys)), num_splits-1)) + [None]
    
    return [dict((k, d[k]) for k in keys[split_points[i]:split_points[i+1]]) for i in range(num_splits)]


def generate_random_list_from_dict_with_key_before_value(d, shuffled_keys=None):
    """Generates random list from a dict and respects the rule that key appears before value."""
    if shuffled_keys is None:
        keys = list(d.keys())  
        random.shuffle(keys)
    else:
        keys = list(shuffled_keys) if not isinstance(shuffled_keys, list) else shuffled_keys

    result = []
    for key in keys:
        value = d[key]

        # Insert the key at a random index
        key_index = random.randint(0, len(result))
        result.insert(key_index, key)

        # Insert the value at a random index greater than the key's index
        value_index = random.randint(key_index + 1, len(result))
        result.insert(value_index, value)
    
    return result

