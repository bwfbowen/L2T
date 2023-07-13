from abc import ABC, abstractmethod
from typing import Any, Tuple
import dataclasses
from dataclasses import dataclass
import jax 
import numpy as np

from .. import SliceableDeque


@dataclass
class MultiODTransition:
    obs: Any = 0.
    delta: float = 0.
    delta_best: float = 0.
    k_recent_action: SliceableDeque = SliceableDeque([])
    k_recent_delta_sign: SliceableDeque = SliceableDeque([])
    a: int = 0
    r: float = 0.
    done: bool = False
    Rn: float = 0. 
    v: float = 0. 
    pi: Any = 0.
    w: float = 1.

    def __iter__(self):
      for field in dataclasses.fields(self):
        yield getattr(self, field.name)

    def __getitem__(self, index):
      return MultiODTransition(*(_attr[index] for _attr in self))

def flatten_transition_func(transition: MultiODTransition) -> Tuple:
  return iter(transition), None 

def unflatten_transition_func(treedef, leaves) -> MultiODTransition:
  return MultiODTransition(*leaves)

jax.tree_util.register_pytree_node(
    MultiODTransition,
    flatten_func=flatten_transition_func,
    unflatten_func=unflatten_transition_func
)  