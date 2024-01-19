from typing import List 
from . import actions
from . import operators


def get_default_candidate_actions_v2() -> List[actions.Action]:
    return []

def get_default_action_dict(env_instance):
    _actions = [ 
               'actions.InBlockAction({idx}, operator=operators.TwoOptOperator())',
               'actions.InBlockAction({idx}, operator=operators.SameBlockExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.SegmentTwoOptOperator())',
               'actions.PathAction({idx}, operator=operators.TwoKOptOperator())',
               'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.InsertOperator())',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=5))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=6))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=7))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=8))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=9))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=5))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=6))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=7))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=8))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=9))',
               'actions.PathAction({idx}, operator=operators.ODPairsExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.MixedBlockExchangeOperator())'
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_naive_action_dict(env_instance):
    _actions = [ 
               'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.InsertOperator())',
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution
    return _action_dict

def get_od_pair_action_dict(env_instance):
    _actions = [
         'actions.PathAction({idx}, operator=operators.ODPairsExchangeOperator())',
         'actions.PathAction({idx}, operator=operators.InsertOperator())'
         ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_intra_nxo_action_dict(env_instance):
    _actions = [
          'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
          'actions.InBlockAction({idx}, operator=operators.TwoOptOperator())',
     ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_inter_nxo_action_dict(env_instance):
    _actions = [
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=1))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=5))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=6))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=7))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=8))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=9))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=1))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=5))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=6))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=7))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=8))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=9))',
    ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_same_bxo_action_dict(env_instance):
    _actions = [
         'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
         'actions.InBlockAction({idx}, operator=operators.SameBlockExchangeOperator())']
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_mix_bxo_action_dict(env_instance):
    _actions = [
         'actions.PathAction({idx}, operator=operators.ExchangeOperator())',
         'actions.PathAction({idx}, operator=operators.MixedBlockExchangeOperator())']
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_feasible_mapping_action_dict(env_instance):
    _actions = [ 
               'actions.InBlockAction({idx}, operator=operators.TwoOptOperator())',
               'actions.InBlockAction({idx}, operator=operators.SameBlockExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.SegmentTwoOptOperator())',
               'actions.PathAction({idx}, operator=operators.TwoKOptOperator())',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=1))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=5))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=1))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=4))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=5))',
               'actions.PathAction({idx}, operator=operators.ODPairsExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.MixedBlockExchangeOperator())'
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_default_random_actions():
       _random_actions = ['actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.1))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.2))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.2))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomMixedBlockExchangeOperator(change_percentage=0.1))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomSameBlockExchangeOperator(change_percentage=0.1))' 
                          ]
       _random_actions = [eval(a.format(idx=idx)) for idx, a in enumerate(_random_actions)]
       return _random_actions

def get_od_pair_random_actions():
     _random_actions = ['actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.2))']
     _random_actions = [eval(a.format(idx=idx)) for idx, a in enumerate(_random_actions)]
     return _random_actions

def get_nxo_random_actions():
     _random_actions = [
          'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.2))',
          'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.2))'
          ]
     _random_actions = [eval(a.format(idx=idx)) for idx, a in enumerate(_random_actions)]
     return _random_actions

def get_bxo_random_actions():
     _random_actions = [
          'actions.PathRandomAction({idx}, operator=operators.RandomMixedBlockExchangeOperator(change_percentage=0.1))',
          'actions.PathRandomAction({idx}, operator=operators.RandomSameBlockExchangeOperator(change_percentage=0.1))' 
     ]
     _random_actions = [eval(a.format(idx=idx)) for idx, a in enumerate(_random_actions)]
     return _random_actions

def get_pdp_default_action_dict(env_instance):
    _actions = [ 
               'actions.InBlockAction({idx}, operator=operators.TwoOptOperator())',
               'actions.InBlockAction({idx}, operator=operators.SameBlockExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.SegmentTwoOptOperator(include_taxi_node=False))',
               'actions.PathAction({idx}, operator=operators.TwoKOptOperator(include_taxi_node=False))',
               'actions.PathAction({idx}, operator=operators.ExchangeOperator(include_taxi_node=False))',
               'actions.PathAction({idx}, operator=operators.InsertOperator(include_taxi_node=False))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=2, include_taxi_node=False))',
               'actions.PathAction({idx}, operator=operators.OForwardOperator(length=3, include_taxi_node=False))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=2))',
               'actions.PathAction({idx}, operator=operators.DBackwardOperator(length=3))',
               'actions.PathAction({idx}, operator=operators.ODPairsExchangeOperator())',
               'actions.PathAction({idx}, operator=operators.MixedBlockExchangeOperator())',
               'actions.MultiPathsAction({idx}, operator=operators.ODPairsInsertMultiVehicles())',
               'actions.MultiPathsAction({idx}, operator=operators.ODPairsExchangeMultiVehicles())'
               ]
    _action_dict = {idx: eval(_action.format(idx=idx)) for idx, _action in enumerate(_actions, start=1)}
    _action_dict[0] = env_instance._regenerate_feasible_solution_with_random_actions
    return _action_dict

def get_pdp_default_random_actions():
       _random_actions = ['actions.PathRandomAction({idx}, operator=operators.RandomODPairsExchangeOperator(change_percentage=0.1))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomOForwardOperator(change_percentage=0.2, include_taxi_node=False))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomDBackwardOperator(change_percentage=0.2))',
                          'actions.PathRandomAction({idx}, operator=operators.RandomODPairsInsertMultiVehicles(change_percentage=0.1))'
                          ]
       _random_actions = [eval(a.format(idx=idx)) for idx, a in enumerate(_random_actions)]
       return _random_actions