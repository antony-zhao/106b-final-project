import numpy as np
import abc
from typing import Callable

class ObservationWrapper(abc.ABC):
    nested_wrapper: Callable
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        
    def __call__(self, observation):
        return observation
    
class TransposeObservation(ObservationWrapper):
    pass