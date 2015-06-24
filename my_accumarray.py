import numpy as np
def map_functionnames():
  function_name_dict = {'plus'   : (np.add, 0.),
                        'minus'  : (np.subtract, 0.),
                        'times'  : (np.multiply, 1.),
                        'max'    : (np.maximum,  -np.inf),
                        'min'    : (np.minimum, np.inf),
                        'and'    : (np.logical_and, True),
                        'or'     : (np.logical_or, False)
                       }
  return function_name_dict
def my_accumarray(vals, indices, size, func='plus'):
  # get dictionary
  function_name_dict = map_functionnames()
  if not func in function_name_dict:
    raise KeyError('Function name not defined for accumarray')

  #get the function and the default value
  (function, value) = function_name_dict[func]    
  #create an array to hold things
  output = np.ndarray(size)
  output[:] = value
  function.at(output, indices, vals)
  return output
  
