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
def my_accumarray(indices, vals, size, func='plus', fill_value=0):
  # get dictionary
  function_name_dict = map_functionnames()
  if not func in function_name_dict:
    raise KeyError('Function name not defined for accumarray')
  if np.isscalar(vals):
    vals = np.repeat(vals, indices.size)    
  #get the function and the default value
  (function, value) = function_name_dict[func]    
  #create an array to hold things
  output = np.ndarray(size)
  output[:] = value
  function.at(output, indices, vals)

  # also check whether indices have been used or not
  isthere = np.ndarray(size, 'bool')
  istherevals = np.ones(vals.shape, 'bool')
  (function, value) = function_name_dict['or']
  isthere[:] = value
  function.at(isthere, indices, istherevals)

  #fill things that were not used with fill value
  output[np.invert(isthere)] = fill_value
  return output
  
