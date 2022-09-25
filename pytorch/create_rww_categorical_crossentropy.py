import numpy as np
import torch
import torchvision


# adapted from Keras via https://github.com/yaoshiang/The-Real-World-Weight-Crossentropy-Loss-Function/blob/master/Real_World_Weighted_Crossentropy_Loss_Function.ipynb

def create_rww_categorical_crossentropy(k, loss_type, fn_weights=None, fp_weights=None, return_weights=False):
  """Real-World-Weighted crossentropy between an output tensor and a target tensor.
  
  The loss_types other than rww_categorical_crossentropy reimplement existing 
  functions in Keras but are not as well optimized. 
  These loss_types are usable directly, but, are more useful when calling 
  return_weights=True, which then returns fn and fp weights matrixes of size (k,k). 
  Editing those to reflect real world costs, then passing them back into 
  create_rww_crossentropy with loss_type "rww_crossentropy" is the recommended approach. 

  Example Usage: 

  Suppose you have three classes: cat, dog, and other.
  
  Cat is one-hot encoded as [1,0,0], dog as [0,1,0], other as [0,0,1]
  
  The the following code increases the incremental penalty of 
  mislabeling a true target 0 (cat) with a false label 1 (dog) at a cost of 99, 
  versus the default of zero. Note that the existing fn_weights also has a 
  default cost of 1 for missing the true target of 1, for a total cost of 
  100 versus the default cost of 1. 
  
  fn_weights, fp_weights = create_rww_categorical_crossentropy(10, "categorical_crossentropy", return_weights=True)
  fp_weights[0, 1] = 99
  loss = create_rww_categorical_crossentropy(10, "rww_crossentropy", fn_weights, fp_weights)

... 
  
  The fn and fp weights are easy to reason about. 
  
  fn_weights is [x1, __, __]
                [__, x2, __]
                [__, __, x3]
 
  x1 represents the scale of the cost for a fn for cat, x2 for dog, and x3 for other.
  
  This is calculated as fn_weight * log(y_pred). 
  
  In the case of loss_type=categorical_crossentropy, 
  x1, x2, and x3 all equal the value one. 
  All elements not on the main axis must equal zero. 
  
  Note that fn_weights could have been represented as a vector, 
  not a matrix, however, we use a matrix to keep symmetry with 
  fp_weights, and, to prepare for 
  multi-label classification. 
    
  ...

  fp_weights is concerned with the costs of the fps from the other classes. 

  fp_weights of [__, x1, x2]
                [x3, __, x4]
                [x5, x6, __]
 
  x1 represents the cost of predicting 1 for dog, when it should be 0 for cat. 
  x2 represents predicting 2 for other, when the target is 0 for cat. 
  x3 represents predicing 0 for cat, when the target is 1 for dog.
  etc. 
  
  Args:
    * k: 2 or more for number of categories, including "other". 
    * loss_type: "categorical_crossentropy" to initialize to 
      standard softmax_crossentropy behavior, 
      or "weighted_categorical_crossentropy" for standard behavior, or, 
      or "rww_crossentropy" for full weight matrix of all possible fn/fp combinations. 
    * fn_weights: a numpy array of shape (k,k). The main diagonal can
      contain non-zero values; all other values must be zero. 
    * fp_weights: a numpy array of shape (k,k) to define specific combinations 
      of false positive. The main diag should be zeros. 
    * return_weights: If False (default), returns cost function. If True, 
      returns fn and fp weights as np.array. 
Returns:
    * retval: Loss function for use Keras.model.fit, or if return_weights
      arg is True, the fn_weights and fp_weights matrixes. 
  """
  epsilon = 1e-7
  full_fn_weights = None
  full_fp_weights = None

  anti_eye = np.ones((k,k)) - np.eye(k)
    
  if (loss_type=="categorical_crossentropy"):
    full_fn_weights = np.identity((k))
    full_fp_weights = np.zeros((k, k)) # Softmax crossentropy ignores fp.

  elif(loss_type=="weighted_categorical_crossentropy"):
    full_fn_weights = np.eye(k) * fn_weights
    full_fp_weights = np.zeros((k, k)) # softmax crossentropy ignores fp
    
  elif(loss_type=="rww_crossentropy"):
    assert not np.count_nonzero(fn_weights * anti_eye)
    assert not np.count_nonzero(fp_weights * np.eye(k))

    full_fn_weights = fn_weights
    # Novel piece: allow any combination of fp.
    full_fp_weights = fp_weights
    
  else:
    raise Exception("unknown loss_type: " + str(loss_type))
   
  fn_wt = torch.nn.Parameter(torch.tensor(full_fn_weights, dtype=torch.double, device = device)) # (k,k), always sparse along main diag. 
  fp_wt = torch.nn.Parameter(torch.tensor(full_fp_weights, dtype=torch.double, device = device)) # (k,k), always dense except main diag. 

  def loss_function( output,target):

    output = torch.nn.Parameter(torch.clamp(output, epsilon, 1 - epsilon))
    
    logs = torch.nn.Parameter(torch.log(output)) # shape (m, k), dense. 1 is good. 
    logs_1_sub = torch.nn.Parameter(torch.log(1-output)) # shape (m, k), dense. 0 is good. 

    m_full_fn_weights = torch.nn.Parameter(torch.matmul(target ,fn_wt)) # (m,k) . (k, k)
    m_full_fp_weights = torch.nn.Parameter(torch.matmul(target ,fp_wt)) # (m,k) . (k, k)
    
    return - torch.nn.Parameter(torch.mean(m_full_fn_weights * logs + 
                    m_full_fp_weights * logs_1_sub))
  
  if (return_weights):
    return full_fn_weights, full_fp_weights
  else:
    return loss_function