import sys, os
import numpy as np

def find_all_0_sub_mtx(mtx):
  """
    Find all sub matrix which had 0 sum.
      mtx: ndarray of martix.
    Return: number of total 0 sum sub matrix.
  """
  ret = 0
  def _sub_mtx(mtx):
    ret = 0
    if(max(mtx.shape[0],mtx.shape[1])>2):
      ax0 = mtx.shape[0]
      ax1 = mtx.shape[1]
      if(ax0>2):
        for i in range(ax0-1):
          ret += _sub_mtx(mtx[i:i+2,:])
      if(ax1>2):
        for i in range(ax1-1):
          ret += _sub_mtx(mtx[:,i:i+2])
    elif(max(mtx.shape[0],mtx.shape[1])==1):return 0
    if(np.sum(mtx)==0):
      ret+=1
    return ret
  ret = _sub_mtx(mtx)
  return ret
        
