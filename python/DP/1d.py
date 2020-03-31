import sys, os

def dp_1d_finite_reg(src_pool,target):
  """
    Finite 1d regression, find the closest combination from src_pool.
    Args:
      src_pool: 1d list.
      target: regression target.
    Return:
      Difference value, combination list.
  """
  cdiff = -1
  if(len(src_pool)==1):
    if(target>src_pool[0]):
      cdiff = target - src_pool[0]
      csubp = src_pool
  else:
    for i in range(len(src_pool)):
      if(target>src_pool[i]):
        dif, subp = dp_1d_finite_reg(src_pool[0:i]+src_pool[i+1:],target-src_pool[i])
        if(cdiff==-1 or dif<cdiff):
          cdiff = dif
          csubp = [src_pool[i]]+subp
  if(cdiff==-1):
    cdiff = target
    csubp = []
  return cdiff,csubp

