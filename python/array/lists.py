import sys, os

def pascal_triangle(row):
  """
    Given an index k, return the kth row of the Pascal’s triangle.
    k start from 0.
    Return: k-th list.
  """
  ret = [1]
  if(row<1):return ret
  for i in range(0,row):
    tmp = []
    for j in range(len(ret)+1):
      if(j==0 or j==len(ret)):tmp.append(1)
      else:tmp.append(ret[j-1]+ret[j])
    ret = tmp
  return ret