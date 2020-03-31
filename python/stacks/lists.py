import os,sys

def arithmetic_ep(ep_slist):
  """
    Evaluate the value of an arithmetic expression in Reverse Polish Notation.
    Valid operators are +, -, *, /. Each operand may be an integer or another expression.
    Args: ep_slist: a string type list. e.g. ["2", "1", "+", "3", "*"]
    Return: value
  """
  ops = {'+': lambda a,b: a+b, 
  '-': lambda a,b: a-b, 
  '*': lambda a,b: a*b, 
  '/': lambda a,b: a/b}
  if(not(ep_slist[-1] in ops)):return int(ep_slist[-1])
  else: return ops[ep_slist[-1]](arithmetic_ep(ep_slist[:-2]),int(ep_slist[-2]))


if __name__ == "__main__":
  print(arithmetic_ep(["2", "1", "+", "3", "*"]))