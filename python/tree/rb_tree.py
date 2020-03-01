# utf-8
# Red black tree lib
import sys
from btree import BTree, Node

class RBNode(Node):
  def __init__(self,data,deep=0):
    Node.__init__(self,data,deep)
    self.color = 'R' # ONLY in big 'R' and 'B'

class RBTree(BTree):
  def __init__(self,node):
    BTree.__init__(self,node)

  def fix_insert(self,node):
    np = node.parent
    if(np.coler==1):

  
if __name__ == "__main__":
  tr = RBTree(RBNode(50))
  print("end\n")