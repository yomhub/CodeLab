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
    self.prlmb = lambda node: '({},{},{},{},{})'.format(node.data,node.lr_child,node.deep,node.l_max_height,node.r_max_height)

  def fix_insert(self,node):
    np = node.parent
    if(np.coler=='R'):
      """"""
  def print_tree(self):
    self.print_helper(self.root,'',self.prlmb)
    
if __name__ == "__main__":
  tr = RBTree(RBNode(50))
  tr.linsert(RBNode(60))
  tr.linsert(RBNode(40))
  tr.linsert(RBNode(47))
  tr.linsert(RBNode(30))
  tr.linsert(RBNode(33))
  tr.linsert(RBNode(73))
  tr.linsert(RBNode(20))
  tr.linsert(RBNode(120))
  tr.print_tree()
  tr.lrotate(40)
  tr.print_tree()
  tr.remove(50)
  tr.remove(33)
  tr.remove(60)
  tr.print_tree()
  print("end{}\n".format(tr.max_deep))