# utf-8
# Binary Trees lib
import sys

class Node():
  def __init__(self,data,deep=0):
    self.data = data  # holds the key
    self.parent = None # pointer to the parent
    self.lr_child = None # ONLY in big 'L' or 'R' or None
    self.left = None # pointer to left child
    self.right = None #pointer to right child
    self.deep = deep # current height
    self.l_max_height = 0 # left sub-tree height
    self.r_max_height = 0 # right sub-tree height

class BTree():
  def __init__(self,node):
    assert(node!=None)
    self.root = node
    self.root.deep = 1
    self.max_deep = 1
    self.node_list = [self.root]
    self.key_list = [self.root.data]

  def __shallow_subtree(self,node):
    node.deep-=1
    if(node.left!=None):
      self.__shallow_subtree(node.left)
    if(node.right!=None):
      self.__shallow_subtree(node.right)

  def lrotate(self,node):
    if(node==None):return
    nr = node.right
    if(nr==None):return
    nrl = node.right.left
    node.deep += 1
    nf = node.parent

    # Process node->right->left
    if(nrl!=None):
      node.r_max_height = max(nrl.l_max_height,nrl.r_max_height)+1
      nrl.parent = node
      node.right = nrl
    else:
      node.r_max_height = 0
      node.right = None
    
    # Proces node->right
    nr.deep -= 1
    nr.left = node
    nr.parent = nf
    node.parent = nr
    nr.lr_child = None
    if(nf!=None):
      nr.lr_child = node.lr_child
      if(node.lr_child=='L'):
        nf.left = nr
        nf.l_max_height = max(nr.l_max_height,nr.r_max_height)+1
      else:
        nf.right = nr
        nf.r_max_height = max(nr.l_max_height,nr.r_max_height)+1
    node.lr_child = 'L'
    nr.l_max_height = max(node.l_max_height,node.r_max_height)+1

    # Update deep
    if(nr.right!=None):
      self.__shallow_subtree(nr.right)
    if(node.left!=None):
      self.__shallow_subtree(node.left)
    nd_h = node.deep + max(node.l_max_height,node.r_max_height)
    if(nd_h > self.max_deep):self.max_deep=nd_h

  def rrotate(self,node):
    if(node==None):return
    nl = node.left
    if(nl==None):return
    nlr = node.left.right
    nf = node.parent
    node.deep += 1

    # Process node->left->right
    if(nlr!=None):
      node.l_max_height = max(nlr.l_max_height,nlr.r_max_height)+1
      nlr.parent = node
      node.left = nlr
    else:
      node.r_max_height = 0
      node.left = None
    
    # Proces node->right
    nl.deep -= 1
    nl.right = node
    nl.parent = nf
    node.parent = nl
    nl.lr_child = None
    if(nf!=None):
      nl.lr_child = node.lr_child
      if(node.lr_child=='L'):
        nf.left = nl
        nf.l_max_height = max(nl.l_max_height,nl.r_max_height)+1
      else:
        nf.right = nl
        nf.r_max_height = max(nl.l_max_height,nl.r_max_height)+1
    node.lr_child = 'R'
    nl.r_max_height = max(node.l_max_height,node.r_max_height)+1

    # Update deep
    if(nl.left!=None):
      self.__shallow_subtree(nl.left)
    if(node.right!=None):
      self.__shallow_subtree(node.right)
    nd_h = node.deep + max(node.l_max_height,node.r_max_height)
    if(nd_h > self.max_deep):self.max_deep=nd_h

  def linsert(self,node):
    if(node==None):return
    key = node.data
    if(key in self.key_list):return
    inp = self.root
    lrch = None
    while(inp!=None):
      if(key<inp.data):
        if(inp.left==None):
          lrch = 'L'
          inp.left=node
          inp.l_max_height=1
          break
        inp=inp.left
      else:
        if(inp.right==None):
          lrch = 'R'
          inp.right=node
          inp.r_max_height=1
          break
        inp=inp.right
    node.deep = inp.deep+1
    node.parent = inp
    node.lr_child = lrch
    if(node.deep > self.max_deep):self.max_deep=node.deep
    self.key_list+=[key]
    self.node_list+=[node]

  def delete_node(self,key):
    try:
      ind = self.key_list.index(key)
      node = self.node_list[ind]

    except:
      return


  def print_helper(self, node, indent, nodelamb=None):
    if(node==None):return
    sys.stdout.write(indent)
    if(node.lr_child):sys.stdout.write(node.lr_child+"---")

    if(nodelamb):sys.stdout.write(nodelamb(node)+'\n')
    else:sys.stdout.write('({})\n'.format(node.data))
    indent+='|   '
    if(node.left):self.print_helper(node.left,indent,nodelamb)
    if(node.right):self.print_helper(node.right,indent,nodelamb)

  def print_tree(self,nodelamb=None):
    self.print_helper(self.root,'',nodelamb)