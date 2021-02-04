# utf-8
# Binary Trees lib
import sys


class Node():
    def __init__(self, data, deep=0):
        self.data = data  # holds the key
        self.parent = None  # pointer to the parent
        self.lr_child = None  # ONLY in big 'L' or 'R' or None
        self.left = None  # pointer to left child
        self.right = None  # pointer to right child
        self.deep = deep  # current height
        self.l_max_height = 0  # left sub-tree height
        self.r_max_height = 0  # right sub-tree height

    def __iter__(self):
        if(self.left != None):
            yield from self.left.__iter__()
        yield self.data
        if(self.right != None):
            yield from self.right.__iter__()

    def __eq__(self, other):
        if(other == None):
            return False
        return self.data == other.data


class BTree():
    def __init__(self, node):
        assert(node != None)
        self.root = node
        self.root.deep = 1
        self.max_deep = 1
        self.node_list = [self.root]
        self.key_list = [self.root.data]

    def __shallower(self, node):
        node.deep -= 1
        if(node.left != None):
            self.__shallower(node.left)
        if(node.right != None):
            self.__shallower(node.right)

    def __deeper(self, node):
        node.deep += 1
        if(node.left != None):
            self.__deeper(node.left)
        if(node.right != None):
            self.__deeper(node.right)

    def __biggest(self, node):
        inp = node
        while(inp.right != None):
            inp = inp.right
        return inp

    def __smallest(self, node):
        inp = node
        while(inp.left != None):
            inp = inp.left
        return inp

    def __update_height(self, node):
        inp = node
        while(inp.parent != None):
            if(inp.lr_child == 'L'):
                inp.parent.l_max_height = max(
                    inp.l_max_height, inp.r_max_height)+1
            else:
                inp.parent.r_max_height = max(
                    inp.l_max_height, inp.r_max_height)+1
            inp = inp.parent

    def lrotate(self, key):
        try:
            inc = self.key_list.index(key)
        except:
            return
        node = self.node_list[inc]
        nr = node.right
        if(nr == None):
            return
        nrl = node.right.left
        nf = node.parent

        # Process node->right->left
        node.deep += 1
        if(nrl != None):
            node.r_max_height = max(nrl.l_max_height, nrl.r_max_height)+1
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
        if(nf != None):
            nr.lr_child = node.lr_child
            if(node.lr_child == 'L'):
                nf.left = nr
                nf.l_max_height = max(nr.l_max_height, nr.r_max_height)+1
            else:
                nf.right = nr
                nf.r_max_height = max(nr.l_max_height, nr.r_max_height)+1
        node.lr_child = 'L'
        nr.l_max_height = max(node.l_max_height, node.r_max_height)+1

        # Update deep
        if(nr.right != None):
            self.__shallower(nr.right)
        if(node.left != None):
            self.__deeper(node.left)
        nd_h = node.deep + max(node.l_max_height, node.r_max_height)
        if(nd_h > self.max_deep):
            self.max_deep = nd_h

    def rrotate(self, key):
        try:
            inc = self.key_list.index(key)
        except:
            return
        node = self.node_list[inc]
        nl = node.left
        if(nl == None):
            return
        nlr = node.left.right
        nf = node.parent
        node.deep += 1

        # Process node->left->right
        if(nlr != None):
            node.l_max_height = max(nlr.l_max_height, nlr.r_max_height)+1
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
        if(nf != None):
            nl.lr_child = node.lr_child
            if(node.lr_child == 'L'):
                nf.left = nl
                nf.l_max_height = max(nl.l_max_height, nl.r_max_height)+1
            else:
                nf.right = nl
                nf.r_max_height = max(nl.l_max_height, nl.r_max_height)+1
        node.lr_child = 'R'
        nl.r_max_height = max(node.l_max_height, node.r_max_height)+1

        # Update deep
        if(nl.left != None):
            self.__shallower(nl.left)
        if(node.right != None):
            self.__shallower(node.right)
        nd_h = node.deep + max(node.l_max_height, node.r_max_height)
        if(nd_h > self.max_deep):
            self.max_deep = nd_h

    def linsert(self, node):
        if(node == None):
            return
        key = node.data
        if(key in self.key_list):
            return
        if(self.root == None):
            self.key_list += [key]
            self.node_list += [node]
            self.root = node
            return
        inp = self.root

        while(inp != None):
            if(key < inp.data):
                if(inp.left == None):
                    node.lr_child = 'L'
                    inp.left = node
                    break
                inp = inp.left
            else:
                if(inp.right == None):
                    node.lr_child = 'R'
                    inp.right = node
                    break
                inp = inp.right
        node.deep = inp.deep+1
        node.parent = inp
        if(node.deep > self.max_deep):
            self.max_deep = node.deep
        self.key_list += [key]
        self.node_list += [node]
        inp = node
        while(inp.parent != None):
            if(inp.lr_child == 'L'):
                inp.parent.l_max_height = max(
                    inp.l_max_height, inp.r_max_height)+1
            else:
                inp.parent.r_max_height = max(
                    inp.l_max_height, inp.r_max_height)+1
            inp = inp.parent

    def remove(self, key):
        try:
            ind = self.key_list.index(key)
        except:
            return
        node = self.node_list[ind]

        # case 1: have LR child
        if(node.left != None and node.right != None):
            tar = self.__biggest(node.left)
            if(tar.left != None):
                tar.left.lr_child = tar.lr_child
                tar.left.parent = tar.parent
                if(tar.lr_child == 'L'):
                    tar.parent.left = tar.left
                    tar.parent.l_max_height = max(
                        tar.left.l_max_height, tar.left.r_max_height)+1
                else:
                    tar.parent.right = tar.left
                    tar.parent.r_max_height = max(
                        tar.left.l_max_height, tar.left.r_max_height)+1
                self.__shallower(tar.left)
            if(node.left == tar):
                node.left = None
                node.l_max_height = 0
            else:
                node.l_max_height = max(
                    node.left.l_max_height, node.left.r_max_height)+1
            node.data = tar.data
        # case 2: only have L child
        elif(node.left != None):
            if(node.parent != None):
                node.left.lr_child = node.lr_child
                node.left.parent = node.parent
                if(node.lr_child == 'L'):
                    node.parent.left = node.left
                    node.parent.l_max_height = max(
                        node.left.l_max_height, node.left.r_max_height)+1
                else:
                    node.parent.right = node.left
                    node.parent.r_max_height = max(
                        node.left.l_max_height, node.left.r_max_height)+1
            else:
                node.left.parent = None
                node.left.lr_child = None
                self.root = node.left
            self.__shallower(node.left)
        # case 3: only have R child
        elif(node.right != None):
            if(node.parent != None):
                node.right.lr_child = node.lr_child
                node.right.parent = node.parent
                if(node.lr_child == 'L'):
                    node.parent.left = node.right
                    node.parent.l_max_height = max(
                        node.right.l_max_height, node.right.r_max_height)+1
                else:
                    node.parent.right = node.right
                    node.parent.r_max_height = max(
                        node.right.l_max_height, node.right.r_max_height)+1
            else:
                node.right.parent = None
                node.right.lr_child = None
                self.root = node.right
            self.__shallower(node.right)
        # case 4:leaf node
        else:
            if(node.parent == None):  # root
                self.root = None
            else:
                if(node.lr_child == 'L'):
                    node.parent.left = None
                    node.parent.l_max_height = 0
                else:
                    node.parent.right = None
                    node.parent.r_max_height = 0
        if(node.parent != None):
            self.__update_height(node.parent)

        del(self.key_list[ind])
        del(self.node_list[ind])

    def print_helper(self, node, indent, nodelamb=None):
        if(node == None):
            return
        sys.stdout.write(indent)
        if(node.lr_child):
            sys.stdout.write(node.lr_child+"---")

        if(nodelamb):
            sys.stdout.write(nodelamb(node)+'\n')
        else:
            sys.stdout.write('({})\n'.format(node.data))
        indent += '|   '
        if(node.left):
            self.print_helper(node.left, indent, nodelamb)
        if(node.right):
            self.print_helper(node.right, indent, nodelamb)

    def print_tree(self, nodelamb=None):
        self.print_helper(self.root, '', nodelamb)



class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

def init_from_value_list(node_list:list):
    """
    node_list: [1,2,3....] value of all nodes
    return:
        List of TreeNode with root at ret[0]
    """
    ret = []
    
    level = 0
    i=j=0
    tmp=[]
    while(i+j<len(node_list)):
        tmp.append(TreeNode(node_list[i+j]))
        j+=1
        if(j>=2**level):
            i+=j
            j=0
            level+=1
            ret.append(tmp)
            tmp = []

    for i,nl in enumerate(ret[:-1]):
        for j,o in enumerate(nl):
            if(j*2<len(ret[i+1])):
                o.left = ret[i+1][j*2] if(ret[i+1][j*2].val!=None)else None
            if(j*2+1<len(ret[i+1])):
                o.right = ret[i+1][j*2+1] if(ret[i+1][j*2+1].val!=None)else None

    return ret

def init_from_link_list(link_list:list)->list:
    """
    link_list: e.g. [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], 
        [[from,to]] of all nodes
    return:
        List of TreeNode with root at ret[0]
    """
    ret = [TreeNode(i) for i in range(len(link_list))]
    for i,o in enumerate(link_list):
        if(ret[o[0]].val>=ret[o[1]].val):
            ret[o[0]].left=ret[o[1]]
        else:
            ret[o[0]].right=ret[o[1]]
    return ret
