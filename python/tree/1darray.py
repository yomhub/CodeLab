# BFS, DFS:
# 都是建立一个搜素队列，其中：
# BFS 是先进先出，也就是pop(0)
# DFS 是先进后出，也就是pop()
# 要点是如何保存带有层级依赖的状态
# 用空间换时间

import math
from collections import deque
import heapq
import collections


def numOfMinutes(n: int, headID: int, manager: list, informTime: list) -> int:
    """
    n个员工，工号[0,...,n-1]，有一个领头工号headID，
    所有员工的上级工号存放于manager，上级通知下级的时间为informTime，长度为n
    求最长通知时间
    用BFS做
    """
    max_time = 0
    mem_dict = {}
    for i, o in enumerate(manager):
        if(o == -1):
            continue
        if(o in mem_dict):
            mem_dict[o].append(i)
        else:
            mem_dict[o] = [i]

    search_queue = [(headID, 0)]
    while(search_queue):
        # BFS, pop(0)
        c_id, c_time = search_queue.pop(0)
        if(c_id not in mem_dict):
            continue
        c_time += informTime[c_id]
        for o in mem_dict[c_id]:
            search_queue.append((o, c_time))
        max_time = max(max_time, c_time)
    return max_time


def isCompleteTree(root) -> bool:
    node_list = []
    bfs_list = [root]
    while(bfs_list):
        node = bfs_list.pop(0)
        node_list += [node.val if(node != None)else None]
        if(node == None):
            continue
        bfs_list.append(node.left)
        bfs_list.append(node.right)
    cnt = 0
    i = 0
    is_full = True
    last_have_null = False
    have_null = False
    while(i < len(node_list)):
        last_have_null = have_null
        have_null = False
        for o in node_list[i:i+2**cnt]:
            if(o == None):
                have_null = True
            elif(have_null or last_have_null):
                is_full = False
                break
        if(not is_full):
            break
        i += 2**cnt
        cnt += 1
    return is_full


def networkDelayTime(times: list, N: int, K: int) -> int:
    """
    最短图路径 Dijkstra 
    times: [[始节点(1-N)，终结点(1-N)，权重]]
    N：节点数
    K：root节点
    计算从root节点开始通知整个图的最长时间
    如果存在无法通知的节点返回-1
    """
    graph = collections.defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    list1 = [(0, K)]
    # N 通知成本表，初始化为inf
    dist = {i: float('inf') for i in range(1, N+1)}
    dist[K] = 0
    while len(list1):
        curr_weight, node = heapq.heappop(list1)

        for next_node, in_cost in graph[node]:
            if in_cost+dist[node] < dist[next_node]:
                heapq.heappush(list1, ((in_cost+dist[node]), next_node))
                dist[next_node] = in_cost+dist[node]
    return (max(dist.values())) if (max(dist.values())) != float('inf') else -1


def networkDelayTime2(graph: list, N: int, K: int) -> int:
    dit = collections.defaultdict(list)
    times = {i: float('inf') for i in range(1, N+1)}
    for o in graph:
        u, v, w = o
        dit[u].append((w, v))
    search_list = [(0, K)]
    times[K] = 0
    while(search_list):
        cur_w, cur_n = heapq.heappop(search_list)
        for w, next_n in dit[cur_n]:
            if(w+cur_w < times[next_n]):
                times[next_n] = w+cur_w
                heapq.heappush(search_list, (w+cur_w, next_n))
    ret = max(times.values())
    return int(ret) if(ret != float('inf'))else -1


class TreeAncestor:
    """
    找k-th祖先
    给定n个节点的parent in [0,n-1]，根节点的父节点-1
    建立父节点表dp，计算所有父节点
    建立深度表depth，存储每个节点的深度
    """

    def __init__(self, n: int, parent: list):
        self.log = math.ceil(math.log(n, 2))
        self.dp = [[0]*self.log for _ in range(n)]
        self.depth = [0]*n
        g = [set() for _ in range(n)]
        for i in range(n):
            g[parent[i]].add(i)
        q = deque([0])
        while q:
            rmv = q.popleft()
            for i in g[rmv]:
                if parent[i] == rmv:
                    self.depth[i] = 1+self.depth[rmv]
                    q.append(i)
        for h in range(self.log):
            for node in range(n):
                if h == 0:
                    self.dp[node][h] = parent[node]
                else:
                    self.dp[node][h] = self.dp[self.dp[node][h-1]][h-1]

    def getKthAncestor(self, node: int, k: int) -> int:
        diff = self.depth[node]
        if diff < k:
            return -1
        for i in range(self.log):
            if ((k) & (1 << i)) > 0:
                node = self.dp[node][i]
        return node


class TreeNode:
    def __init__(self, x=0):
        self.val = x
        self.left = None
        self.right = None
        self.next = None


def connectNext(root: TreeNode):
    """
    将整个满树的兄弟节点用单向链表连起来
    复杂度O(n)，内存O(1)，dfs
    """
    _dfs(root)

    def _dfs(root: TreeNode):
        if(root.left != None and root.right != None):
            root.left.next = root.right
            if(root.next != None and root.next.left != None):
                root.right.next = root.next.left
        if(root.left != None):
            _dfs(root.left)
        if(root.right != None):
            _dfs(root.right)


class Solution:
    def findMinHeightTrees(self, n: int, edges: list) -> list:
        if(not edges or n == 1):
            return [n]

        self.nodes_map = [[] for i in range(n)]
        for edg in edges:
            self.nodes_map[edg[0]].append(edg[1])
            self.nodes_map[edg[1]].append(edg[0])
        mp_lens = list(set([len(self.nodes_map[i]) for i in range(n)]))
        mp_lens.sort(reverse=True)
        depths = []
        mht = n
        for ndi, nebs in enumerate(self.nodes_map):
            if(len(nebs) in mp_lens[:2]):
                dep = self.depth(ndi)
                depths.append((ndi, dep))
                mht = min(mht, dep)
        return [o[0] for o in depths if(o[1] == mht)]

    def depth(self, nd: int) -> int:
        bfs = self.nodes_map[nd].copy()
        mask = [False]*len(self.nodes_map)
        mask[nd] = True
        dep = 1
        while(bfs):
            tmp = []
            while(bfs):
                tag = bfs.pop(0)
                if(mask[tag]):
                    continue
                mask[tag] = True
                cad = self.nodes_map[tag]
                for o in cad:
                    if(mask[o] == False):
                        tmp.append(o)
            bfs = tmp
            dep += 1
        return dep


def computeGCD(x, y):

    while(y):
        x, y = y, x % y

    return x


def compute_array_GCD(arr):
    div_num = min(arr)

    while(div_num>0):
        new_div_num = max([o%div_num for o in arr])
        arr = [div_num]
        div_num=new_div_num
    
    return arr[0]


class GenBinartTree:
    def __init__(self):
        self.nd_dict={}
        self.nd_dict[1]=[[TreeNode()]]
        c,l,r = TreeNode(),TreeNode(),TreeNode()
        c.left=l
        c.right=r
        self.nd_dict[3]=[[c,l,r]]
    def allPossibleFBT(self, N: int):
        if(N%2==0):
            N+=1
        return self.genTree(N)
        
    def genTree(self, nd_number:int):
        ans = []
        if(nd_number==1):
            return self.nd_dict[1].copy()
        elif(nd_number==3):
            return self.nd_dict[3].copy()
        else:
            l_nd_number=1
            while(l_nd_number<(nd_number-1) and (nd_number-1-l_nd_number)>=1):
                if(l_nd_number in self.nd_dict):
                    l_gen = self.nd_dict[l_nd_number].copy()
                else:
                    l_gen = self.genTree(l_nd_number)
                    self.nd_dict[l_nd_number]=l_gen.copy()
                r_nd_number=nd_number-1-l_nd_number
                if(r_nd_number in self.nd_dict):
                    r_gen = self.nd_dict[r_nd_number].copy()
                else:
                    r_gen = self.genTree(r_nd_number)
                    self.nd_dict[r_nd_number]=r_gen.copy()
                for l_slc in l_gen:
                    for r_slc in r_gen:
                        l_slc_cp = l_slc.copy()
                        r_slc_cp = r_slc.copy()
                        root = TreeNode()
                        root.left=l_slc_cp[0]
                        root.right=r_slc_cp[0]
                        ans.append([root]+l_slc_cp+r_slc_cp)
                l_nd_number+=2
        return ans

if __name__ == "__main__":
    o = GenBinartTree()
    p7=o.allPossibleFBT(7)
    print('end')
    pass
