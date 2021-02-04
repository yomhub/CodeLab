import sys
import os
import math
import numpy as np



ALL_MAGIC_SQUARE=[
    # 所有的3*3奇异阵组合
    [[8, 1, 6],
    [3, 5, 7],
    [4, 9, 2]],

    [[6, 1, 8],
    [7, 5, 3],
    [2, 9, 4]],

    [[4, 9, 2],
    [3, 5, 7],
    [8, 1, 6]],

    [[2, 9, 4],
    [7, 5, 3],
    [6, 1, 8]],

    [[8, 3, 4],
    [1, 5, 9],
    [6, 7, 2]],

    [[4, 3, 8],
    [9, 5, 1],
    [2, 7, 6]],

    [[6, 7, 2],
    [1, 5, 9],
    [8, 3, 4]],

    [[2, 7, 6],
    [9, 5, 1],
    [4, 3, 8]],
]

def maxPoints(points: list) -> int:
    if(len(points)<=1):return len(points)
    point_dict = {}
    ret = 0
    dup_points = []
    for o in points:
        if(o[0] not in point_dict):point_dict[o[0]] = [o[1]]
        elif(o[1] in point_dict[o[0]]): dup_points.append(tuple(o))
        else:point_dict[o[0]] += [o[1]]
        if(len(point_dict[o[0]])>ret):ret=len(point_dict[o[0]])
    
    line_dict = {}
    
    x_list = sorted(point_dict.keys())
    if(len(x_list)==1):
        return len(point_dict[x_list[0]])+len(dup_points)

    for i in range(len(x_list)-1):
        for j in range(i+1,len(x_list)):
            for ss in point_dict[x_list[i]]:
                for es in point_dict[x_list[j]]:
                    k0 = es-ss
                    k1 = x_list[j]-x_list[i]
                    gcd = math.gcd(k1, k0)
                    k0 = int(k0/gcd)
                    k1 = int(k1/gcd)
                    a0 = ss*k1 - k0*x_list[i]
                    a1 = k1
                    gcd = math.gcd(a0, a1)
                    a0 = int(a0/gcd)
                    a1 = int(a1/gcd)
                    if((k0,k1,a0,a1) not in line_dict):line_dict[k0,k1,a0,a1]=[(x_list[i],ss),(x_list[j],es)]
                    else:
                        if((x_list[j],es) not in line_dict[k0,k1,a0,a1]):line_dict[k0,k1,a0,a1].append((x_list[j],es))
                        if((x_list[i],ss) not in line_dict[k0,k1,a0,a1]):line_dict[k0,k1,a0,a1].append((x_list[i],ss))
                    if(len(line_dict[k0,k1,a0,a1])>ret):
                        ret = len(line_dict[k0,k1,a0,a1])

    if(len(dup_points)!=0):
        # if(maxka!=None):
        #     for o in dup_points:
        #         if(o in line_dict[maxka]):
        #             ret+=1
        # else:
        for o in dup_points:
            ret = max(len(point_dict[o[0]])+1,ret)
        for i in line_dict:
            tmp = 0
            for o in dup_points:
                if(o in line_dict[i]):tmp+=1
            ret = max(len(line_dict[i])+tmp,ret)
    return ret


def maxPoints2(points: list) -> int:
    """
    :type points: List[List[int]]
    :rtype: int
    """
    def max_points_on_a_line_containing_point_i(i):
        """
        Compute the max number of points
        for a line containing point i.
        """
        def slope_coprime(x1, y1, x2, y2):
            """ to avoid the precision issue with the float/double number,
                using a pair of co-prime numbers to represent the slope.
            """
            delta_x, delta_y = x1 - x2, y1 - y2
            if delta_x == 0:    # vertical line
                return (0, 0)
            elif delta_y == 0:  # horizontal line
                return (sys.maxsize, sys.maxsize)
            elif delta_x < 0:
                # to have a consistent representation,
                #   keep the delta_x always positive.
                delta_x, delta_y = - delta_x, - delta_y
            gcd = math.gcd(delta_x, delta_y)
            slope = (delta_x / gcd, delta_y / gcd)
            return slope


        def add_line(i, j, count, duplicates):
            """
            Add a line passing through i and j points.
            Update max number of points on a line containing point i.
            Update a number of duplicates of i point.
            """
            # rewrite points as coordinates
            x1 = points[i][0]
            y1 = points[i][1]
            x2 = points[j][0]
            y2 = points[j][1]
            # add a duplicate point
            if x1 == x2 and y1 == y2:  
                duplicates += 1
            # add a horisontal line : y = const
            elif y1 == y2:
                nonlocal horizontal_lines
                horizontal_lines += 1
                count = max(horizontal_lines, count)
            # add a line : x = slope * y + c
            # only slope is needed for a hash-map
            # since we always start from the same point
            else:
                slope = slope_coprime(x1, y1, x2, y2)
                lines[slope] = lines.get(slope, 1) + 1
                count = max(lines[slope], count)
            return count, duplicates
        
        # init lines passing through point i
        lines, horizontal_lines = {}, 1
        # One starts with just one point on a line : point i.
        count = 1
        # There is no duplicates of a point i so far.
        duplicates = 0
        # Compute lines passing through point i (fixed)
        # and point j (interation).
        # Update in a loop the number of points on a line
        # and the number of duplicates of point i.
        for j in range(i + 1, n):
            count, duplicates = add_line(i, j, count, duplicates)
        return count + duplicates
        
    # If the number of points is less than 3
    # they are all on the same line.
    n = len(points)
    if n < 3:
        return n
    
    max_count = 1
    # Compute in a loop a max number of points 
    # on a line containing point i.
    for i in range(n - 1):
        max_count = max(max_points_on_a_line_containing_point_i(i), max_count)
    return max_count

import copy
class Solution:
    def calculateMinimumHP(self, dungeon: list) -> int:
        self.dungeon = dungeon
        self.ans = 999999
        flags = [[False for ax1 in range(len(self.dungeon[0]))] for ax0 in range(len(self.dungeon))]
        self.dfs((0,0),0,0,flags)
        return self.ans+1
        
        
    def dfs(self,sp:tuple,curr_hal:int,min_hal:int,flags):
        if(flags[sp[0]][sp[1]]):
            return
        flags[sp[0]][sp[1]]=True
        curr_hal += self.dungeon[sp[0]][sp[1]]
        if(curr_hal<0):
            min_hal=max(-curr_hal,min_hal)
        if(sp[0]==len(self.dungeon)-1 and sp[1]==len(self.dungeon[0])-1):
            self.ans = min(min_hal,self.ans)
            return
        
        if(sp[0]<len(self.dungeon)-1 and not flags[sp[0]+1][sp[1]]):
            self.dfs((sp[0]+1,sp[1]),curr_hal,min_hal,copy.deepcopy(flags))
        if(sp[1]<len(self.dungeon[0])-1 and not flags[sp[0]][sp[1]+1]):
            self.dfs((sp[0],sp[1]+1),curr_hal,min_hal,copy.deepcopy(flags))
        if(sp[0]>0 and not flags[sp[0]-1][sp[1]]):
            self.dfs((sp[0]-1,sp[1]),curr_hal,min_hal,copy.deepcopy(flags))
        if(sp[1]>0 and not flags[sp[0]][sp[1]-1]):
            self.dfs((sp[0],sp[1]-1),curr_hal,min_hal,copy.deepcopy(flags))            
        return

from scipy.optimize import curve_fit
import torch
def train_fit():
    # xs = np.array(
    #     [[22.58,53.86,60.16,66.69397613,137.43,],
    #     [0.56,0.7,0.8,0.97,0.99,]]
    #     )
    # xs[0] -= xs[0].min()
    # xs[0] /= xs[0].max()
    # xs[1] -= xs[1].min()
    # xs[1] /= xs[1].max()
    xs = np.array([[0.1825,0.148611111,0.118833333,0.156666667,0.088888889,0.058888889,],
        [0.02258,0.05386,0.06016,0.01906,0.066693976,0.13743,]])

    xs = np.moveaxis(xs,0,1)
    # ys = np.array([[205.26,221.93,294.93,550,790,]])
    # ys -= ys.min()
    # ys /= ys.max()
    ys = np.array([[0.146614286,0.158521429,0.210664286,0.1755,0.392857143,0.564285714,]])

    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    nn = torch.nn.Linear(2,1,bias=True).float()
    # nn.weight.data[0,0]=-2.3174439272338
    # nn.weight.data[0,1]=1.31295069384142
    # nn.bias.data[0] = 0.487386364229795
    opt = torch.optim.SGD(nn.parameters(),0.001)
    # opt = torch.optim.SGD(nn.parameters(),0.001)
    while(1):
        opt.zero_grad()
        pred = nn(xs)
        pred = torch.sum(pred,dim=-1,keepdim=True)
        loss = torch.abs(ys-pred)
        loss = torch.topk(loss,4)
        loss = torch.mean(loss[0])
        loss.backward()
        opt.step()
        print(loss)
        if(loss<0.3):
            break
    # torch.save(nn,"D:\\development\\3.pkl")
    cnt = 1
    while(1):
        opt.zero_grad()
        pred = nn(xs)
        pred = torch.sum(pred,dim=-1,keepdim=True)
        loss = torch.abs(ys-pred)
        loss = torch.topk(loss,4)
        loss = torch.mean(loss[0])
        # loss = torch.mean(loss)
        loss.backward()
        opt.step()
        print(loss)
        if(loss<0.1):
            break
        cnt+=1
        if(cnt%80000==0):
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.99

    print(nn.weight.data)
    print(nn.bias.data)
    # torch.save(nn,"D:\\development\\1.pkl")


def PCA():
    def f(x,a1,a2,a3):
        return a1*x[0]+a2*np.exp(a3*x[1])
    m = np.ones((6,6))
    # xs = np.array([[0.029737265,0.29399341,0.347216355,0,0.402415951,1,],
    #     [0,0.325581395,0.558139535,0.813953488,0.953488372,1,]])
    # ys = np.array([0,0.028508397,0.153350207,0.069158942,0.589561172,1])
    # xs = np.array([[0,0.272355246,0.327209404,0.384100793,1,],
    #     [0,0.325581395,0.558139535,0.953488372,1,]])
    # ys = np.array([0,0.028508397,0.153350207,0.589561172,1,])
    # xs = np.moveaxis(xs,0,1)
    xs = np.array([[22.58,53.86,60.16,66.69397613,137.43,],
        [0.56,0.7,0.8,0.97,0.99,]])
    ys = np.array([205.26,221.93,294.93,550,790,])
    # x = 
    popt, pcov = curve_fit(f,xs,ys)
#     np.linalg.svd(popt)
    print(popt)
    print(pcov)
    # nn = torch.load("D:\\development\\3.pkl")
    # weight = nn.state_dict()['weight'].numpy()
    # bias = nn.state_dict()['bias'].numpy()


    # print(weight)
    # print(bias)
    # U,sigma,VT = np.linalg.svd(weight,full_matrices=1,compute_uv=1)
    # sigma,b = np.linalg.eig(weight)
    # for o in nn.state_dict():
    #     ns = nn.state_dict()[o]
    # print(b)

def matrixBlockSum(mat: list, k: int) -> list:
    ans = [[0 for j in range(len(mat[0]))] for i in range(len(mat))]
    for i in range(len(ans)):
        for j in range(len(ans[0])):
            spj = max(0,j-k)
            epj = min(len(ans[0]),j+k+1)
            for di in range(max(0,i-k),min(len(ans),i+k+1)):
                ans[i][j]+=sum(mat[di][spj:epj])
    return ans

import numpy as np

DEF_CH_MAP = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
'O','P','Q','R','S','T','U','V','W','X','Y','Z']
def search_cc(mask,cc_mask):
    """
    Search connected compoments via DFS
    Args:
        mask: number mask
        cc_mask: bool mask
    Return:
        list of 
            (total number,[(idx0,idx1)...])
    """
    CC_list = []
    idaxs = np.stack(np.where(np.logical_and(mask>0,cc_mask==False)),axis=1).tolist()
    while(not np.all(cc_mask)):
        cidx0,cidx1 = idaxs.pop(0)
        if(cc_mask[cidx0,cidx1]):
            continue
        DFS = [(cidx0,cidx1)]
        tmp = []
        cnt = 0
        while(DFS):
            cur_idx = DFS.pop(-1)
            cnt += mask[cur_idx[0],cur_idx[1]]
            tmp.append(cur_idx)
            cc_mask[cur_idx[0],cur_idx[1]]=True
            if(cur_idx[0]>0 and cc_mask[cur_idx[0]-1,cur_idx[1]]==False):
                DFS.append((cur_idx[0]-1,cur_idx[1]))
            if(cur_idx[0]<cc_mask.shape[0]-1 and cc_mask[cur_idx[0]+1,cur_idx[1]]==False):
                DFS.append((cur_idx[0]+1,cur_idx[1]))
            if(cur_idx[1]>0 and cc_mask[cur_idx[0],cur_idx[1]-1]==False):
                DFS.append((cur_idx[0],cur_idx[1]-1))
            if(cur_idx[1]<cc_mask.shape[1]-1 and cc_mask[cur_idx[0],cur_idx[1]+1]==False):
                DFS.append((cur_idx[0],cur_idx[1]+1))
        CC_list.append((cnt,tmp))
    # end of CC search
    CC_list.sort(key = lambda x:x[0],reverse=True)
    return CC_list

# def calculate_score(mask,assign_mask)

def DPmain(lines):
    # このコードは標準入力と標準出力を用いたサンプルコードです。
    # このコードは好きなように編集・削除してもらって構いません。
    # ---
    # This is a sample code to use stdin and stdout.
    # Edit and remove this code as you like.
    DEF_N,DEF_H,DEF_W,DEF_X,DEF_Y,DEF_Z = list(map(int,lines[0].split(' ')))
    score = 0
    team_member_num = list(map(int,lines[1].split(' ')))
    team_member_num_id = [(o,i) for i,o in enumerate(team_member_num)]
    team_member_num_id.sort(key=lambda x:x[0],reverse=True)
    mask = np.array([[int(m) if(m.isdigit())else -1 for m in o] for o in lines[2:]],dtype=np.int16)
    assign = np.zeros(mask.shape,dtype=np.int16)-1
    total_sit = np.sum(mask)
    # find connected compoments
    cc_mask = np.zeros(mask.shape,dtype=np.bool)
    cc_mask[mask<0]=True
    CC_list = search_cc(mask,cc_mask)
    cc_rm = []
    tm_rm = []
    for i,ccs in enumerate(CC_list):
        for j,o in enumerate(team_member_num_id):
            if(o[0]==ccs[0]):
                if(j not in tm_rm):
                    tm_rm.append(j)
                    cc_rm.append(i)
                    break
    for ccid,tmid in zip(cc_rm,tm_rm):
        cur_tm = team_member_num_id[tmid]
        cur_cc = CC_list[ccid]
        cur_score = DEF_Z*cur_tm[0]

        for cidx0,cidx1 in cur_cc[1]:
            assign[cidx0,cidx1]=cur_tm[1]

        for cidx0,cidx1 in cur_cc[1]:
            if(cidx0>0 and assign[cidx0-1,cidx1]>-1 and assign[cidx0-1,cidx1]!=cur_tm[1]):
                cur_score-=DEF_X
            if(cidx0<assign.shape[0]-1 and assign[cidx0+1,cidx1]>-1 and assign[cidx0+1,cidx1]!=cur_tm[1]):
                cur_score-=DEF_X
            if(cidx1>0 and assign[cidx0,cidx1-1]>-1 and assign[cidx0,cidx1-1]!=cur_tm[1]):
                cur_score-=DEF_X
            if(cidx1<assign.shape[1]-1 and assign[cidx0,cidx1+1]>-1 and assign[cidx0,cidx1+1]!=cur_tm[1]):
                cur_score-=DEF_X
    CC_list = [o for i,o in enumerate(CC_list) if(i not in cc_rm)]
    team_member_num_id = [o for j,o in enumerate(team_member_num_id) if(j not in tm_rm)]

    for i in range(mask.shape[0]):
        s = ''
        for j in range(mask.shape[1]):
            if(mask[i,j]<0):
                s+='#'
            elif(assign[i,j]>-1):
                s+=DEF_CH_MAP[assign[i,j]]
            else:
                s+='.'
        print(s)
    

if __name__ == "__main__":

    lines = ['3 4 3 100 0 1',
    '4 3 6',
    '3#1',
    '133',
    '932',
    '##2',]
    DPmain(lines)
    pass

