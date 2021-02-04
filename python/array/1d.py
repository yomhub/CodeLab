import sys
import os
import heapq
import math
from collections import Counter

def pascal_triangle(row):
    """
      Given an index k, return the kth row of the Pascal’s triangle.
      k start from 0.
      Return: k-th list.
    """
    ret = [1]
    if(row < 1):
        return ret
    for i in range(0, row):
        tmp = []
        for j in range(len(ret)+1):
            if(j == 0 or j == len(ret)):
                tmp.append(1)
            else:
                tmp.append(ret[j-1]+ret[j])
        ret = tmp
    return ret


def max_sub_array(A):
    """
    Max sub array question
    Find the contiguous subarray within an array, A of length N which has the largest sum.
    E.g. [-1,-1,0,1,2] => sum([1,2])
    """
    if(len(A) == 1):
        return A[0]
    c = 0
    mx = A[0]
    # (0 or -int) + o < b
    for o in A:
        c = o if(c <= 0)else c+o
        mx = max(mx, c)

    return mx


def min_sub_array(A, div):
    """
    Min subarray question with binary search method
    Ginven divide number div, divide A into div sub array
    s.t. the max sum of all sub array is minimum
    E.g. [7,2,5,10,8], div=2 => [7,2,5][10,8]
    将问题转换为二值回归问题
    查找出一个max sum 阈值能最小化拆分整个数组
    复杂度O(NlogN)
    """
    def __can_split(A, div, maxsum) -> bool:
        count = 0
        csum = 0
        for o in A:
            csum += o
            if(csum > maxsum):
                count += 1
                csum = o
                if(count > div):
                    return False
        # Because we have [1]/[2]/.../[n], so the final part should be count
        if((count+1) > div):
            return False
        return True

    low = A[0]
    high = A[0]
    for o in A[1:]:
        high += o
        if(low > o):
            low = o
    # calculate threshold
    while(low < high):
        mid = (low + high)/2
        if(__can_split(A, div, mid)):
            high = mid
        else:
            low = mid+1
    # split the entire array
    csum = 0
    csums = []
    for o in A:
        if((csum+o) > low):
            csums += [csum]
            csum = o
        else:
            csum += o

    return min(csums)

def shortestSubarray(arr, k):
    """
    Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K.
    If there is no non-empty subarray with sum at least K, return -1.

    转化子数组最小和问题为最小和差值问题
    存储从[0-i]的数列和sum[i]，寻找所有sum[i2]-sum[i1]>=k的例子计算i2-i1(子串长度)
    并返回最短长度
    """
    s = []    # Inc    # (sum, ind)
    curr = 0
    ans = len(arr)+2
    for i in range(len(arr)):
        curr += arr[i]
        if curr>=k: ans=min(ans, i+1)
        while s and curr - s[0][0]>=k:
            ans = min(ans, i-s.pop(0)[1])
        while s and s[-1][0]>curr:
            s.pop()
        s.append((curr, i))
    
    return ans if(ans<=len(arr))else -1


def longestSubarray(arr, d: int) -> int:
    """
    给定arr, d，返回最长连续子数组长度使 abs(max(subarr)-min(subarr))<=d
    手动维护subarr的max和min比较麻烦
    于是使用有序堆heapq库来维护 max 和 min 队列
    """
    min_heap = []
    max_heap = []
    left = 0
    max_lenght = 0
    import heapq
    for right in range(len(arr)):
        # heapq: 有序堆队列，从小到大
        heapq.heappush(min_heap, [arr[right], right])
        # -arr的从小到大 == arr从大到小
        heapq.heappush(max_heap, [-arr[right], right])
        min_elt, max_elt = min_heap[0][0], -max_heap[0][0]
        # 满足条件则更新 max_lenght
        if abs(max_elt - min_elt) <= d:
            if right - left + 1 > max_lenght:
                max_lenght = right - left + 1
        # 不满足条件则调整到满足条件
        while max_heap and min_heap and -max_heap[0][0] - min_heap[0][0] > d:
            while max_heap and max_heap[0][1] <= left:
                heapq.heappop(max_heap)
            while min_heap and min_heap[0][1] <= left:
                heapq.heappop(min_heap)
            left += 1

    return max_lenght


def isPossibleMakeTarget(target: list) -> bool:
    """
    给定一个目标数组长n，判断是否能从[1]*n开始组合成目标数组
    每次组合步骤：求和后替换某个元素
    通过最大堆逆像推导能否合成，假设 K0-Kn-1
    最大元素一定由之前的数组构成，即 K0=sum(K1-Kn-1)+a，所以K0>sum(K1-Kn-1)
    那么如何把 K0-Kn-1 还原成 K1-Kn-1,a
    a = K0-sum(K1-Kn-1)，为了更加效率使用 a = K0%sum(K1-Kn-1)
    最后能还原成n个1则成立
    """
    tag_size = len(target)
    cnt = 0
    max_heap = []
    for o in target:
        heapq.heappush(max_heap,-o)
    c_sum=sum(target)

    while(cnt<tag_size):
        top = heapq.heappop(max_heap)
        top = -top
        c_sum-=top
        if(c_sum==1 or top==1):return True
        elif(c_sum<=0 or c_sum>=top):return False
        elif(top%c_sum==1):cnt+=1
        heapq.heappush(max_heap,0-top%c_sum)
        c_sum+=top%c_sum
            
    return True

def canPartitionKSubsets(nums: list, k: int) -> bool:
    traget_sum = sum(nums)
    if(traget_sum%k!=0):return False
    traget_sum = traget_sum//k
    
    nums.sort()
    if(nums[-1]>traget_sum):return False
    tmp_sum=0
    
    while(nums):
        if(tmp_sum==0):
            tmp_sum=nums.pop(-1)
        det = traget_sum-tmp_sum
        if(len(nums)==0 and det!=0):return False
        if(det==0):
            tmp_sum=0
            continue
        for i in range(len(nums)):
            if(nums[i]>=det):break
        if(nums[i]==det):
            nums.pop(i)
            tmp_sum=0
        elif(i==0):return False
        elif(nums[i]>=det):
            tmp_sum += nums.pop(i-1)
        else:
            tmp_sum += nums.pop(i)
        
    return False

def maxProfit(prices: list,k: int) -> int:
    """
    股价买卖k次求最大收益
    使用buy,sell构建最佳连续买入卖出点N个
    profit为每个点组对应的利益N个
    dif为每个点组之间的落差差价N-1个
    profit等同于上扬区间，dif等同于下降区间
    所以min(dif) < min(profit)的时候可以合并
    反之丢弃小profit
    """
    l = len(prices)
    if l == 0: return 0
    buy, sell, profit, dif = [], [], [], []
    
    for price in prices:
        if len(buy) == len(sell):
            if sell and price >= sell[-1]:
                sell.pop()
                sell.append(price)
            else:
                buy.append(price)
        elif len(buy) > len(sell):
            if price <= buy[-1]:
                buy.pop()
                buy.append(price)
                continue
            else:
                sell.append(price)
        if sell and len(buy) > len(sell):                 
            i = len(sell) - 1
            profit.append(sell[i]-buy[i])
            if i > 0 : dif.append(sell[i-1]-buy[i])
            
    if len(buy) == len(sell):
        i = len(sell) - 1
        profit.append(sell[i]-buy[i])
        dif.append(sell[i-1]-buy[i])         
    
    while len(sell) > k:
        if min(dif) < min(profit):
            idx = dif.index(min(dif))
            buy.pop(idx+1)
            sell.pop(idx)
            if idx > 0: dif[idx-1] = (sell[idx-1]-buy[idx])
            profit[idx] = sell[idx] - buy[idx]
            dif.pop(idx)
            profit.pop(idx+1)
        else:
            idx = profit.index(min(profit)) 
            buy.pop(idx)
            sell.pop(idx)
            if 0 < idx < len(profit)-1: dif[idx] = (sell[idx-1]-buy[idx])
            profit.pop(idx)
            if idx == 0: dif.pop(0)
            else: dif.pop(idx-1)            
        
    return  sum(sell[x]-buy[x] for x in range(len(sell)))


def largestDivisibleSubset(nums: list) -> list:
    
    if(len(nums)<=1):return nums
    nums.sort()
    cnt_tab = [[nums[j]%nums[i] for j in range(i+1,len(nums))] for i in range(len(nums)-1)]
    cnt0 = [o.count(0) for o in cnt_tab]
    ind = -1
    ans = []
    while(True):
        try:
            ind = cnt0.index(max(cnt0),ind+1)
        except:
            break
        if(nums[ind]==1):
            ind = cnt0.index(max(cnt0[:ind]+cnt0[ind+1:]))
        tmp = []
        c = nums[ind]
        for o in nums:
            if(o%c==0 or c%o==0):
                tmp.append(o)
                c=max(o,c)
        if(len(tmp)>len(ans)):
            ans = tmp
    return ans

def ranges(nb):
    if(not nb):
        return ''
    if(len(nb)==1):
        return str(nb[0])
    # Write your code here
    nb = list(set(nb))
    nb.sort()
    ans = str(nb[0])
    hit=False
    for i,o in enumerate(nb[1:]):
        if(nb[i]+1!=o):
            if(ans[-1].isdigit() and not hit):
                ans+=',{}'.format(o)
            else:
                ans+='-{},{}'.format(nb[i],o)
                hit=False
        else:
            hit=True
    if(nb[-2]+1==nb[-1]):
        ans+='-{}'.format(nb[-1])
    return ans

def calculate(ex):
    # Write your code here
    nums,ops = [],[]
    def calc(nums,ops):
        if(ops and len(nums)>=2 and nums[-1] not in ['(',')'] and nums[-2] not in ['(',')']):
            while('#' in ops):
                try:
                    idx = len(ops)-ops[::-1].index('#')-1
                    idy = -len(ops)-1+idx
                    nums[idy]=(nums[idy]*nums[idy]*nums[idy+1])%10007
                    del nums[idy+1]
                    del ops[idx]
                except:
                    break
            while('$' in ops):
                try:
                    idx = len(ops)-ops[::-1].index('$')-1
                    idy = -len(ops)-1+idx
                    nums[idy]=(nums[idy]*nums[idy+1]*nums[idy+1])%10007
                    del nums[idy+1]
                    del ops[idx]
                except:
                    break
            while('!' in ops):
                try:
                    idx = ops.index('!')
                    idy = -len(ops)-1+idx
                    nums[idy]=(nums[idy]-nums[idy+1])%10007
                    del nums[idy+1]
                    del ops[idx]
                except:
                    break
            while('@' in ops):
                try:
                    idx = ops.index('@')
                    idy = -len(ops)-1+idx
                    nums[idy]=(nums[idy]+nums[idy+1])%10007
                    del nums[idy+1]
                    del ops[idx]
                except:
                    break  
        return nums[-1]
    sp = 0
    while(sp<len(ex)):
        ep = sp
        while(ep<len(ex) and ex[ep].isdigit()):
            ep+=1
        if(ep!=sp):
            nums.append(int(ex[sp:ep]))
            sp=ep
            continue
        if(ex[sp] in ['!','@']):
            ops.append(ex[sp])
        elif(ex[sp] in ['#','$']):
            ops.append(ex[sp])
        elif(ex[sp]=='('):
            nums.append(ex[sp])
        elif(ex[sp]==')'):
            idx = len(nums)-nums[::-1].index('(')-1
            idy = len(nums)-idx
            ret = calc(nums[idx+1:],ops[idy:])
            nums = nums[:idx]
            nums[-1]=ret
            ops=ops[:idy]
        sp+=1
    return calc(nums,ops)

def solvePath(cvl, stp):
    # Write your code here
    mask = [[False for j in range(len(cvl[0]))] for i in range(len(cvl))]
    MX_STP = stp
    def dfs(i,j,rot,god,mask,stp):
        if(i==len(mask)-1 and j==len(mask[0])-1):
            return rot,god
        elif(stp==0):
            return [[999,999]],-1
        mask[i][j]=True
        if(cvl[i][j]==2):
            god+=1
        if(i+1<len(cvl) and cvl[i+1][j]!=0 and not mask[i+1][j]):
            nrot,ngod = dfs(i+1,j,rot+[[j,i+1]],god,mask.copy(),stp-1)
            if(ngod>=god):
                rot,god=nrot,ngod
        if(i-1>=0 and cvl[i-1][j]!=0 and not mask[i-1][j]):
            nrot,ngod = dfs(i-1,j,rot+[[j,i-1]],god,mask.copy(),stp-1)
            if(ngod>=god):
                rot,god=nrot,ngod
        if(j+1<len(cvl[0]) and cvl[i][j+1]!=0 and not mask[i][j+1]):
            nrot,ngod = dfs(i,j+1,rot+[[j+1,i]],god,mask.copy(),stp-1)
            if(ngod>=god):
                rot,god=nrot,ngod
        if(j-1<len(cvl[0]) and cvl[i][j-1]!=0 and not mask[i][j-1]):
            nrot,ngod = dfs(i,j-1,rot+[[j-1,i]],god,mask.copy(),stp-1)
            if(ngod>=god):
                rot,god=nrot,ngod
        return rot,god
    rot,god = dfs(0,0,[[0,0]],0,mask,stp)
    if(len(rot)>stp or rot[-1]!=[len()]):
        return [[999,999]]
    return rot

def nonDivisibleSubset(k, numbers):
    """
    计算数组中的最大组合大小使组合中两两相加无法被k整除
    
    """
    counts = [0] * k
    for number in numbers:
        counts[number % k] += 1

    count = min(counts[0], 1)
    for i in range(1, k//2+1):
        if i != k - i:
            count += max(counts[i], counts[k-i])
    if k % 2 == 0: 
        count += 1

    return count

def wetset(total,wet,sets):
    wetset = [(o,i) for i,o in enumerate(sets) if(i%2==1)]
    dryset = [(o,i) for i,o in enumerate(sets) if(i%2==0)]
    dp = [[] for i in range(len(sets))]
    dp[0]=[sets[0],wet,0]
    for i in range(len(sets)-1):
        if((i+1)%2==0):
            dp[i+1]=[[p[0]+sets[i],p[1],p[2]] for p in dp[i]]
        else:
            for p in dp[i]:
                if(p[1]>=sets[i+1]):
                    dp[i+1].append([p[0],p[1]-sets[i+1],p[2]])
            dp[i+1].append([0,wet,i+2])
    

def cellCompete(states, days):
    # WRITE YOUR CODE HERE
    states_new = states.copy()
    for d in range(days):
        for i in range(len(states)):
            if(i==0):
                if(states[i+1]==0):
                    states_new[i]=0
                else:
                    states_new[i]=1
            if(i==(len(states)-1)):
                if(states[i-1]==0):
                    states_new[i]=0
                else:
                    states_new[i]=1
            else:
                if(states[i-1]==states[i+1]):
                    states_new[i]=0
                else:
                    states_new[i]=1
        states=states_new
        states_new=states_new.copy()
            
    return states

if __name__ == "__main__":
    t=(
    [1,1,1,0,1,1,1,1],
    1
    )
    cellCompete(*t)
    pass