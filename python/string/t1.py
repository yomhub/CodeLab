import math
from bisect import bisect_right
def solution(nums):
    
    nums.sort()
    total = sum(nums)
    if(nums[-1]==2):
        return total
    sp=bisect_right(nums,2)
    ans=sum(nums[:sp])
    while(True):
        ep = len(nums)-1
        cp=ep-1
        tmp_l=1
        while(cp>=sp):
            if(nums[cp]==nums[ep]):
                cp-=1
                tmp_l+=1
                continue
            else:
                break
        if(tmp_l==1):
            nums[ep]-=1
            ans+=1



    
    while(True):
        tmp = [o//2 for o in nums]
        if(max(tmp)==0):
            break
        numsd2.append(tmp)
    
    return ans
    
if __name__ == "__main__":
    n = int(input())
    nums=[]
    for i in range(n):
        cs = input()
        cs=cs.split(':')
        nums.append(int(cs[-1]))
    print(solution(nums))
    pass