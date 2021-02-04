
def ispalindromic(s):
    for i in range(len(s)//2):
        if(s[i]!=s[-i-1]):
            return False
    return True
def solution(s,lens):
    ans =0
    for det in lens:
        if(det==1):
            ans+=len(s)
            continue
        for i in range(len(s)-det+1):
            if(ispalindromic(s[i:i+det])):
                ans+=1
    return ans
    
if __name__ == "__main__":
    n = int(input())
    s = input()
    n = int(input())
    cs = input().split()
    lens=[int(o) for o in cs]
    print(solution(s,lens))
    pass