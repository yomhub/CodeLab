
def solution(max_cnt,budget,costs,profits):
    ans = 0
    # dp = [[0]*(budget+1) for _ in range(len(costs))]
    dp = [[0,float(i)] for i in range((budget+1))]
    for i in range(len(costs)):
        for k in range(1,max_cnt+1):
            cst = costs[i]
            cst_i = int(cst)
            pf = profits[i]
            for j in range(budget,cst_i-1,-1):
                if(dp[int(dp[j][1]-cst)][0]+pf>dp[j][0]):
                    dp[j][0]=dp[int(dp[j][1]-cst)][0]+pf
                    if(int(dp[int(dp[j][1]-cst)][1]+cst)>j):
                        print('err')
                    dp[j][1]=dp[int(dp[j][1]-cst)][1]+cst

                
    return int(dp[-1][0])
    
if __name__ == "__main__":
    args = input().split()
    n,max_cnt,budget = int(args[0]),int(args[1]),int(args[2])
    cs = input().split()
    ps = input().split()
    costs = [float(o) for o in cs]
    profits = [(float(o)*costs[i])/100 for i,o in enumerate(ps)]
    
    print(solution(max_cnt,budget,costs,profits))
    pass