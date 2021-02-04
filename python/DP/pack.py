def profitableSchemes(G: int, P: int, group: list, profit: list) -> int:

    dp_list = [[0 for i in range(G+1)] for j in range(P+1)]
    dp_list[0][0]=1
    for i,o in enumerate(group): 
        # for j in range(P,-1,-1):
        for j in range(0,P+1):
            # for k in range(G-o,-1,-1):
            for k in range(0,G-o):
                # 
                dp_list[min(P,profit[i]+j)][k+o]+=dp_list[j][k]

    cnt=0
    for i in range(1,G+1):
        cnt+=dp_list[P][i]

    return cnt

if __name__ == "__main__":
    tt=(
        10,
        1,
        [6,3,6,1,10,1,11,6,8,8,11,10,9,10,4,7,9,6,7,9,10,8,4,6,7,7,9,4,4,4,8,6,7,10,5,2,1,6,11,3,8,9,3,2,8,4,7,10,9,5,3,6,10,4,5,4,10,3,8,6,11,10,6,9,8,11,3,7,2,7,7,9,7,10,1,3,3,9,6,3,11,3,5,10,9,4,10,6,4,10,9,2,1,1,9,10,5,10,7,6],
        [2,0,0,1,2,0,0,1,2,1,1,2,2,2,1,0,2,2,1,1,0,0,2,2,0,2,2,2,0,1,2,1,1,0,0,2,2,2,2,0,0,0,0,2,0,0,1,0,2,1,0,2,0,0,1,2,2,1,1,2,1,1,2,0,2,0,0,1,1,1,0,1,1,2,2,1,0,0,1,0,2,2,1,2,2,0,0,2,0,2,2,1,0,2,0,1,0,1,0,2],
    )
    tt=(
        5,
        3,
        [2,2],
        [2,3],
    )

    print(profitableSchemes(*tt))
    
    pass