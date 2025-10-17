# f - partition function that outputs a tuple of (b1, b2, b3,..) for
# each permutation cycle type
# g - the function that takes a permutation, and calculates the number of
# conjugates
# i will introduce a function that will compute the fix(g) for every
# conjugacy class
# furthermore, the main function

def count_graphs(N):
    # part 1: get all cycle types (b1, b2, b3..., bn)
    # count all conjugates by combinatoric approach, i.e: n!, then overcounted by a factor of product of bk, furthermore,
    # overcounted by product factor of k**bk
    memo_fact = {}
    def fact_memo(n):
        if n == 0:
            return 1
        if n not in memo_fact:
            memo_fact[n] = fact_memo(n - 1)*n
        return memo_fact[n]

    def count_conjugates(args):
        res = fact_memo(N)
        for key in args:
            res //= fact_memo(args[key])* pow(key, args[key])
        return int(res)
            
    def integer_partitions_with_cycle_counts(n):
        def _partition(n, max_part, path, result):
            if n == 0:
                cycle_counts = {}  # hit a leaf
                for num in path:
                    if num in cycle_counts:
                        cycle_counts[num] += 1
                    else:
                        cycle_counts[num] = 1
                result.append(cycle_counts)
                return
            for part in range(min(max_part, n), 0, -1):
                # first search n with remainder 0:
                # then search n - 1 with remainder 1
                # then search n - 2 with remainder 2.. => after n,2, pop 2 and
                # search n-2 and 1=> search n-2,1 and remainder 1
                # then lets check for repeats, lets say i already have (5, 3, 2)
                # then the next search would be (5, 3) and trying to add 1s
                # which would lead to (5, 3, 1, 1)
                # then the next number to search would be (5, 2), but since the inner loop
                # is from min(n, 2) to 1, then 3 will never be repeated again.
                
                path.append(part)
                _partition(n - part, part, path, result)
                path.pop()
        
        result = []
        n_max = n  # Maximum possible part is n
        _partition(n, n, [], result)
        return result
    # TODO: Partition integer with DP
##    def integer_partitions_with_cycle_counts(n):
##        DP = [[1]]
##        for i in range(n):
##            DP[i].append[1]
##            for j in range(i):
##                pass
    
    # TODO: CHOOSE FUNCTION
    memo_choose = {}
    def NCHOOSEM(N, M):
        if M > N:
            return 0
        if M == 0:
            return 1
        if M == 1:
            return N
        if (N, M) in memo_choose:
              return memo_choose[(N, M)]
        memo_choose[(N, M)] = NCHOOSEM(N - 1, M - 1) + NCHOOSEM(N - 1, M)
        return memo_choose[(N,M)]
    # TODO: GCD FUNCTION:

    # TODO: FIX OF G CALCULATOR
    def fix(permutation):
        res = 0
        for cycle_len in permutation:
            res += permutation[cycle_len]*(cycle_len//2)
        length_types = list(permutation.keys())
        for i in range(len(length_types) - 1): # matching cycle length to another unique one'
            # at n = 0, iter j  from [1, the end]
            for j in range(i + 1, len(length_types)):
                total_combinations_of_i_j = permutation[length_types[i]] * permutation[length_types[j]]
                res += int(gcd(length_types[i], length_types[j]))*total_combinations_of_i_j
        for length in length_types:
            if permutation[length] > 1:
                res += NCHOOSEM(permutation[length], 2) * length
        return res
                
    #MAIN FUNCTION
    res = 0
    # first i need to get total number of edges, let F be the choose function
    TOTAL_EDGES = NCHOOSEM(N, 2)
    SIZE_Sn = fact_memo(N)
    # then i obtain the tuple of tuples that give the cycle type
    CYCLE_TYPES = integer_partitions_with_cycle_counts(N)
    # count number of conjugates per cycle type, then multiply by its fix, add to res
    for cycle in CYCLE_TYPES:
        res += count_conjugates(cycle) * 2**fix(cycle)
    return int(res//SIZE_Sn)

def count_partitions(N):
    dp = [[0 for i in range(N)] for k in range(N)]
    for n in range(N):
        for k in range(n + 1):
            if k == 0 or n == k:
                dp[k][n] = 1
            else:
                dp[k][n] = dp[k - 1][n - 1] + dp[k][n - k - 1]
    res = 0
    for i in range(N):
        res += dp[i][N - 1]
    print(dp)
    return res
        
        

print(count_partitions(1000))

        
        
        





