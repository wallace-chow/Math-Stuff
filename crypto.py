from random import *
from math import *
from time import *

class DH(object):
    def __init__(self, g, n):
        self.n = n
        self.g = g
        self.private_key = None
        self.public_key = None

    def generate_pub_priv(self, *args):
        n = self.n
        if not args:
            private_key = randint(2, n - 2)
            print('private_key', private_key)
            self.public_key = fast_exp(self.g, private_key, n) #h
        else:
            private_key = args[0]
            self.public_key = args[1]
        self.private_key = private_key
            
    def encrypt(m, other_pubkey):
        n = self.n
        secret = fast_exp(other_pubkey, self.private_key, n)
        
class Pohlig_attack(object):
    def __init__(self, person):
        # to return an x such that g*x = h mod p
        self.g = person.g
        self.h = person.public_key
        self.p = person.n

    def attack(self):
        g = self.g
        p = self.p
        h = self.h
        #print('g, p, h in attack', g, p, h)
        #factors = Number(p - 1).factorise()
        factors = [2, 55539525649117, 159563476023619, 180641133241909, 184042660526033, 281722834832897, 298629764679251, 313012934884873, 314443562221879, 315528670305557, 321835382158219, 328733044251319, 346020793060567, 364626576379843, 489004382860667, 505754365422919, 543346025904713, 578114950608101, 592474286953183, 610894712370133, 669315310186493, 707101203134057, 709992592536571, 727886679703067, 762683130279779, 778944610838227, 911972686930199, 923361591739577, 964238390129099, 1055803336959821, 1067373223356673]
        #factors = [2, 904193, 952859, 1746737, 4050647, 5320409, 6670933, 6734831, 8079917, 8618327, 9159259, 11259793, 13102151, 14333773, 14458439, 15716273, 16811807, 17026873, 17186801, 17704741, 19396127, 20208817, 24011297, 25480813, 27489817, 27828209, 28744543, 31580957, 31991893, 32121253, 32286217]
        
        a, m = [], []
        for prime_factor in factors:
            # factors is a dict
            #subgrp_order = prime_factor**factors[prime_factor]
            #print('subgrp_order', subgrp_order)
            subgrp_order = prime_factor
            M = (p - 1)//subgrp_order #lagrange theorem
            generator = fast_exp(g, M, p) # subgrp generator
            print('new generator', fast_exp(g, M, p))
            new_h = fast_exp(h, M, p)
            print('new_h', fast_exp(h, M, p))
            #print(generator, new_h)
            # brute force
##            temp = 1
##            for i in range(subgrp_order):
##                if i % 10000000 == 0:
##                    print('finding h in subgrp', subgrp_order, 'iteration', i)
##
##                if new_h == temp:
##                    a.append(i)
##                    m.append(subgrp_order)
##                    print('found h in subgrp', subgrp_order)
##                    break
##                temp = temp * generator % p

            ## pollard rho
            # with subgrp DLP new_h and generator
            x = self.pollard_function(generator, new_h, subgrp_order, p)
            if x < 0:
                print('pollard_function unable to find x')
            print('found x in subgroup', subgrp_order, 'which is', x)
            a.append(x)
            m.append(subgrp_order)
        print("list of a and m before CRT: ", a, m)
        system = System_of_congurencies(a, m)
        return system.CRT()

    def pollard_function(self, g, h, subgrp_order, p, retries = 100):
        '''
        uses primitives, g, h, p and computes by random walk
        Inputs xi, yi and function f, returns xi + 1, yi + 1]
        i will use a helper function that takes X and Y, compute their next terms,
        and put a and b in a dict, but before that i must check if there is a collision'''
        if g == h:
            return 1
        elif h == 1:
            return 0
        def make_dlp_iteration_function(g, h, subgrp_order, p, num_multipliers=16):
            """
            Creates an optimized iteration function for Pollard's Rho DLP
            Returns a function next_step(x, a, b)
            
            Parameters:
            g, h, p: DLP parameters (g^x ≡ h mod p)
            n: order of the subgroup (or p-1 if unknown)
            num_multipliers: number of precomputed multipliers (default 16)
            """
            # Precompute multipliers and exponent increments
            multipliers = []
            a_increments = []
            b_increments = []
            
            for _ in range(num_multipliers):
                a_inc = randint(1, subgrp_order - 1)
                b_inc = randint(1, subgrp_order - 1)
                multiplier = fast_exp(g, a_inc, p) * fast_exp(h, b_inc, p) % p
                
                multipliers.append(multiplier)
                a_increments.append(a_inc)
                b_increments.append(b_inc)
            
            def next_step(x, a, b):
                
                # Select multiplier based on current state
                selector = x % num_multipliers
                m = multipliers[selector]
                a_inc = a_increments[selector]
                b_inc = b_increments[selector]
                
                # Update group element and exponents
                new_x = (x * m) % p
                new_a = (a + a_inc) % subgrp_order
                new_b = (b + b_inc) % subgrp_order
                
                return (new_x, new_a, new_b)
            
            return next_step
            
##        def helper(X, X_a, X_b):
##            case = X % 3
##
##            if case == 0:
##                # hx, update a, b
##                X = X * h % p
##                X_b = (X_b + 1) % (subgrp_order)
##            elif case == 1:
##                # x**2
##                X = X * X % p
##                X_a = (X_a *2) % (subgrp_order)
##                X_b = (X_b *2) % (subgrp_order)
##            elif case == 2:
##                # gx'
##                X = X * g % p
##                X_a = (X_a + 1) % (subgrp_order)
##                
##            #print(X, X_a, X_b)
##            return (X, X_a, X_b)
##        #print('now testing subgrp with g and h', g, h)

        # making helper function
        helper = make_dlp_iteration_function(g, h, subgrp_order, p)
        for _ in range(retries):
            a = randint(0, p - 2)
            b = randint(0, p - 2)
            x = ((fast_exp(g, a, p) * fast_exp(h, b, p))% p, a, b)
            y = x
            k = 0

            for i in range(10**100000):
                if k % 10**6 == 0:
                    print('on iter in pollard', k)
                x = helper(*x)
                y = helper(*y)
                y = helper(*y)
                k += 1
                if x[0] == y[0] and x[1:] != y[1:]:
                    # we found a collision
                    # where a1 - b1 = k(b2 - a2)
                    a1, a2 = x[1:]
                    b1, b2 = y[1:]
                    print("collision found mod", subgrp_order,"a1:", a1, "a2", a2, 'b1', b1, 'b2', b2)
                    r = (a1 - b1) % subgrp_order
                    if b2 == a2:
                        break
                    l = (b2 - a2) % (subgrp_order)
                    
##                    gcd_check = gcd(l, subgrp_order)
##                    l //= gcd_check
##                    subgrp_order //= gcd_check
                    
                    if l == r:
                        return 1
                    inv_l = fast_exp(l, subgrp_order-2, subgrp_order)  # Fermat's inverse
                    x_candidate = (r * inv_l) % subgrp_order
                    print(x_candidate)
                    if fast_exp(g, x_candidate, p) == h:  # Verify
                        return x_candidate
            print('unable to find x for dlp involving g: ', g, 'h: ', h)
    def attack2(self):
        p = self.p
        g = self.g
        h = self.h

        x = self.pollard_function(g, h, p, p)
        return x

    def hensel (self, g, h, N, p, n):
        '''
        solves g**x = h mod N for subgrp p**n
        '''
        subgrp_order = fast_exp(p, n, N)
        result = 0
        a = 1
        for i in range(n):
            mini_order = fast_exp(p, i + 1, N)
            new_g = (fast_exp(g, N//mini_order, N) * fast_exp(a, N - 2, N)) % N
            new_h = fast_exp(h, N//mini_order, N)
            # a is mod p
            # trying to solve a0 in (g**(N/p_i))**a0 = y**(N/p_i)
            a = self.pollard_function(new_g, new_h, mini_order, N)
            result += a * fast_exp(p, i, N)
        return result
        

def extended_gcd(a, b):
    """Returns (gcd, x, y) such that ax + by = gcd(a, b)"""
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return (g, x, y)
def gcd(a, b):
    if a == 0: return b
    if b == 0: return a

    # Find power of two divisor for both a and b
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1

    while (a & 1) == 0:
        a >>= 1

    while b != 0:
        while (b & 1) == 0:
            b >>= 1

        if a > b:
            a, b = b, a

        b = b - a

    return a << shift

def modinv(a, m):
    """Finds the modular inverse of a mod m, if it exists"""
    # ax = 1 mod m
    # ax = 1 + km
    # ax + km = 1
    a = a%m
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse exists for {a} mod {m}")
    return x % m

def fast_exp(a, b, m):
    result = 1
    a = a % m  # Ensure base is within mod
    while b > 0:
        if b % 2 == 1:  # If exponent is odd
            result = (result * a) % m
        a = (a * a) % m  # Square the base
        b = b // 2  # Halve the exponent    
    return result

class Number(object):
    def __init__(self, N):
        self.n = N
        self.factors = {}

    def get_n(self):
        return self.n

    def miillerTest(self, d):
        
        n = self.n
        # Pick a random number in [2..n-2]
        # Corner cases make sure that n > 4
        a = 2 + randint(1, n - 4);

        # Compute a^d % n
        x = fast_exp(a, d, n);

        if (x == 1 or x == n - 1):
            return True;
        while (d != n - 1):
            x = (x * x) % n;
            d *= 2;

            if (x == 1):
                return False;
            if (x == n - 1):
                return True;

        # Return composite
        return False;

    def is_prime(self, *args, k = 20):
        if not args:
            n = self.n
        else:
            n = args[0]
        # Corner cases
        if (n <= 1 or n == 4):
            return False;
        if (n <= 3):
            return True;

        # Find r such that n = 
        # 2^d * r + 1 for some r >= 1
        d = n - 1;
        while (d % 2 == 0):
            d //= 2;
        number = Number(n)
        # Iterate given number of 'k' times
        for i in range(k):
            if (number.miillerTest(d) == False):
                return False;

        return True;


    def factorise(self, lower_bound = 10000):
        # 1. Trial division
        N = self.n
        for i in range(2, min(lower_bound, N + 1)):
            if N % i == 0:
                r = 0
                while N % i == 0:
                    N //= i
                    r += 1
                self.factors[i] = self.factors.get(i, 0) + r
        while not self.is_prime(N) and N != 1:
            d = self.pollards_rho(N)
            r = 0
            if d == None:
                continue
            while N % d == 0:
                N //= d
                r += 1
            self.factors[d] = self.factors.get(d, 0) + r
        if N != 1:
            self.factors[N] = 1 
        return self.factors
            
            
    def pollards_rho(self, n, max_retries=100):
        """Pollard’s Rho with random function parameters and retries."""
        for _ in range(max_retries):
            x0 = randint(2, n - 1)
            c = randint(1, n - 1)
            f = lambda x: (x * x + c) % n
            x, y, d = x0, x0, 1
            prod = 1 
            for i in range(10**100000): 
                y = f(y) 
                y = f(y) 
                x = f(x) 
                prod = (prod * (y-x)) % n 
                if i % 100000 == 0: 
                    print(i) 
                    g = gcd(prod, n) 
                    if g>1 and g != n: 
                        print("Found after %d iterations." % i) 
                        return g 
        return None  # All attempts failed
    def is_generator(self, g):
        """Check if g is a generator of Z_p*"""
        ## must be factorised before using
        p = self.n # prime
        factors = self.factors
        for q in factors:
            if fast_exp(g, (p - 1) // q, p) == 1:
                return False
        return True
    
    def find_random_generator(self):
        if not self.is_prime():
            raise ValueError("p must be prime")
        p = self.n
        phi = p - 1
        factors = Number(p - 1).factorise()
        self.factors = factors
        print(factors)

        while True:
            g = randint(2, p - 2)
            print(g)
            if self.is_generator(g):
                return g

        raise Exception("No generator found")

    def find_safe_generator(self):
        """Finds a generator of subgroup of order q in Z_p*"""
        self.n = p
        q = (p - 1)/2
        while True:
            h = randrange(2, p - 1)
            g = fast_exp(h, 2, p)
            if fast_exp(g, q, p) == 1 and g != 1:
                return g



class System_of_congurencies(object):
    def __init__(self, a1, m1):
        self.a1= a1
        self.m1 = m1

    def CRT(self):
        """Solves x ≡ a_i mod m_i for pairwise coprime m_i using CRT"""
        a_list = self.a1
        m_list = self.m1
        assert len(a_list) == len(m_list)
        
        M = 1
        for m in m_list:
            M *= m

        x = 0
        for a_i, m_i in zip(a_list, m_list):
            M_i = M // m_i
            inv = modinv(M_i, m_i)  # Use extended Euclidean algorithm
            x += a_i * M_i * inv
            #print('ai, M_i and inv of M_i mod m_i in CRT ', a_i, M_i, inv)
        return x % M
    
def generate_safe_prime(n_digits):
    """Generate an n-digit prime number"""
    lower = 10**(n_digits - 1) // 2
    upper = 10**n_digits // 2

    while True:
        q = randrange(lower, upper)
        if q % 2 == 0:
            q += 1
        if Number(q).is_prime():
            p = 2 * q + 1
            if Number(p).is_prime():
                return p

def generate_large_prime(n_digits):
    lower = 10**(n_digits - 1) // 2
    upper = 10**n_digits // 2

    while True:
        q = randrange(lower, upper)
        if q % 2 == 0:
            q += 1
        if Number(q).is_prime():
            return q

#n1 = Number(generate_large_prime(3))
#n2 = Number(generate_large_prime(50))
#alice = DH(n1.find_random_generator(), n1.get_n())
#bob = DH(1, n2.get_n())
#alice.generate_pub_priv()
#print('alice info', 'n', alice.n, 'g', alice.g, 'h', alice.public_key)
#bob.generate_pub_priv()
#toattack = Pohlig_attack(alice)
       
alice = DH(2, 76828866009009212758379411732236880241731224275724865758121309075234297534316757630232811971827547228845365201584968114050004834185594477774482598001417546542543966098598459854358962990680293891276660066293274572660945959880795920918231558341849294134407490986663189370858201669472605949791371242794635882707881837762009802373716460543625315172594883828187845726055022852016649003150488593071162533508912157035945619704940396608039902552187)
#alice = DH (2, 429863736707892171326248553526931473966003433021201832015526216802471642975106774184360295252997897124116654776394887753501547995303667526808009184161651517432747827172161845085950734684687093782158361310843822867)
#alice = DH(2, 31)
alice.generate_pub_priv(None, 3)
toattack = Pohlig_attack(alice)
starttime = time()
x = toattack.attack()
endtime = time()
print(endtime - starttime)
print(x)

#print(fast_exp(2, 249893561005597658109886726392164142556084843459540779820362710339525663939076856273828622283325240320164546437789592428521921734620242570051347484888930718922045016851968222018504470298831932814470594997243987422, \
