from random import *
from math import *
def extended_gcd(a, b):
    """Returns (gcd, x, y) such that ax + by = gcd(a, b)"""
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return (g, x, y)

def gcd(m, n):
    while n:
        m, n = n, m % n
    return m


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


# output is a dict of 'prime factor':'power'
print(Number(23023728963).factorise())
print(Number(12038726).factorise())
