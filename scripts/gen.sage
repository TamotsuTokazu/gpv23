def gen_prime(N, m, q=64):
    m = lcm(m, q)
    N = N // m * m + 1;
    while not is_prime(N):
        N -= m
        if N < 0:
            raise ValueError('No prime found')
    g = primitive_root(N)
    return N, g


pp = 97
print(primitive_root(pp))
p = 12289
m = p * (p - 1) * pp
N, g = gen_prime(2^49, m)
print(factor(N - 1))
print(f'{N}LL, {g}LL, {p}, {primitive_root(p)}')

N -= m
N, g = gen_prime(N, m)
print(factor(N - 1))
print(f'{N}LL, {g}LL, {p}, {primitive_root(p)}')

N -= m
N, g = gen_prime(N, m)
print(factor(N - 1))
print(f'{N}LL, {g}LL, {p}, {primitive_root(p)}')

N -= m
N, g = gen_prime(N, m)
print(factor(N - 1))
print(f'{N}LL, {g}LL, {p}, {primitive_root(p)}')