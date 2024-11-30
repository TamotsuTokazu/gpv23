def gen_prime(N, m, q=64):
    m = lcm(m, q)
    print(m, m.bit_length())
    N = N // m * m + 1;
    while not is_prime(N):
        N -= m
        if N < 0:
            raise ValueError('No prime found')
    g = primitive_root(N)
    return N, g

p = 7
m = p * (p - 1)
N, g = gen_prime(4294960321 - 2, m)
print(factor(N - 1))

print(f'{N}LL, {g}LL, {p}, {primitive_root(p)}')