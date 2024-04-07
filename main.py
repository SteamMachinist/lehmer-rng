import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from sympy import primerange


def create_generator(U0, M, C, p):
    U = U0

    def rng():
        nonlocal U
        while True:
            U = (U * M + C) % p
            yield U, U / p

    return rng


def generate_unique_sequence(U0, M, C, p):
    Us = set()
    Rs = []
    for U, R in create_generator(U0, M, C, p)():
        length = len(Us)
        Us.add(U)
        Rs.append(R)
        if length == len(Us):
            return np.array(Rs)


def test_6(R):
    counter = 0
    for i in range(0, len(R) - 1, 2):
        value = np.hypot(R[i], R[i + 1])
        if value < 1:
            counter += 1
    return 8 * counter / len(R)


def create_points(R, k):
    points_number = R.size // k
    return R[:points_number * k].reshape(-1, k)


def fill_buckets(points, bucket_boundaries, points_per_bucket):
    for point in points:
        bucket_indices = [np.digitize(point[i], bucket_boundaries) - 1 for i in range(len(point))]
        bucket_key = tuple(bucket_indices)
        points_per_bucket[bucket_key] = points_per_bucket.get(bucket_key, 0) + 1
    return points_per_bucket


def test_k_dims(R, k):
    q = 8
    p = 0.025

    points = create_points(R, k)

    bucket_boundaries = np.linspace(0, 1, q + 1)
    points_per_bucket = {}

    points_per_bucket = fill_buckets(points, bucket_boundaries, points_per_bucket)

    N = len(points)
    M = np.power(q, k)
    NdM = N / float(M)

    hi_squared_exp = np.sum([(mj - NdM) ** 2 for mj in points_per_bucket.values()]) / NdM
    hi_squared_ideal = chi2.ppf(p, M - 2)

    return hi_squared_exp - hi_squared_ideal


def generate_and_test(U0, M, C, p):
    R = generate_unique_sequence(U0, M, C, p)
    return len(R), R.mean(), np.median(R), R.std(), test_k_dims(R, 3), test_6(R)


def plot(param_name, x, lengths, means, medians, stds, k_dims, test6s):
    plt.plot(x, lengths)
    plt.title(param_name + " length")
    plt.show()
    plt.plot(x, means)
    plt.title(param_name + " mean")
    plt.show()
    plt.plot(x, medians)
    plt.title(param_name + " median")
    plt.show()
    plt.plot(x, stds)
    plt.title(param_name + " std")
    plt.show()
    plt.plot(x, k_dims)
    plt.title(param_name + " k_dim(3)")
    plt.show()
    plt.plot(x, test6s)
    plt.title(param_name + " test6")
    plt.show()


def append_test_values(lengths, means, medians, stds, k_dims, test6s, test_values):
    length, mean, median, std, k_dim, test6 = test_values
    lengths.append(length)
    means.append(mean)
    medians.append(median)
    stds.append(std)
    k_dims.append(k_dim)
    test6s.append(test6)


def test_for_U0():
    U0s = list(primerange(150, 5000))[::10]
    M = 631
    p = 22501

    lengths, means, medians, stds, k_dims, test6s = ([] for _ in range(6))
    for U0 in U0s:
        append_test_values(lengths, means, medians, stds, k_dims, test6s,
                           generate_and_test(U0, M, 0, p))

    plot("U0", U0s, lengths, means, medians, stds, k_dims, test6s)


def test_for_M():
    U0 = 883
    Ms = list(primerange(150, 5000))[::10]
    p = 22501

    lengths, means, medians, stds, k_dims, test6s = ([] for _ in range(6))
    for M in Ms:
        append_test_values(lengths, means, medians, stds, k_dims, test6s,
                           generate_and_test(U0, M, 0, p))

    plot("M", Ms, lengths, means, medians, stds, k_dims, test6s)


def test_for_p():
    U0 = 883
    M = 631
    ps = list(primerange(10000, 70000))[::50]

    lengths, means, medians, stds, k_dims, test6s = ([] for _ in range(6))
    for p in ps:
        append_test_values(lengths, means, medians, stds, k_dims, test6s,
                           generate_and_test(U0, M, 0, p))

    plot("p", ps, lengths, means, medians, stds, k_dims, test6s)


def error(mean, median, test_k_dims, test_6):
    return 0.25 * (np.abs(mean - 0.5)
                   + np.abs(median - 0.5)
                   + np.clip(test_k_dims, 0, None)
                   + np.abs(test_6 - np.pi))


def find_best_params(U0_range, M_range, p_range):
    best_params = None
    minimal_error = float('inf')

    for U0 in U0_range:
        for M in M_range:
            for p in p_range:
                lengths, mean, median, std, k_dim, test6 = generate_and_test(U0, M, 0, p)
                current_error = error(mean, median, k_dim, test6)
                if current_error < minimal_error:
                    minimal_error = current_error
                    best_params = (U0, M, p)

    return best_params, minimal_error


# test_for_U0()
# test_for_M()
# test_for_p()

# U0_range = list(primerange(150, 5000))[::60]
# M_range = list(primerange(150, 5000))[::60]
# p_range = list(primerange(30000, 60000))[::300]
#
# best_params, minimal_error = find_best_params(U0_range, M_range, p_range)
# print(f"Best parameters: {best_params}")
# print(f"Minimal error: {minimal_error}")
#
# print(generate_and_test(best_params[0], best_params[1], 0, best_params[2]))


# Best: (2267, 151, 0, 52291)
print(generate_and_test(2267, 151, 0, 52291))
