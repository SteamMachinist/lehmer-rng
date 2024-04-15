import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from sympy import primerange

# Датчик №7
def create_generator(R0, z0, n):
    R = R0
    z = z0

    def rng():
        nonlocal R
        nonlocal z
        while True:
            z = z + 10 ** float(-n)
            temp = R / z + np.pi
            R = temp - int(temp)
            yield R

    return rng


def generate_unique_sequence(R0, z0, n):
    Rs = set()
    for R in create_generator(R0, z0, n)():
        length = len(Rs)
        Rs.add(R)
        if length == len(Rs) or length > 50000:
            return np.array(list(Rs))


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


def generate_and_test(R0, z0, n):
    R = generate_unique_sequence(R0, z0, n)
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


def test_for_R0():
    R0s = np.linspace(0, 0.999, 100)
    z0 = 0.092
    n = 7

    lengths, means, medians, stds, k_dims, test6s = ([] for _ in range(6))
    for R0 in R0s:
        append_test_values(lengths, means, medians, stds, k_dims, test6s,
                           generate_and_test(R0, z0, n))

    plot("R0", R0s, lengths, means, medians, stds, k_dims, test6s)


def test_for_z0():
    R0 = 0.312
    z0s = np.linspace(0, 0.999, 100)
    n = 7

    lengths, means, medians, stds, k_dims, test6s = ([] for _ in range(6))
    for z0 in z0s:
        append_test_values(lengths, means, medians, stds, k_dims, test6s,
                           generate_and_test(R0, z0, n))

    plot("z0", z0s, lengths, means, medians, stds, k_dims, test6s)


def test_for_n():
    R0 = 0.312
    z0 = 0.092
    ns = np.linspace(1, 40, 40)

    lengths, means, medians, stds, k_dims, test6s = ([] for _ in range(6))
    for n in ns:
        append_test_values(lengths, means, medians, stds, k_dims, test6s,
                           generate_and_test(R0, z0, n))

    plot("n", ns, lengths, means, medians, stds, k_dims, test6s)


def error(mean, median, test_k_dims, test_6):
    return 0.25 * (np.abs(mean - 0.5)
                   + np.abs(median - 0.5)
                   + np.clip(test_k_dims, 0, None)
                   + np.abs(test_6 - np.pi))


def find_best_params(R0_range, z0_range, n_range):
    best_params = None
    minimal_error = float('inf')

    for R0 in R0_range:
        print(R0)
        for z0 in z0_range:
            for n in n_range:
                lengths, mean, median, std, k_dim, test6 = generate_and_test(R0, z0, n)
                current_error = error(mean, median, k_dim, test6)
                if current_error < minimal_error:
                    minimal_error = current_error
                    best_params = (R0, z0, n)

    return best_params, minimal_error


# test_for_R0()
# test_for_z0()
# test_for_n()

# n >= 5, иначе плохо
# маленькие z < 0.2 в среднем лучше, но зависит от n
# для R есть удачные значения

# R0_range = np.linspace(0, 0.999, 15)
# z0_range = np.linspace(0, 0.999, 15)
# n_range = np.arange(5, 11, 1)
#
# best_params, minimal_error = find_best_params(R0_range, z0_range, n_range)
# print(f"Best parameters: {best_params}")
# print(f"Minimal error: {minimal_error}")
#
# print(generate_and_test(best_params[0], best_params[1], best_params[2]))


# Best: (0.9276428571428571, 0.1427142857142857, 9)
print(generate_and_test(0.9276428571428571, 0.1427142857142857, 9))
