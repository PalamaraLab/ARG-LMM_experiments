"""python -m pytest -v -s test_utils.py"""

import msprime
import numpy as np
import pytest

from utils import create_list_if_not_exists, modify_data_return_new_ts

def test_create_list_if_not_exists():
    results = {}
    with pytest.raises(AssertionError):
        create_list_if_not_exists(results, "aaa")
    with pytest.raises(AssertionError):
        create_list_if_not_exists(results, [])

    create_list_if_not_exists(results, ["foo"])
    with pytest.raises(Exception, match="Expected to see a dict"):
        create_list_if_not_exists(results, ["foo", "bar"])

    create_list_if_not_exists(results, ["bar", "baz", "foo"])
    create_list_if_not_exists(results, ["bar", "baz", "oo"])
    with pytest.raises(Exception, match="Expected to see a list"):
        create_list_if_not_exists(results, ["bar", "baz"])

    create_list_if_not_exists(results, ["bar", -1])
    with pytest.raises(Exception, match="Expected to see a dict"):
        create_list_if_not_exists(results, ["bar", -1, 'a'])

    create_list_if_not_exists(results, ["bar", 0, 'a'])
    create_list_if_not_exists(results, ["bar", 0, 'b'])
    create_list_if_not_exists(results, ["bar", 0, 'c'])

    # check that we can append to what is returned
    create_list_if_not_exists(results, ["bar", 0, 'd']).append("hello")
    create_list_if_not_exists(results, ["bar", 0, 'd']).append("world")

    expected = {
        'foo': [],
        'bar': {
            'baz': {'foo': [], 'oo': []},
            -1: [],
            0: {'a': [], 'b': [], 'c': [], 'd': ['hello', 'world']}
        }
    }
    assert str(results) == str(expected)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("sample_size", [100])
@pytest.mark.parametrize("length", [5e3])
@pytest.mark.parametrize("error_rate", [0, 1e-4, 1e-3, 1e-2, 1e-1])
def test_add_errors_return_new_ts(seed, sample_size, length, error_rate):
    ts1 = msprime.simulate(
        sample_size=sample_size, Ne=4e4, length=length, recombination_rate=2e-8,
        mutation_rate=2e-8, random_seed=seed)

    ts2 = modify_data_return_new_ts(ts1, error_rate=error_rate, random_seed=seed)

    ts1_geno = ts1.genotype_matrix()
    ts2_geno = ts2.genotype_matrix()

    assert ts1_geno.shape == ts2_geno.shape
    diff_geno = ts1_geno ^ ts2_geno
    num_diffs = np.sum(diff_geno)

    n = diff_geno.size
    p = error_rate
    q = 1 - p
    binomial_mean = n*p
    binomial_std = np.sqrt(n*p*q)

    # Use CLT to approximate binomial as a Gaussian
    # print(num_diffs, n, p, binomial_mean, binomial_std)
    # 0.3% chance of this failing for any one test
    assert np.abs(num_diffs - binomial_mean) <= 3*binomial_std
    # replacing 3 by 1.5 should break the test
    # assert np.abs(num_diffs - binomial_mean) <= 1.5*binomial_std


@pytest.mark.parametrize("seed", [1, 2, 10])
@pytest.mark.parametrize("sample_size", [20])
@pytest.mark.parametrize("length", [5e3])
@pytest.mark.parametrize("permutation_seed", [42, 43, 44])
def test_permute_data_return_new_ts(seed, sample_size, length, permutation_seed):
    ts1 = msprime.simulate(
        sample_size=sample_size, Ne=4e4, length=length, recombination_rate=2e-8,
        mutation_rate=2e-8, random_seed=seed)

    rng = np.random.default_rng(permutation_seed)
    permutation = rng.permutation(sample_size)
    ts2 = modify_data_return_new_ts(ts1, permutation=permutation)

    ts1_geno = ts1.genotype_matrix()
    ts2_geno = ts2.genotype_matrix()
    ts1_geno_permuted = ts1_geno[:, permutation]

    assert not np.array_equal(ts1_geno, ts2_geno)
    assert np.array_equal(ts1_geno_permuted, ts2_geno)
