from contextlib import contextmanager
import errno
import gc
import gzip
import logging
import msprime
import numpy as np
import os
from packaging import version
import psutil; process = psutil.Process(os.getpid())
import shutil
import subprocess
import sys
import time
import tskit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# https://www.pythoncentral.io/measure-time-in-python-time-time-vs-time-clock/
if sys.platform == 'win32':
    # On Windows, the best timer is time.clock
    default_timer = time.clock
else:
    # On most other platforms the best timer is time.time
    default_timer = time.time
# from timeit import default_timer  # we don't want this as it disables gc


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def prepend_current_dir(path):
    return os.path.join(CURRENT_DIR, path)


def check_paths(paths_to_check):
    assert isinstance(paths_to_check, list)
    for path_to_check in paths_to_check:
        # Thanks to https://stackoverflow.com/q/36077266
        if not os.path.exists(path_to_check):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_check)


def btime_default_f(time_in_seconds):
    print("Time elapsed (seconds):", time_in_seconds)


@contextmanager
def btime(f=btime_default_f):
    """Times a block of code and applies function f to the resulting time in seconds.

    Inspired by https://stackoverflow.com/a/30024601.
    """
    start = default_timer()
    yield
    end = default_timer()
    time_in_seconds = end - start
    f(time_in_seconds)


def collect_garbage(sleep_seconds=5.):
    gc.collect()
    time.sleep(sleep_seconds)
    logging.info("Ran garbage collection")
    memory_in_bytes = process.memory_info().rss
    logging.info("Memory: {}".format(process.memory_info().rss))
    return memory_in_bytes


def create_list_if_not_exists(recursive_dict, key_list):
    assert isinstance(recursive_dict, dict)
    assert isinstance(key_list, list)
    assert len(key_list) > 0

    next_key = key_list[0]
    if len(key_list) == 1:
        if next_key not in recursive_dict:
            recursive_dict[next_key] = []
        elif not isinstance(recursive_dict[next_key], list):
            raise Exception("Expected to see a list, instead got {}".format(recursive_dict[next_key]))
        return recursive_dict[next_key]
    else:
        if next_key not in recursive_dict:
            recursive_dict[next_key] = {}
        elif not isinstance(recursive_dict[next_key], dict):
            raise Exception("Expected to see a dict, instead got {}".format(recursive_dict[next_key]))
        return create_list_if_not_exists(recursive_dict[next_key], key_list[1:])


def run_sequential_augment(sample_data, num_threads=0, random_seed=1):
    # Based on
    # https://github.com/mcveanlab/treeseq-inference/blob/master/human-data/tsutil.py
    # Also helpful was
    # https://github.com/tskit-dev/tsinfer/pull/112/files
    import tsinfer
    num_samples = sample_data.num_samples

    ancestors = tsinfer.generate_ancestors(sample_data, num_threads=num_threads)
    # other arguments are recombination_rate, mismatch_ratio, and path_compression
    # https://tsinfer.readthedocs.io/en/latest/inference.html#matching-ancestors-samples
    ancestors_ts = tsinfer.match_ancestors(sample_data, ancestors, num_threads=num_threads)

    # Compute the total samples required.
    n = 2
    total = 0
    while n < num_samples // 4:
        total += n
        n *= 2

    np.random.seed(random_seed)
    samples = np.random.choice(np.arange(num_samples), size=total, replace=False)

    n = 2
    j = 0
    while n < num_samples // 4:
        subset = samples[j:j + n]
        subset.sort()
        ancestors_ts = tsinfer.augment_ancestors(
            sample_data, ancestors_ts, subset, num_threads=num_threads)
        j += n
        n *= 2

    final_ts = tsinfer.match_samples(
        sample_data, ancestors_ts, num_threads=num_threads, simplify=False)
    logging.info("Tables may be unsorted for sequential augment tsinfer, so sorting now")
    tables = final_ts.dump_tables()
    tables.sort() # added by brianzhang01, not in tsutil version
    final_ts = tables.tree_sequence()
    logging.info("Removing unary nodes. Currently {} nodes, {} edges".format(final_ts.num_nodes, final_ts.num_edges))
    final_ts = final_ts.simplify(keep_unary=False) # added by brianzhang01, not in tsutil version
    logging.info("Done removing unary nodes. Now {} nodes, {} edges".format(final_ts.num_nodes, final_ts.num_edges))
    return final_ts


def wrapped_tsinfer(simulation, snp_ids=None, keep_unary=False, array_strategy=False):
    import tsinfer
    # https://github.com/tskit-dev/tsinfer/issues/194
    with tsinfer.SampleData(sequence_length=simulation.sequence_length) as sample_data:
        snp_ids_index = 0
        for i, variant in enumerate(simulation.variants()):
            if snp_ids is None or snp_ids[snp_ids_index] == i:
                sample_data.add_site(variant.site.position, variant.genotypes)
                snp_ids_index += 1
                if snp_ids is not None and snp_ids_index == len(snp_ids):
                    break

    if array_strategy:
        if keep_unary:
            raise ValueError("Keeping unary nodes for array strategy is not supported")
        # Note: this will always remove unary nodes
        ts = run_sequential_augment(sample_data, num_threads=0, random_seed=1)
    else:
        # In this case, unary nodes exist in standard inference
        # See https://github.com/tskit-dev/tsinfer/issues/146
        if (version.parse(tsinfer.__version__) >= version.parse("0.2.0")):
            # other arguments are recombination_rate, mismatch_ratio, and path_compression
            # https://tsinfer.readthedocs.io/en/latest/inference.html#matching-ancestors-samples
            ts = tsinfer.infer(sample_data)
            if not keep_unary:
                logging.info("Removing unary nodes. Currently {} nodes, {} edges".format(ts.num_nodes, ts.num_edges))
                ts = ts.simplify(keep_unary=False) # added by brianzhang01, not in tsutil version
                logging.info("Done removing unary nodes. Now {} nodes, {} edges".format(ts.num_nodes, ts.num_edges))
        # In this case, unary nodes do not exist in standard inference, so we error if that is requested
        else:
            if keep_unary:
                raise ValueError("For unary nodes, upgrade to tsinfer >= 0.2.0")
            else:
                # other arguments are recombination_rate, mismatch_ratio, and path_compression
                # https://tsinfer.readthedocs.io/en/latest/inference.html#matching-ancestors-samples
                ts = tsinfer.infer(sample_data)
    return ts


def wrapped_relate(relate_bin_dir, relate_tmp_dir, simulation, condition=0,
                   num_relate_samples=None, snp_ids=None):
    """Run Relate on a simulation

    Two key ideas here: 1) write our own output format where we round to
    integer but also enforce strictly increasing positions, 2) calling Relate
    through a subprocess.

    We previously had a version that wrote VCFs, but that didn't allow us to
    play with the strictly increasing rounding, so writing haps instead.

    Set parameter is mu = 1.65e-8. condition allows for other simulation options.
    """
    if num_relate_samples is None:
        num_relate_samples = simulation.num_samples
    if num_relate_samples > simulation.num_samples:
        raise ValueError("Simulation does not have this many samples")
    if num_relate_samples % 2 != 0:
        raise ValueError("Even number of samples expected")

    logging.info("Using temporary directory " + relate_tmp_dir)
    os.makedirs(relate_tmp_dir, exist_ok=True)

    in_prefix = "in"
    out_prefix = "out"
    haps_path = os.path.join(relate_tmp_dir, in_prefix + ".haps")
    snp_positions = []
    with open(haps_path, 'w') as out_file:
        last = -1
        snp_ids_index = 0
        for i, variant in enumerate(simulation.variants()):
            if snp_ids is None or snp_ids[snp_ids_index] == i:
                pos = int(variant.site.position)
                # in contrived cases, this might overflow the bounds of the genome
                if pos <= last:
                    pos = last + 1
                snp_positions.append(pos)
                last = pos
                row_list = ['1', '.', str(pos), '0', '1']
                row_list += [str(entry) for entry in variant.genotypes[:num_relate_samples]]
                out_file.write(' '.join(row_list))
                out_file.write('\n')

                snp_ids_index += 1
                if snp_ids is not None and snp_ids_index == len(snp_ids):
                    break

    sample_path = os.path.join(relate_tmp_dir, in_prefix + ".sample")
    with open(sample_path, 'w') as out_file:
        out_file.write('\t'.join(["ID_1", "ID_2", "missing"]) + '\n')
        out_file.write('\t'.join(["0", "0", "0"]) + '\n')
        for i in range(num_relate_samples // 2):
            out_file.write('\t'.join(["sample_" + str(i), "sample_" + str(i), "0"]) + '\n')

    relate_ceu_script = prepend_current_dir("relate_ceu.sh")
    relate_const_script = prepend_current_dir("relate_const.sh")
    ceu_diploid_file = prepend_current_dir("ceu2.coal")
    demo_argument = ceu_diploid_file
    map_file = prepend_current_dir("genetic_map_12e-9.txt")
    if condition == 0:
        relate_script = relate_ceu_script
    elif condition == 1:
        relate_script = relate_const_script
        demo_argument = "30000"
    elif condition == 2:
        relate_script = relate_const_script
        demo_argument = "20000"
    # Previous versions; no longer used
    # elif condition == 3:
    #     relate_script = relate_ceu_script
    #     map_file = prepend_current_dir("genetic_map_12e-8.txt")
    # elif condition == 4:
    #     relate_script = relate_ceu_script
    #     map_file = prepend_current_dir("genetic_map_12e-10.txt")
    elif condition == 5:
        relate_script = relate_ceu_script
        map_file = prepend_current_dir("genetic_map_24e-9.txt")
    else:
        assert condition == 6
        relate_script = relate_ceu_script
        map_file = prepend_current_dir("genetic_map_6e-9.txt")

    if relate_script == relate_ceu_script:
        check_paths([ceu_diploid_file])
    check_paths([
        relate_script,
        map_file,
        os.path.join(relate_bin_dir, "Relate"),
        os.path.join(relate_bin_dir, "RelateFileFormats")])

    subprocess.run([relate_script, relate_bin_dir, relate_tmp_dir, map_file, demo_argument])

    relate_short_ts = tskit.load(os.path.join(relate_tmp_dir, out_prefix + ".trees"))
    shutil.rmtree(relate_tmp_dir)
    # For memory-profiling, we returned here without fixing the tree sequence
    if False:
        logging.info("Returning without fixing the tree sequence (for memory-profiling purposes)")
        return relate_short_ts

    # Change the breakpoints within the Relate ARG to be at midpoints
    # Also set the end of the Relate ARG to match the end of the simulation
    pos_map = {}
    all_breakpoints = list(relate_short_ts.breakpoints())
    for pos in all_breakpoints:
        int_pos = int(pos)
        idx = np.searchsorted(snp_positions, int_pos, side="right") - 1
        # print(pos, int_pos, idx, snp_positions[idx])
        if pos == all_breakpoints[-1]:
            assert int_pos == snp_positions[idx] + 1
            pos_map[pos] = simulation.sequence_length
        elif pos == all_breakpoints[0]:
            assert pos == 0
            assert int_pos == 0
            pos_map[pos] = pos
        else:
            assert int_pos == snp_positions[idx]
            assert idx >= 1
            pos_map[pos] = 0.5 * (snp_positions[idx - 1] + snp_positions[idx])
    # Now rewrite edge extents using the pos_map!

    tables = relate_short_ts.dump_tables()
    edges_left = tables.edges.left
    edges_right = tables.edges.right

    edges_left_new = edges_left.copy()
    edges_right_new = edges_right.copy()

    for i in range(len(edges_left)):
        edges_left_new[i] = pos_map[edges_left[i]]
        edges_right_new[i] = pos_map[edges_right[i]]

    tables.edges.set_columns(
        left=edges_left_new,
        right=edges_right_new,
        parent=tables.edges.parent,
        child=tables.edges.child)
    tables.sequence_length = simulation.sequence_length
    tables.sort()
    logging.info("Done with rewrite and sort")
    relate_fixed_ts = tables.tree_sequence()
    logging.info("Created fixed Relate ts object, returning")

    return relate_fixed_ts


def modify_data_return_new_ts(simulation, snp_ids=None, error_rate=0, random_seed=None, permutation=None):
    import tsinfer

    if random_seed is not None:
        assert isinstance(random_seed, int)
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    if permutation is not None:
        assert len(permutation) == simulation.num_samples
        assert sorted(permutation) == np.arange(simulation.num_samples).tolist()

    # https://github.com/tskit-dev/tsinfer/issues/194
    with tsinfer.SampleData(sequence_length=simulation.sequence_length) as sample_data:
        snp_ids_index = 0
        for i, variant in enumerate(simulation.variants()):
            if snp_ids is None or snp_ids[snp_ids_index] == i:
                genotypes = variant.genotypes
                if permutation is not None:
                    genotypes = genotypes[permutation]
                if error_rate > 0:
                    bits_to_flip = rng.uniform(size=simulation.num_samples) < error_rate
                    genotypes ^= bits_to_flip
                sample_data.add_site(variant.site.position, genotypes)
                snp_ids_index += 1
                if snp_ids is not None and snp_ids_index == len(snp_ids):
                    break

    # other arguments are recombination_rate, mismatch_ratio, and path_compression
    # https://tsinfer.readthedocs.io/en/latest/inference.html#matching-ancestors-samples
    new_ts = tsinfer.infer(sample_data)
    return new_ts


def phase_data_return_new_ts(beagle5_path, beagle_tmp_dir, simulation, snp_ids, num_sequence_samples, num_snp_samples):
    import tsinfer

    # get SNP positions
    snp_positions = np.array([variant.site.position for variant in simulation.variants()])[snp_ids]
    logging.info("Number of SNPs: {} = {}".format(len(snp_ids), len(snp_positions)))

    run_beagle5_phasing(beagle5_path, beagle_tmp_dir, simulation, snp_ids, num_sequence_samples, num_snp_samples)
    # Read in .csv results and create data for tsinfer
    # https://github.com/tskit-dev/tsinfer/issues/194
    with tsinfer.SampleData(sequence_length=simulation.sequence_length) as sample_data:
        with open(beagle_tmp_dir + "phased.csv", 'r') as in_file:
            for i, line in enumerate(in_file):
                split_line = line.strip('\n').split()
                phased_row = [[int(y) for y in x.split("|")] for x in split_line[3:]]
                phased_row_flatten = np.array(phased_row).flatten()
                assert len(phased_row_flatten) == num_snp_samples * 2
                sample_data.add_site(snp_positions[i], phased_row_flatten)

    shutil.rmtree(beagle_tmp_dir)

    # other arguments are recombination_rate, mismatch_ratio, and path_compression
    # https://tsinfer.readthedocs.io/en/latest/inference.html#matching-ancestors-samples
    new_ts = tsinfer.infer(sample_data)
    assert new_ts.num_samples == num_snp_samples * 2
    return new_ts


def get_mutation_times(ts, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    mutation_info = [[ts.site(m.site).position, m.node] for m in ts.mutations()]
    # mutation_info.sort()  # it's already sorted so we don't need to sort
    mutation_times = []
    mutation_index = 0
    position, node = mutation_info[mutation_index]
    for tree in ts.trees():
        while mutation_index < len(mutation_info) and tree.interval[0] <= position and position < tree.interval[1]:
            # process this mutation
            parent = tree.parent(node)
            if parent == tskit.NULL:
                raise RuntimeError("This case not supported")
            mutation_times.append(np.random.uniform(tree.time(node), tree.time(parent)))
            mutation_index += 1
            if mutation_index < len(mutation_info):
                position, node = mutation_info[mutation_index]

    assert len(mutation_times) == len(mutation_info)
    return mutation_times


def ts_hist(simulation, hist_bins, mutation_rate, seed):
    temp_ts = msprime.mutate(simulation, rate=mutation_rate, random_seed=seed)
    mutation_times = get_mutation_times(temp_ts, random_seed=seed)
    histogram = np.histogram(mutation_times, hist_bins, density=False)[0]
    return histogram.tolist()


def ukb_sample(simulation, start_sample_id=None, end_sample_id=None, verbose=False):
    """Use UKB spectrum to sample variants and obtain realistic SNP data.

    Arguments:
        simulation
        start_sample_id: first sample of range to compute allele frequency over
        end_sample_id: end sample of range to compute allele frequency over

    Returns:
        numpy array containing SNP indices
    """
    # Hard-coded values
    spectrum_path = prepend_current_dir("chr2.hist")
    spectrum_segment_size = 243*(10**6)
    sim_length = simulation.sequence_length
    factor = sim_length / spectrum_segment_size
    bin_starts = np.arange(51) / 100

    # Read in the spectrum file
    scaled_bin_counts = []
    with open(spectrum_path, 'r') as infile:
        for line in infile:
            values = line.strip('\n').split()
            scaled_bin_counts.append(int(round(int(values[2]) * factor)))
    scaled_bin_counts = np.array(scaled_bin_counts)
    if scaled_bin_counts.shape != bin_starts.shape:
        raise ValueError("Uh-oh, unexpected spectrum file")

    if False and verbose:
        logging.info(bin_starts)
        logging.info(scaled_bin_counts)

    binned_ids = [[] for i in range(len(bin_starts))]

    af = np.zeros(simulation.num_mutations)
    for i, variant in enumerate(simulation.variants()):
        subset = variant.genotypes[start_sample_id:end_sample_id]
        af[i] = float(np.sum(subset)) / float(len(subset))

    if len(subset) <= 100:
        raise ValueError("Number of samples for computing allele frequency is too small")

    maf = np.minimum(af, 1 - af)
    binned_maf = np.floor(maf * 100).astype(int)
    num_skipped = 0
    for i in range(len(binned_maf)):
        if maf[i] == 0:
            num_skipped += 1
        else:
            binned_ids[binned_maf[i]].append(i)
    if num_skipped > 0:
        logging.info("Skipping {} monomorphic SNPs".format(num_skipped))

    chosen_ones = []
    for i in range(len(bin_starts)):
        if len(binned_ids[i]) < scaled_bin_counts[i]:
            warning_string = "Can only choose " + str(len(binned_ids[i]))
            warning_string += " instead of " + str(scaled_bin_counts[i])
            warning_string += " IDs from [" + "{:.2f}".format(bin_starts[i])
            warning_string += ", " + "{:.2f}".format(bin_starts[i] + 0.01) + ")"
            logging.info(warning_string)
            chosen_ones.extend(binned_ids[i])
        else:
            chosen_ones.extend(np.random.choice(binned_ids[i], scaled_bin_counts[i], replace=False))
    chosen_ones = np.sort(chosen_ones)
    logging.info("Chose " + str(len(chosen_ones)) + " out of " + str(maf.shape[0]) + " SNPs")
    return chosen_ones


def run_impute4(impute4_path, impute_tmp_dir, simulation, snp_ids, num_sequence_samples, num_snp_samples, diploid=False):
    """
    Note that number of sequence samples + number of SNP samples can be less
    than total number of samples in the simulation. In this case, sequence
    samples are selected from the front and SNP samples are selected from the
    back.
    """
    assert num_sequence_samples + num_snp_samples <= simulation.num_samples
    assert num_sequence_samples % 2 == 0
    if diploid:
        assert num_snp_samples % 2 == 0
    logging.info("Using temporary directory " + impute_tmp_dir)
    os.makedirs(impute_tmp_dir, exist_ok=True)

    # Write haps file, reference file, and legend file
    out_path = impute_tmp_dir + "in.haps"
    ref_path = impute_tmp_dir + "ref.hap"
    legend_path = impute_tmp_dir + "ref.legend"
    with open(out_path, 'w') as out_file, \
        open(ref_path, 'w') as ref_file, \
        open(legend_path, 'w') as legend_file:
        last = -1
        snp_ids_index = 0
        legend_file.write('id position a0 a1\n')
        for i, variant in enumerate(simulation.variants()):
            pos = int(variant.site.position)
            if pos <= last:
                pos = last + 1
            last = pos
            row_list = [str(entry) for entry in variant.genotypes[:num_sequence_samples]]
            ref_file.write(' '.join(row_list))
            ref_file.write('\n')

            legend_row_list = ["foo_" + str(pos), str(pos), "A", "G"]
            legend_file.write(' '.join(legend_row_list))
            legend_file.write('\n')

            if snp_ids is None or snp_ids[snp_ids_index] == i:
                row_list = ['1', 'foo_' + str(pos), str(pos), 'A', 'G']
                if diploid:
                    row_list += ["{}".format(entry) for entry in variant.genotypes[-num_snp_samples:]]
                else:
                    # we need to duplicate to go from haploid to diploid
                    row_list += ["{} {}".format(entry, entry) for entry in variant.genotypes[-num_snp_samples:]]
                out_file.write(' '.join(row_list))
                out_file.write('\n')

                snp_ids_index += 1
                if snp_ids is not None and snp_ids_index == len(snp_ids):
                    break

    impute4_script = prepend_current_dir("run_impute4.sh")
    map_file = prepend_current_dir("genetic_map_12e-9.txt")
    check_paths([impute4_path, impute4_script, map_file])

    subprocess.run([
        impute4_script,
        impute4_path,
        impute_tmp_dir,
        map_file,
        str(int(simulation.sequence_length))
    ])

    out_path = impute_tmp_dir + "out.sample"
    with open(out_path, 'w') as out_file:
        out_file.write(' '.join(["ID_1", "ID_2", "missing"]) + '\n')
        out_file.write(' '.join(["0", "0", "0"]) + '\n')
        if diploid:
            for i in range(num_snp_samples // 2):
                out_file.write(' '.join([str(i+1), str(i+1), "0"]) + '\n')
        else:
            for i in range(num_snp_samples):
                out_file.write(' '.join([str(i+1), str(i+1), "0"]) + '\n')

def run_beagle5(beagle5_path, beagle_tmp_dir, simulation, snp_ids, num_sequence_samples, num_snp_samples):
    """
    For BEAGLE5, we require the number of sequence + SNP samples to match the
    total number of samples exactly.
    """
    assert num_sequence_samples + num_snp_samples == simulation.num_samples
    logging.info("Using temporary directory " + beagle_tmp_dir)
    os.makedirs(beagle_tmp_dir, exist_ok=True)

    def legacy_position_transform(positions):
        """
        Transforms positions in the tree sequence into VCF coordinates under
        the pre 0.2.0 legacy rule.
        """
        last_pos = 0
        transformed = []
        for pos in positions:
            pos = int(round(pos))
            if pos <= last_pos:
                pos = last_pos + 1
            transformed.append(pos)
            last_pos = pos
        return transformed

    with gzip.open(beagle_tmp_dir + "sim.vcf.gz", "wt") as vcf_file:
        simulation.write_vcf(vcf_file, ploidy=1, position_transform="legacy")
    phys_pos = np.array([variant.site.position for variant in simulation.variants()])
    all_positions = np.array(legacy_position_transform(phys_pos))
    snp_positions = all_positions[snp_ids]
    with open(beagle_tmp_dir + "sim.chip.variants", 'w') as out_file:
        for pos in snp_positions:
            out_file.write('\t'.join(["1", str(pos), str(pos)]) + '\n')

    beagle5_script = prepend_current_dir("run_beagle5.sh")
    map_file = prepend_current_dir("genetic_map_12e-9.map") # genetic_map_12e-9.txt is different
    check_paths([beagle5_path, beagle5_script, map_file])

    subprocess.run([
        beagle5_script,
        beagle5_path,
        beagle_tmp_dir,
        map_file,
        str(num_sequence_samples),
        str(num_snp_samples)
    ])


def run_beagle5_phasing(beagle5_path, beagle_tmp_dir, simulation, snp_ids, num_sequence_samples, num_snp_samples):
    """
    For BEAGLE5, we require the number of sequence + SNP samples to match the
    total number of samples exactly.
    """
    logging.info("Using temporary directory " + beagle_tmp_dir)
    os.makedirs(beagle_tmp_dir, exist_ok=True)

    def legacy_position_transform(positions):
        """
        Transforms positions in the tree sequence into VCF coordinates under
        the pre 0.2.0 legacy rule.
        """
        last_pos = 0
        transformed = []
        for pos in positions:
            pos = int(round(pos))
            if pos <= last_pos:
                pos = last_pos + 1
            transformed.append(pos)
            last_pos = pos
        return transformed

    with gzip.open(beagle_tmp_dir + "sim.vcf.gz", "wt") as vcf_file:
        simulation.write_vcf(vcf_file, ploidy=2, position_transform="legacy")
    phys_pos = np.array([variant.site.position for variant in simulation.variants()])
    all_positions = np.array(legacy_position_transform(phys_pos))
    snp_positions = all_positions[snp_ids]
    with open(beagle_tmp_dir + "sim.chip.variants", 'w') as out_file:
        for pos in snp_positions:
            out_file.write('\t'.join(["1", str(pos), str(pos)]) + '\n')

    beagle5_script = prepend_current_dir("run_beagle5_phasing.sh")
    map_file = prepend_current_dir("genetic_map_12e-9.map") # genetic_map_12e-9.txt is different
    check_paths([beagle5_path, beagle5_script, map_file])

    # Run Beagle5 phasing script
    subprocess.run([
        beagle5_script,
        beagle5_path,
        beagle_tmp_dir,
        map_file,
        str(num_sequence_samples),
        str(num_snp_samples)
    ])


if __name__ == "__main__":
    with btime():
        time.sleep(1)
    # Nesting example inspired by https://github.com/hector-sab/ttictoc
    with btime():
        time.sleep(1)
        with btime():
            time.sleep(1)
        time.sleep(1)
