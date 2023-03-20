"""Utils for ARG complex trait analysis.
"""

# Python imports
import glob
import gzip
import logging
import numpy as np
import os
from scipy.stats import linregress
import subprocess

# Our packages
import arg_needle_lib
from arg_needle_lib import gower_center, write_grm

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_allele_counts(simulation, snp_indices):
    allele_counts = []
    idx = 0
    if snp_indices is None:
        last_index = simulation.num_mutations - 1
    else:
        last_index = snp_indices[-1]
    for i, variant in enumerate(simulation.variants()):
        if snp_indices is None or (idx < len(snp_indices) and i == snp_indices[idx]):
            ac = np.sum(variant.genotypes) # automatically becomes int64
            allele_counts.append(ac)
            idx += 1

    if snp_indices is None:
        assert len(allele_counts) == simulation.num_mutations
    else:
        assert len(allele_counts) == len(snp_indices)

    return np.array(allele_counts, dtype=np.int64)


def get_allele_frequencies(simulation, snp_indices):
    allele_counts = get_allele_counts(simulation, snp_indices)
    # the following gave different results than np.mean
    # return np.array(allele_counts, dtype=np.float32) / float(simulation.num_samples)
    # the following gave same results as np.mean
    return np.array(allele_counts) / float(simulation.num_samples)


def make_grm(simulation, snp_indices, alpha, diploid=False, batch_size=256, verbose=False):
    """
    See also arg_needle_lib.compute_grm, arg_needle_lib.exact_arg_grm,
    arg_needle_lib.monte_carlo_arg_grm
    """
    num_samples = simulation.num_samples
    if diploid:
        assert num_samples % 2 == 0
        num_samples //= 2

    grm = np.zeros((num_samples, num_samples), dtype=np.float32)

    idx = 0
    num_batches = 0
    num_skipped = 0
    genotype_lists = []
    if snp_indices is None:
        last_index = simulation.num_mutations - 1
    else:
        last_index = snp_indices[-1]
    for i, variant in enumerate(simulation.variants()):
        if snp_indices is None or (idx < len(snp_indices) and i == snp_indices[idx]):
            row_sum = np.sum(variant.genotypes)
            if row_sum == 0 or row_sum == len(variant.genotypes):
                num_skipped += 1
                continue
            genotype_lists.append(variant.genotypes)
            idx += 1
            if len(genotype_lists) == batch_size or i == last_index:
                if verbose:
                    logging.info("Batch {}".format(num_batches))
                geno = np.array(genotype_lists, dtype=np.float32)
                if diploid:
                    geno = geno.reshape((geno.shape[0], num_samples, 2)).sum(axis=-1)
                geno -= np.mean(geno, axis=1)[:, np.newaxis]
                std = np.std(geno, axis=1, ddof=1)
                scales = std**alpha
                geno *= scales[:, np.newaxis]
                if verbose:
                    logging.info("About to dot")
                grm += np.dot(np.transpose(geno), geno)
                if verbose:
                    logging.info("Done with dot")
                num_batches += 1
                genotype_lists = []
    logging.info("total skipped = {}".format(num_skipped))
    return grm


def write_mgrm(grms, num_sites_list, temp_base_path, write_binary=True):
    if not isinstance(grms, list):
        raise ValueError("Expects a list of GRMs")
    assert len(grms) > 0
    num_samples = grms[0].shape[0]
    if not isinstance(num_sites_list, list):
        raise ValueError("Expects a list of number of SNPs used for each GRM")
    assert len(grms) == len(num_sites_list)

    with open(temp_base_path + '.mgrm.txt', 'w') as outfile:
        for k in range(len(grms)):
            outfile.write(temp_base_path + '.' + str(k) + '\n')
    for k in range(len(grms)):
        grm = grms[k]
        assert len(grm.shape) == 2
        assert grm.shape[0] == num_samples
        assert grm.shape[1] == num_samples
        num_sites = num_sites_list[k]
        path_prefix = temp_base_path + '.' + str(k)
        write_grm(grm, num_sites, path_prefix, write_binary)


def write_haps(simulations, snp_indices_list, temp_base_path, diploid=False, chr_start=1, snp_prefix=None, offset=0, write_gz=False):
    assert len(simulations) == len(snp_indices_list)

    # Write .haps / .sample
    with open(temp_base_path + ".sample", 'w') as out_file:
        out_file.write(' '.join(["ID_1", "ID_2", "missing"]) + '\n')
        out_file.write(' '.join(["0", "0", "0"]) + '\n')
        num_samples = simulations[0].num_samples
        if diploid:
            assert num_samples % 2 == 0
            num_samples //= 2
        for i in range(num_samples):
            out_file.write(' '.join([str(i+1), str(i+1), "0"]) + '\n')

    if write_gz:
        out_file = gzip.open(temp_base_path + ".haps.gz", 'wb', compresslevel=1)
    else:
        out_file = open(temp_base_path + ".haps", 'w')
    for chromosome in range(len(simulations)):
        idx = 0
        last_pos = -1
        snp_indices = snp_indices_list[chromosome]
        for i, variant in enumerate(simulations[chromosome].variants()):
            if idx < len(snp_indices) and i == snp_indices[idx]:
                pos = int(variant.position + offset)
                if pos <= last_pos:
                    pos = last_pos + 1
                last_pos = pos
                row_sum = np.sum(variant.genotypes)
                if row_sum == 0 or row_sum == len(variant.genotypes):
                    logging.info("skipping site " + str(i) + ", row sum = " + str(row_sum))
                    continue
                if snp_prefix is None or snp_prefix == "":
                    prefix = str(chromosome + chr_start)
                else:
                    assert len(simulations) == 1
                    prefix = snp_prefix
                row_list = [str(chromosome + chr_start), prefix + ':' + str(idx + 1), str(pos), '0', '1']
                if diploid:
                    row_list += [str(int(entry)) for entry in variant.genotypes]
                else:
                    # we need to double because we go haploid to diploid
                    row_list += [str(int(entry)) + ' ' + str(int(entry)) for entry in variant.genotypes]
                if write_gz:
                    out_file.write((' '.join(row_list) + '\n').encode())
                else:
                    out_file.write(' '.join(row_list) + '\n')
                idx += 1
    out_file.close()


# Writes simulated genotypes in PLINK format
# Memory-efficient version
def write_plink_new(plink2_path, base_tmp_dir, simulations, snp_indices_list, pkl_path=None, diploid=False, chr_start=1, snp_prefix=None, offset=0):
    logging.info("Entering write_plink function")
    if pkl_path is not None:
        temp_base_path = os.path.join(base_tmp_dir, pkl_path.split("/")[-1][:-4] + ".temp")
    else:
        temp_base_path = "temp"

    write_haps(
        simulations, snp_indices_list, temp_base_path, diploid, chr_start,
        snp_prefix, offset, write_gz=True)

    logging.info("About to run PLINK2")
    subprocess.run([
        plink2_path,
        "--threads",
        "1",
        "--memory",
        "2048",
        "--haps",
        temp_base_path + ".haps.gz",
        "--sample",
        temp_base_path + ".sample",
        "--make-bed",
        "--out",
        temp_base_path
    ], check=True)

    # Remove files that are no longer needed
    os.remove(temp_base_path + ".haps.gz")
    os.remove(temp_base_path + ".sample")
    logging.info("Exiting write_plink function")

    return temp_base_path


# Write all bitsets in haps[.gz] / sample format
# Note: if max_mac = 0, it's as if it isn't set
def arg_write_haps(base_tmp_dir, arg, pkl_path=None, diploid=False, efficient=True, chromosome=1,
                   snp_prefix=None, min_mac=0, max_mac=0, write_gz=True):
    if pkl_path is not None:
        temp_base_path = os.path.join(base_tmp_dir, pkl_path.split("/")[-1][:-4] + ".temp")
    else:
        temp_base_path = "temp"
    if snp_prefix is None:
        snp_prefix = ""

    suffix = ""
    if efficient:
        if write_gz:
            suffix = ".gz"
        num_bitsets = arg_needle_lib.write_bitsets(
            arg, temp_base_path + ".arg", diploid, chromosome, snp_prefix,
            min_mac, max_mac, use_gz=write_gz)
    else:
        num_bitsets = arg_needle_lib.write_bitsets_detailed(
            arg, temp_base_path + ".arg", diploid, chromsome, snp_prefix)

    # see how many things were written
    logging.info("Number of bitsets: {}".format(num_bitsets))
    return temp_base_path, num_bitsets


# Write all bitsets in PLINK format
# Note: if max_mac = 0, it's as if it isn't set
def arg_write_plink(plink2_path, base_tmp_dir, arg, pkl_path=None, diploid=False, efficient=True,
                    chromosome=1, snp_prefix=None, min_mac=0, max_mac=0, write_gz=True):
    if efficient and write_gz:
        suffix = ".gz"
    temp_base_path, num_bitsets = arg_write_haps(
        base_tmp_dir, arg, pkl_path, diploid, efficient, chromosome,
        snp_prefix, min_mac, max_mac, write_gz)

    subprocess.run([
        plink2_path,
        "--threads",
        "1",
        "--memory",
        "2048",
        "--haps",
        temp_base_path + ".arg.haps" + suffix,
        "--sample",
        temp_base_path + ".arg.sample",
        "--make-bed",
        "--out",
        temp_base_path + ".arg"
    ])

    # Remove files that are no longer needed
    os.remove(temp_base_path + ".arg.haps" + suffix)
    os.remove(temp_base_path + ".arg.sample")

    # Note: need to remove .arg.bitinfo, .arg.bed / .arg.bim / .arg.fam
    return temp_base_path, num_bitsets


# Write all bitsets in BGEN v1.2 bits=8 format
# Note: if max_mac = 0, it's as if it isn't set
def arg_write_bgen(plink2_path, base_tmp_dir, arg, pkl_path=None, diploid=False, efficient=True,
                    chromosome=1, snp_prefix=None, min_mac=0, max_mac=0, write_gz=True):
    if efficient and write_gz:
        suffix = ".gz"
    temp_base_path, num_bitsets = arg_write_haps(
        base_tmp_dir, arg, pkl_path, diploid, efficient, chromosome,
        snp_prefix, min_mac, max_mac, write_gz)

    if not diploid:
        subprocess.run([
            plink2_path,
            "--threads",
            "1",
            "--memory",
            "2048",
            "--haps",
            temp_base_path + ".arg.haps" + suffix,
            "--sample",
            temp_base_path + ".arg.sample",
            "--export",
            "bgen-1.2",
            "bits=8",
            "--out",
            temp_base_path + ".test"
        ])

        # Remove files that are no longer needed
        os.remove(temp_base_path + ".arg.haps" + suffix)
        os.remove(temp_base_path + ".arg.sample")
    else:
        subprocess.run([
            plink2_path,
            "--threads",
            "1",
            "--memory",
            "2048",
            "--haps",
            temp_base_path + ".arg.haps" + suffix,
            "--sample",
            temp_base_path + ".arg.sample",
            "--make-pgen",
            "erase-phase",
            "--out",
            temp_base_path + ".arg"
        ])

        # Remove files that are no longer needed
        os.remove(temp_base_path + ".arg.haps" + suffix)
        os.remove(temp_base_path + ".arg.sample")

        subprocess.run([
            plink2_path,
            "--threads",
            "1",
            "--memory",
            "2048",
            "--pfile",
            temp_base_path + ".arg",
            "--export",
            "bgen-1.2",
            "bits=8",
            "--out",
            temp_base_path + ".test"
        ])

        # Remove files that are no longer needed
        os.remove(temp_base_path + ".arg.pgen")
        os.remove(temp_base_path + ".arg.pvar")
        os.remove(temp_base_path + ".arg.psam")

    # Note: need to remove .arg.bitinfo, .test.bgen / .test.sample
    return temp_base_path, num_bitsets


def sim_pheno_new(simulations, h2=0.25, alpha=-1., random_seed=None, diploid=False):
    if random_seed is not None:
        np.random.seed(random_seed)
    num_samples = simulations[0].num_samples
    if diploid:
        assert num_samples % 2 == 0
        num_samples //= 2
    phenotypes = np.zeros(num_samples)

    num_sites = 0
    for simulation in simulations:
        for variant in simulation.variants():
            row = variant.genotypes.astype('float64')
            if diploid:
                row = row.reshape((num_samples, 2)).sum(axis=-1)
            std = np.std(row, ddof=1)
            if std == 0:
                logging.info("skipping site " + str(num_sites) + ", row sum = " + str(row.sum()))
                continue
            beta = np.random.randn()
            phenotypes += row * (beta * std**alpha)
            num_sites += 1

    # normalize pheno to have mean 0 and var = h2
    phenotypes -= np.mean(phenotypes)
    phenotypes /= np.std(phenotypes, ddof=1)
    phenotypes *= np.sqrt(h2)

    # sample environmental component with variance 1-h2 and add it to phenotype
    phenotypes += np.random.randn(num_samples) * np.sqrt(1-h2)

    # normalize it all
    phenotypes -= np.mean(phenotypes)
    phenotypes /= np.std(phenotypes, ddof=1)

    return phenotypes, num_sites


def write_pheno(phen_to_write, path, header=False):
    num_samples = phen_to_write.shape[0]
    with open(path, 'w') as outfile:
        if header:
            outfile.write('\t'.join(["FID", "IID", "pheno"]) + '\n')
        for i in range(num_samples):
            outfile.write('\t'.join([str(i+1), str(i+1), str(phen_to_write[i])]) + '\n')


# From .gen[.gz] file, most commonly used after imputation
def get_allele_frequencies_gen(file_name, gen_gz=False):
    allele_frequencies = []

    # https://stackoverflow.com/a/56504533/
    if gen_gz:
        opener = gzip.open
        opener_status = "rt"
    else:
        opener = open
        opener_status = "r"

    with opener(file_name, opener_status) as in_file:
        for i, line in enumerate(in_file):
            impute_result = [float(x) for x in line.strip('\n').split()[5:]]
            result_array = np.array(impute_result, dtype=np.float32)
            haploid_dosages = 0.5*result_array[1::3] + result_array[2::3]
            allele_frequencies.append(haploid_dosages.mean())

    return np.array(allele_frequencies, dtype=np.float32)


# From .gen[.gz] file, most commonly used after imputation
def make_grm_gen(file_name, snp_indices, alpha, diploid=False, gen_gz=False, batch_size=256, verbose=False):
    """
    To filter by MAF or other criteria, use snp_indices
    """
    num_samples = None
    grm = None
    idx = 0
    num_batches = 0
    num_skipped = 0
    genotype_lists = []

    # https://stackoverflow.com/a/56504533/
    if gen_gz:
        opener = gzip.open
        opener_status = "rt"
    else:
        opener = open
        opener_status = "r"

    with opener(file_name, opener_status) as in_file:
        for i, line in enumerate(in_file):
            impute_result = [float(x) for x in line.strip('\n').split()[5:]]
            result_array = np.array(impute_result, dtype=np.float32)
            haploid_dosages = 0.5*result_array[1::3] + result_array[2::3]
            if num_samples is None:
                num_samples = haploid_dosages.shape[0]
                if diploid:
                    assert num_samples % 2 == 0
                    num_samples //= 2
                grm = np.zeros((num_samples, num_samples), dtype=np.float32) # could this be causing issues?

            if snp_indices is None or (idx < len(snp_indices) and i == snp_indices[idx]):
                row_sum = np.sum(haploid_dosages)
                if abs(row_sum) < 1e-8 or abs(row_sum - len(haploid_dosages)) < 1e-8:
                    if verbose:
                        logging.info("skipping site " + str(i) + ", row sum = " + str(row_sum))
                    num_skipped += 1
                    continue
                row_std = np.std(haploid_dosages, ddof=1)
                if row_std < 1e-8:
                    num_skipped += 1
                    continue
                if row_std < 1e-6 and row_std > 1e-8:
                    num_skipped += 1
                    if verbose:
                        logging.info("skipping site {}, row std = {}, entry 0 = {}".format(
                            i, row_std, haploid_dosages[0]))
                    continue
                genotype_lists.append(haploid_dosages)
                idx += 1
                if len(genotype_lists) == batch_size:
                    if verbose:
                        logging.info("Batch {}".format(num_batches))
                    geno = np.array(genotype_lists, dtype=np.float32)
                    if diploid:
                        geno = geno.reshape((geno.shape[0], num_samples, 2)).sum(axis=-1)
                    geno -= np.mean(geno, axis=1)[:, np.newaxis]
                    std = np.std(geno, axis=1, ddof=1)
                    scales = std**alpha
                    geno *= scales[:, np.newaxis]
                    if np.isinf(np.sum(geno)) or np.isnan(np.sum(geno)):
                        import pdb; pdb.set_trace();
                    if verbose:
                        logging.info("About to dot")
                    grm += np.dot(np.transpose(geno), geno)
                    if verbose:
                        logging.info("Done with dot")
                    num_batches += 1
                    genotype_lists = []

    if len(genotype_lists) > 0:
        if verbose:
            logging.info("Batch {}".format(num_batches))
        geno = np.array(genotype_lists, dtype=np.float32)
        if diploid:
            geno = geno.reshape((geno.shape[0], num_samples, 2)).sum(axis=-1)
        geno -= np.mean(geno, axis=1)[:, np.newaxis]
        std = np.std(geno, axis=1, ddof=1)
        scales = std**alpha
        geno *= scales[:, np.newaxis]
        if verbose:
            logging.info("About to dot")
        grm += np.dot(np.transpose(geno), geno)
        if verbose:
            logging.info("Done with dot")
        num_batches += 1
        genotype_lists = []

    logging.info("total skipped = {}".format(num_skipped))

    return grm


def gcta(gcta_path, base_tmp_dir, grms, pheno, num_sites_list, pkl_path=None, write_binary=True, memory_efficient=True):
    # TODO: could put all these files into a folder for better organization
    if pkl_path is not None:
        temp_base_path = os.path.join(base_tmp_dir, pkl_path.split("/")[-1][:-4] + ".temp")
    else:
        temp_base_path = "temp"
    write_pheno(pheno, temp_base_path + '.phen')
    logging.info("Starting to write GRMs")
    write_mgrm(grms, num_sites_list, temp_base_path, write_binary)
    logging.info("Done writing GRMs")
    if len(grms) <= 1 and memory_efficient:
        del grms[0]  # delete the GRM so it's not taking space when we run GCTA
        import gc; gc.collect();

    # Run GCTA and read resulting h2
    if write_binary:
        mgrm_flag = "--mgrm"
    else:
        mgrm_flag = "--mgrm-gz"
    subprocess.run([
        gcta_path,
        "--reml-no-constrain",
        "--reml-no-lrt",
        "--cvblup",  # TODO: toggle when this gets turned on based on `method`
        mgrm_flag,
        temp_base_path + ".mgrm.txt",
        "--pheno",
        temp_base_path + ".phen",
        "--out",
        temp_base_path
    ])
    gcta_h2 = None
    gcta_h2_se = None
    gcta_h2_weights = []
    try:
        vgvp_index = 1
        with open(temp_base_path + '.hsq', 'r') as infile:
            for line in infile:
                tokens = line.strip('\n').split()
                if len(tokens) > 0 and tokens[0] == "V(G" + str(vgvp_index) + ")/Vp":
                    gcta_h2_weights.append(float(tokens[1]))
                    vgvp_index += 1
                if len(tokens) > 0 and tokens[0] == "V(G)/Vp":
                    gcta_h2 = float(tokens[1])
                    gcta_h2_se = float(tokens[2])
                    break
                if len(tokens) > 2 and tokens[2] == "V(G)/Vp":
                    assert tokens[0] == "Sum"
                    assert tokens[1] == "of"
                    gcta_h2 = float(tokens[3])
                    gcta_h2_se = float(tokens[4])
                    break
        logging.info("GCTA succeeded")
    except IOError:
        logging.info("GCTA REML failed, returning None")
        # delete files before returning
        for temp_file in glob.glob(temp_base_path + '*'):
            os.remove(temp_file)
        return None, None, None, None

    # In this case, we form the combined GRM and use that for cvBLUP
    # If this behavior is not desired, can turn it off
    if len(grms) > 1:
        assert len(gcta_h2_weights) == len(grms)
        logging.info(gcta_h2_weights)
        num_samples = grms[0].shape[0]
        combined_grm = np.zeros((num_samples, num_samples))
        combined_num_sites = 0
        for i in range(len(grms)):
            combined_grm += gcta_h2_weights[i] * grms[i]
            combined_num_sites += num_sites_list[i]
        combined_grm = gower_center(combined_grm)
        write_mgrm([combined_grm], [combined_num_sites], temp_base_path, write_binary)

        subprocess.run([
            gcta_path,
            "--reml-no-constrain",
            "--reml-no-lrt",
            "--cvblup",  # TODO: toggle when this gets turned on based on `method`
            mgrm_flag,
            temp_base_path + ".mgrm.txt",
            "--pheno",
            temp_base_path + ".phen",
            "--out",
            temp_base_path
        ])

    gcta_cvblup_r2 = None
    gcta_cvblup_mse = None
    blup = []
    residuals = []
    try:
        with open(temp_base_path + '.indi.cvblp', 'r') as infile:
            for line in infile:
                tokens = line.strip('\n').split()
                blup.append(np.sum([float(token) for token in tokens[3:-1:2]]))
                residuals.append(float(tokens[-1]))
        blup = np.array(blup)
        residuals = np.array(residuals)
        slope, intercept, r_value, p_value, std_err = linregress(blup, pheno)
        gcta_cvblup_r2 = r_value ** 2
        gcta_cvblup_mse = np.mean(np.square(pheno - blup))
    except IOError:
        logging.info("GCTA CV BLUP failed")

    # delete files before returning
    for temp_file in glob.glob(temp_base_path + '*'):
        os.remove(temp_file)
    logging.info((gcta_h2, gcta_h2_se, gcta_cvblup_r2, gcta_cvblup_mse))
    return gcta_h2, gcta_h2_se, gcta_cvblup_r2, gcta_cvblup_mse
