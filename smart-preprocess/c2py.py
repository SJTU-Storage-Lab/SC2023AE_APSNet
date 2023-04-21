import numpy as np
import io

def c_plus_plus_to_numpy(n_cols, csv_path, npy_path=''):
    print("reading " + csv_path + "...")
    data = np.loadtxt(io.open(csv_path), delimiter=',', skiprows=1, dtype=int, usecols=np.arange(n_cols))
    print(data)
    if npy_path != '':
        print("saving " + npy_path + "...")
        np.save(npy_path, data)

def transform_npy(npy_in, npy_out):
    print("transforming " + npy_in + "...")
    data = np.load(npy_in)
    cur_disk = []
    last_disk = -1
    final_list = []
    for row in data:
        attributes = []
        for i in range(len(row)):
            attributes.append(row[i])
        diskid = attributes[0]
        if (last_disk == -1) | (diskid == last_disk):
            cur_disk.append(attributes)
        else:
            final_list.append(cur_disk)
            cur_disk = []
            cur_disk.append(attributes)
    final_list.append(cur_disk)
    data = np.array(final_list)
    print("saving " + npy_out + "...")
    np.save(npy_out, data)
    print("...done")

def run_good():
    # good files
    good_path = "./good/"

    partial_mc1_model1_path = good_path + "partial_statistics_mc1_model1.csv"
    partial_mc1_model1_npy_path = good_path + "partial_statistics_mc1_model1.npy"
    c_plus_plus_to_numpy(7, partial_mc1_model1_path, partial_mc1_model1_npy_path)

    partial_mc1_model2_path = good_path + "partial_statistics_mc1_model2.csv"
    partial_mc1_model2_npy_path = good_path + "partial_statistics_mc1_model2.npy"
    c_plus_plus_to_numpy(9, partial_mc1_model2_path, partial_mc1_model2_npy_path)

    partial_mc2_model1_path = good_path + "partial_statistics_mc2_model1.csv"
    partial_mc2_model1_npy_path = good_path + "partial_statistics_mc2_model1.npy"
    c_plus_plus_to_numpy(7, partial_mc2_model1_path, partial_mc2_model1_npy_path)

    partial_mc2_model2_path = good_path + "partial_statistics_mc2_model2.csv"
    partial_mc2_model2_npy_path = good_path + "partial_statistics_mc2_model2.npy"
    c_plus_plus_to_numpy(9, partial_mc2_model2_path, partial_mc2_model2_npy_path)

    # transform
    npy_path = "./npy/"

    transform_npy(partial_mc1_model1_npy_path, npy_path + "partial_statistics_mc1_model1.npy")
    transform_npy(partial_mc1_model2_npy_path, npy_path + "partial_statistics_mc1_model2.npy")
    transform_npy(partial_mc2_model1_npy_path, npy_path + "partial_statistics_mc2_model1.npy")
    transform_npy(partial_mc2_model2_npy_path, npy_path + "partial_statistics_mc2_model2.npy")


def run_bad():
    # bad files
    bad_path = "./failed/"

    bad_mc1_model1_path = bad_path + "mc1/statistics_model1.csv"
    bad_mc1_model1_npy_path = bad_path + "mc1/statistics_model1.npy"
    c_plus_plus_to_numpy(7, bad_mc1_model1_path, npy_path=bad_mc1_model1_npy_path)

    bad_mc1_model2_path = bad_path + "mc1/statistics_model2.csv"
    bad_mc1_model2_npy_path = bad_path + "mc1/statistics_model2.npy"
    c_plus_plus_to_numpy(9, bad_mc1_model2_path, npy_path=bad_mc1_model2_npy_path)

    bad_mc2_model1_path = bad_path + "mc2/statistics_model1.csv"
    bad_mc2_model1_npy_path = bad_path + "mc2/statistics_model1.npy"
    c_plus_plus_to_numpy(7, bad_mc2_model1_path, npy_path=bad_mc2_model1_npy_path)

    bad_mc2_model2_path = bad_path + "mc2/statistics_model2.csv"
    bad_mc2_model2_npy_path = bad_path + "mc2/statistics_model2.npy"
    c_plus_plus_to_numpy(9, bad_mc2_model2_path, npy_path=bad_mc2_model2_npy_path)

    # transform
    npy_path = "./npy/"

    transform_npy(bad_mc1_model1_npy_path, npy_path + "bad_mc1_model1.npy")
    transform_npy(bad_mc1_model2_npy_path, npy_path + "bad_mc1_model2.npy")
    transform_npy(bad_mc2_model1_npy_path, npy_path + "bad_mc2_model1.npy")
    transform_npy(bad_mc2_model2_npy_path, npy_path + "bad_mc2_model2.npy")

if __name__ == "__main__":
    run_good()
    run_bad()

    