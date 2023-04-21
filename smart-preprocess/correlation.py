import pandas as pd
import numpy as np

csv_path1 = ".\\failed\\net_failed_mc1.csv"
write_path1 = ".\\failed\\cor_mc1.csv"
csv_path2 = ".\\failed\\net_failed_mc2.csv"
write_path2 = ".\\failed\\cor_mc2.csv"
write_path3 = ".\\failed\\01_cor_mc1.csv"
write_path4 = ".\\failed\\01_cor_mc1_pearson.csv"
write_path5 = ".\\failed\\01_cor_mc1_spearman.csv"

NUM_DAYS = 30

if __name__ == "__main__":
    list = []

    result_df = pd.DataFrame(np.random.random([NUM_DAYS,45]))
    is_first = True

    for i in range(45):
        list.append(i)
    df = pd.read_csv(csv_path1, usecols=list)
    prev_disk = -1
    n_rows = df.shape[0]
    num_disks = 0

    counts = np.zeros(45, dtype=np.int)

    for i in range(n_rows):
        row = df.loc[i]
        cur_disk = row[0]
        # print(cur_disk)
        if prev_disk != cur_disk:
            if prev_disk != -1:
                this_df = df.loc[i-NUM_DAYS:i-1]
                prev_prev_disk = df.loc[i-NUM_DAYS][0]
                if prev_prev_disk == prev_disk:
                    list0 = []
                    for j in range(NUM_DAYS-1):
                        list0.append(0)
                    list0.append(1)
                    this_df['result'] = list0
                    data = this_df.corr(method='spearman').fillna(value=0)
                    print(prev_disk)
                    num_disks += 1
                    if is_first == True:
                        result_df = data
                        is_first = False
                    else:
                        result_df += data
                    for j in range(45):
                        not_minus1 = False
                        for t in range(NUM_DAYS):
                            if df.loc[i-NUM_DAYS+t][j] != -1:
                                not_minus1 = True
                                counts[j] += 1
                                break

            prev_disk = cur_disk
    print(result_df)
    result_df.to_csv(write_path5, sep='\t', index=False)
    for i in range(45):
        print(counts[i])