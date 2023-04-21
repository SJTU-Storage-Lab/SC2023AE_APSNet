# SMART Attributes Preprocess

## Notes
- The original SMART dataset is not included in this repository. Please refer to Alibaba's [Large-scale SSD Failure Prediction Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95044) [1] for details. The files used are
  - ```smartlog2018ssd/*```
  - ```smartlog2019ssd/*```
  - ```ssd_failure_label.csv```
- For simplicity, only the final output files are shown in this repository under the [npy](https://github.com/YunfeiGu/APSNet/tree/main/smart-preprocess/npy) folder. Intermediate results, including ```output/*```, ```failed/*```, ```good/*```, are not uploaded.


[1] Xu, Fan, et al. "General Feature Selection for Failure Prediction in Large-scale SSD Deployment." 2021 51st Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN). IEEE, 2021.

## File description
- [parse_smart.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/parse_smart.cpp)
  - Read all SMART csv files, and group them into separate files according to diskid/10000 (e.g. MC1 disks with diskid 30000-39999 are placed in MC1_R_3.csv and MC1_N_3.csv)
  - MC1 and MC2 disks are processed separately
  - Raw attributes and normalized attributes are processed separately (raw attributes are denoted as "R", and normalized attributes are denoted as "N"); however, only raw attributes are used in following steps
  - Pre-select attributes for further processing: attributes 3, 4, 10, 189, 206, 207 and 245 are ignored
  - Output: ```output/MC1_R_<n>.csv```, ```output/MC2_R_<n>.csv```, ```output/MC1_N_<n>.csv```, and ```output/MC2_N_<n>.csv```, where n is in [0, 20]
  - Convert timestamp (ds) from into day index starting from 0
- [delete_column_mc1.py](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/delete_column_mc1.py) and [delete_column_mc2.py](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/delete_column_mc2.py)
  - Traverse through all the files in the form of ```output/MC1_R_<n>.csv``` and ```output/MC2_R_<n>.csv```
  - Delete attributes that are not recorded by any SSD (i.e. delete empty columns)
- [delete_row_mc1.py](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/delete_row_mc1.py) and [delete_row_mc2.py](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/delete_row_mc2.py)
  - Traverse through all the files in the form of ```output/MC1_R_<n>.csv``` and ```output/MC2_R_<n>.csv```
  - Delete empty day records (i.e. delete rows that are empty except in the first two fields: disk id, and timestamp)
- [fetch_all_bad.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/fetch_all_bad.cpp)
  - Read failed drive labels under ```output``` that are labeled in ```ssd_failure_label.csv```, and collect all the attributes of failed drives into two csv files, ```failed/failed_mc1.csv``` and ```failed/failed_mc2.csv```, in which each row corresponds to attributes of a certain disk at a certain recorded time
  - Convert timestamp (ds) from into day index starting from 0
- [failed_single.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/failed_single.cpp)
  - Read ```failed/failed_mc1.csv``` and ```failed/failed_mc2.csv```, and save SMART data for each failed drive under ```failed/mc1``` and ```failed/mc2``` according to their types
  - Perform SMART attribute selection in the first round for failed drives
- [partial_statistics_good_model1.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/partial_statistics_good_model1.cpp)
  - Read csv files of raw attributes under ```output```, and neglect bad drives
  - Select 6 key attributes for model 1:
  
    | Attribute No. | Attribute Explanation |
    | --- | --- |
    | 194 | temperature |
    | 9 | power-on hours |
    | 173 | wear leveling status |
    | 171 | program errors |
    | 172 | erase errors |
    | 1 | read error rates |
  - Output: ```good/partial_statistics_mc1_model1.csv``` and ```good/partial_statistics_mc2_model1.csv```
- [partial_statistics_good_model2.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/partial_statistics_good_model2.cpp)
  - Read csv files of raw attributes under ```output```, and neglect bad drives
  - Select 8 key attributes for model 2:

    | Attribute No. | Attribute Explanation |
    | --- | --- |
    | 194 | temperature |
    | 9 | power-on hours |
    | 188 | command timeout |
    | 180 | unused reserved blocks |
    | 174 | unexpected power loss |
    | 183 | sata errors |
    | 1 | read error rates |
    | 187 | uncorrectable errors |
  - Output: ```good/partial_statistics_mc1_model2.csv``` and ```good/partial_statistics_mc2_model2.csv```
- [statistics_bad_model1.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/statistics_bad_model1.cpp)
  - Read all the csv files of bad drives under ```failed/mc1``` and ```failed/mc2```
  - Select attributes for model 1
  - Output: ```failed/mc1/statistics_model1.csv``` and ```failed/mc2/statistics_model1.csv```
- [statistics_bad_model2.cpp](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/statistics_bad_model2.cpp)
  - Select attributes for model 2
  - Output: ```failed/mc1/statistics_model2.csv``` and ```failed/mc2/statistics_model2.csv```
- [c2py.py](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/c2py.py)
  - Convert csv files into npy files
  - The final npy files have shape (N, T, M), where N is the number of drives in the file, T is the length of time series, and M is the number of attributes collected in the file
  - (csv, npy) pairs:
    - (```good/partial_statistics_mc1_model1.csv```, ```npy/partial_statistics_mc1_model1.npy```)
    - (```good/partial_statistics_mc2_model1.csv```, ```npy/partial_statistics_mc2_model1.npy```)
    - (```good/partial_statistics_mc1_model2.csv```, ```npy/partial_statistics_mc1_model2.npy```)
    - (```good/partial_statistics_mc2_model2.csv```, ```npy/partial_statistics_mc2_model2.npy```)
    - (```failed/mc1/statistics_model1.csv```, ```npy/bad_mc1_model1.npy```)
    - (```failed/mc2/statistics_model1.csv```, ```npy/bad_mc2_model1.npy```)
    - (```failed/mc1/statistics_model2.csv```, ```npy/bad_mc1_model2.npy```)
    - (```failed/mc2/statistics_model2.csv```, ```npy/bad_mc2_model2.npy```)
- [correlation.py](https://github.com/YunfeiGu/APSNet/blob/main/smart-preprocess/correlation.py)
  - Calculate spearman correlation between SMART attributes and drive failure
