# SC2023AE_APSNet
Supplemental code and dataset for Exploit both SMART Attributes and NAND Flash Wear Characteristics to Effectively Forecast SSD-based Storage Failures in Clusters.

We provide two experiment reproduction schemes: 
1. Experiment using Local Environment
2. Experiment using Container Image

## Artifact Check-List
The project is structured as follows:

 - The folder [source](souce) contains learning models source code including offline learning models and online learning models.
 - The folder [source/metrics](source/metrics) contains evaluation source codes using metrics that are introduced in Section 5.1.2 in this paper.
 - The folder [source/utils](source/utils) contains source codes of model definitions and common-used tools.
 - The folder [data/offline_dataset](data/offline_dataset) contains the files needed for offline models training and evaluation.
 - The folder [data/mc1_mc2](data/mc1_mc2) and [data/mc2](data/mc2) contains the pre-precessed dataset for training of online learning models. This experiment is used for reproduce the results of Exp#4 in Section 5.2.3 in this paper.
 - The folder [trained_model/offline_model](trained\_model/offline\_model) contains the well-trained APSNet, RF, LSTM, NN, DT, LR, SVM, KNN learning models in offline mode. These models are pre-trained and could be used for evalution directly.
 - The folder [trained_model/mc2](trained\_model/mc2) contains the well-trained APSNet, RF, LSTM, NN, DT, LR, SVM, KNN learning models in online mode. These models are pre-trained and could be used for evalution directly.
 - The folder [loss](loss) contains the loss records while training different learning models with various epoches.
 - The folder [log](log) contains the inference evaluation results logs of APSNet, RF, LSTM, NN, DT, LR, SVM, KNN learning model.
 - The folder [BEC_aging_data](BEC\_aging\_data) contains the open-source normalized SSD aging BEC data over 20,000 SSD drives.  
 - The folder [smart-preprocess](smart\-preprocess) contains SMART preprocess source code and processed SMART data that are used for the training/test SMART pool.
 - [slidingWindow.py]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/slidingWindow.py) is a script to implement sliding window mechanisms to evaluate the model performance on N days lookahead failure prediction. 
 - The folder [ContainerImage](ContainerImage) contains the container image used for facilitating the artifact evaluation. This image has installed all the library dependencies. All the source code and dataset used for training, test, validition have already been included in this image. This image should be run on docker 20.10.14. 
 - [environment.yml](environment.yml) is a file that records the experiment library setup and dependencies.


The APSNet code is released under the BSD-3 [License](LICENSE).

## Prerequisites
### Hardware dependencies
- All the experiments are executed on a server comprising Intel(R) Xeon(R) Gold 5218R CPU@2.10GHz processor, 32GB of DDR4-3200MT/s, and dual NVIDIA GeForce RTX 3090. The system was running Linux operating system, Ubuntu 18.04.
- Minimum hardware requirements: 2 GHz dual-core x86 processor or better, 16 GB system memory, 40GB free hard drive space, and NVIDIA GPU which could support CUDA 11.1 and cudnn 8.0.5.
### Software Dependencies
- It is mainly developed in g++ 7.5.0 and Python 3.7.11. The key libraries should be included: conda 4.10.3, numpy 1.19.5, pandas 0.24.2, pytorch 1.8.0, sklearn 1.0.1, Matplotlib 3.5.0, cuda 11.1 and cudnn 8.0.5
- To facilitate the software dependencies setup, [/APSNet/environment.yml]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/environment.yml) which illustrates all the libraries and dependencies versions is provided. It can be loaded as the initial setup in conda environment.
## SMART Dataset
We use Alibaba's [Large-scale SSD Failure Prediction Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95044) [1] as our SMART dataset. We only use Triple Level Cell (TLC) data in this dataset, which corresponds to SSD models MC1 and MC2.

|  | number of good SSD | number of bad SSD |
| --- | --- | --- |
| MC1 | 189147 | 10508 |
| MC2 | 22672 | 1131 |

The procedure for SMART data preprocessing is explained in [smart-preprocess/README.md]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/smart-preprocess/README.md).
The final output files of preprocessing are under [smart-preprocess/npy](https://github.com/YunfeiGu/APSNet/tree/main/smart-preprocess/npy). We notice that the number of MC2 is too small, and therefore we only use the following files:
- [smart-preprocess/npy/partial_statistics_mc1_model1.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/smart-preprocess/npy/partial_statistics_mc1_model1.npy)
- [smart-preprocess/npy/partial_statistics_mc1_model2.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/smart-preprocess/npy/partial_statistics_mc1_model2.npy)
- [smart-preprocess/npy/bad_mc1_model1.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/smart-preprocess/npy/bad_mc1_model1.npy)
- [smart-preprocess/npy/bad_mc1_model2.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/smart-preprocess/npy/bad_mc1_model2.npy)

[1] Xu, Fan, et al. "General Feature Selection for Failure Prediction in Large-scale SSD Deployment." 2021 51st Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN). IEEE, 2021.

## SSD Aging BEC Dataset
The SSD aging BEC Dataset is composed of the following files:
- [BEC_aging_data/aging.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/BEC_aging_data/aging.npy): contains 2048 aging time series. Please refer to our paper for detailed explanation on SSD aging BEC data, aging time series, and the process for SSD aging BEC data preprocessing.
- [badLoop7_4_45.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/BEC_aging_data/badLoop7_4_45.npy) and [goodLoop7_4_45.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/BEC_aging_data/goodLoop7_4_45.npy): contains the open-source normalized SSD aging BEC data over 20,000 SSD drives. These data are LUN fine-grained. This type of dataset represents the SSD wear-out degree corresponding to different P/E cycles. Since the SSD manufacturer NDA protocol, we open-source dataset corresponding to several certain P/E cycles. 

We hope that the released SSD Aging BEC Dataset will enable future studies to quantify the real-world SSD failure prediction research.

## Dataset Prepare
- This step is optional since all datasets used for machine learning models' training and testing process are already well-prepared and saved in [APSNet/data](https://github.com/YunfeiGu/APSNet/tree/main/data). 
- Due to the Intellectual Property policy of the Raw SSD BEC aging data, we only provide the normalized and well-processed SSD BEC aging data in [APSNet/data/aging.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/aging.npy).
- you can run the following scripts to execute all steps for preparing the SMART training and testing dataset: 

```shell
cd ./smart-preprocess
bash smart-preprocess.sh
cd ..
python3 slidingWindow.py
```
Note: while running silidingWindow.py, it will notify that 'Please input the length of days lookahead in range [3, 120]:'. In our experiment, we set N to 5, 7, 15, 30, 45, 60, 90 and 120.  After settling down the N value, it will generate the SMART training and testing dataset for evaluating learning models N_days_lookahead prediction performance.

In [/APSNet/data]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data), there are one npy file named aging.npy and 8 subfolders. [aging.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/aging.npy) is the SSD Aging BEC Dataset we proposed and collected. It is the dataset that needs to be fed in the sub-network 2.1b of APSNet. In each subfolder, there is an SSD SMART dataset that needs to be fed in the sub-network 2.1a of APSNet and other baseline models including RF, SVM, LR, DT, KNN, NN, and LSTM. (To understand the APSNet network structure and how these two types of dataset feed-in APSNet, please check Figure 8, Section III-B and III-C in our paper). 

These SMART datasets in each subfolder have been processed by [APSNet/slidingWindow.py]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/slidingWindow.py) already, in order to train and test all learning models for evaluating N days lookahead failure prediction performance.

For example, if you want to evaluate the learning models failure prediction performance in 5 days ahead on all metrics, you have to check out the following dataset: 

- [APSNet/data/aging.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/aging.npy): Processed SSD Aging BEC Dataset. It is used by the sub-network 2.1b of APSNet.
- [APSNet/data/5_days_lookahead/smart_train.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/5_days_lookahead/smart_train.npy): Processed SMART dataset used for all learning model training on SSD failure prediction in 5 days ahead. 
- [APSNet/data/5_days_lookahead/test_labels.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/5_days_lookahead/test_labels.npy): Labels of aforementioned SMART traininig dataset.
- [APSNet/data/5_days_lookahead/smart_test.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/smart_test.npy): Processed SMART dataset used for all learning model testing on SSD failure prediction in 5 days ahead. 
- [APSNet/data/5_days_lookahead/test_labels.npy]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/data/test_labels.npy): Labels of aforementioned SMART testing dataset.
## Instructions on Experiment using the Local Environment

### Initial Setup

First install the experimental environment with our [environment.yml](environment.yml):

```shell
conda env create -f environment.yml --name apsnet
```

Then activate the experimental environment:

```shell
conda activate apsnet
```
### Model Training
This step is also optional because all models have been well-trained and saved in [APSNet/trained_model]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/trained_model). The proposed key algorithm is APSNet. In the source code structure, APSNet contains two parts. One is Pnet which is the Pseudo Siamese Network in this network, the other one is RF which is used for RF-based adaptive discriminator. In Exp\#1, the baseline algorithms are RF, DT, LR, KNN, NN, LSTM, SVM. In Exp#2, the compared methods are the APSNet
with alternatives of sub-networks 2.a and 2.b, called APSNet_LL, APSNet_GG, and APSNet_GL respectively (Please check the Table XI in Section IV of our paper). In Exp\#3, the baseline is the APSNet without RF-based adaptive discriminator, called APSNet_NRF (Please check the Exp#3 in Section IV of our paper). The train source code of each model has been involved in [APSNet/main/model_src](https://github.com/YunfeiGu/APSNet/tree/main/model_src). Please read [APSNet/main/model_src/README.md](https://github.com/YunfeiGu/APSNet/tree/main/model_src/README.md) to understand how each learning model corresponds to each python file.

The commands for each machine learning model training:
```shell
cd ./model_src
```
- Train RF: 
```shell
python3 rf.py
```
- Train SVM: 
```shell
python3 svm.py
```
- Train DT: 
```shell
python3 dt.py
```
- Train KNN: 
```shell
python3 knn.py
```
- Train LR: 
```shell
python3 lr.py
```
- Train LSTM: 
```shell
python3 lstm.py
```
- Train NN: 
```shell
python3 nn.py
```
- Train PNet:
```shell
python3 pnet.py
```
- Train PNet_LL:
```shell
python3 pnet_2lstm.py
```
- Train PNet_GL:
```shell
python3 pnet_gru_lstm.py
```
- Train PNet_GG:
```shell
python3 pnet_2gru.py
```
Note: While running each model training script, it will notify that 'Please input the length of N days lookahead in the range [3, 120]:'. In our experiment, we set N to 5, 7, 15, 30, 45, 60, 90 and 120.  After settling down the N value, the training of the corresponding learning model will be trained and the well-trained model will be saved in the corresponding subfolder. 

For example, if you want to train the RF learning model for evaluating the performance on 5 days lookahead SSD failure prediction, input 5 and the well-trained RF learning model will be saved as [rf.pkl]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/trained_model/5_days_lookahead/rf.pkl) in subfolder [APSNet/trained_model/5_days_lookahead]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/trained_model/5_days_lookahead).

In each subfolder, there are 11 well-trained learning models:

- dt.pkl: well-trained decision tree learning model.
- knn.pkl: well-trained K-nearest neighbor learning model.
- lr.pkl: well-trained linear regression learning model.
- rf.pkl: well-trained random forest learning model.
- svm.pkl: well-trained support vector machine model.
- model_lstm_final.pkl: well-trained long-short term memory network learning model.
- model_nn_final.pkl: well-trained neural network model.
- model_pnet_final.pkl:well-trained proposed pseudo siamese network learning model.
- pnet_2gru.pkl: well-trained proposed pseudo siamese network with the alternative sub-networks. Both sub-netowrk 2.a and 2.b are GRU networks (explained in Section IV and Table XI of our paper).
- pnet_2lstm.pkl: well-trained proposed pseudo siamese network with the alternative sub-networks. Both sub-network 2.a and 2.b are LSTM network.
- pnet_gru_lstm.pkl: well-trained proposed pseudo siamese network with the alternative sub-networks. sub-netowrk 2.a is GRU, sub-network 2.b is LSTM.

### Model Testing and Reproducing Experiment Results

To reproduce all experiment results, we provide the jupyter notebook scripts to evaluate the failure prediction performance on all metrics in various days lookahead. Please read the check-list artifact check-list in Section~\ref{sec:check-list} and model testing instruction in the [APSNet/README.md]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/README.md) file. 
Note: While running each model testing script, it will notify that 'Please input the length of N days lookahead in the range [3, 120]:'. In our experiment, we set N to 5, 7, 15, 30, 45, 60, 90 and 120. After settling down the N value, the testing of the corresponding learning model will start and finally print the result of each metric in N days lookahead failure prediction. 
Use jupyter notebook to open each script and run. For example:
```shell
cd APSNet/
jupyter notebook --no-browser --port=8001
```
Use your local browser to open http://server\_ip:8001/
Then click 'RUN', and it will run the model testing and print all metrics' results

 - [metrics_apsnet.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_apsnet.ipynb)is a jupyter notebook script that used for testing our proposed APSNet on all metrics. 
 - [metrics_lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_lstm.ipynb) is a jupyter notebook script that used for testing lstm on all metrics.
 - [metrics_nn.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_nn.ipynb) is a jupyter notebook script that used for testing nn on all metrics.
 - [metrics_rf_knn_lr_dt_svm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_rf_knn_lr_dt_svm.ipynb) is a jupyter notebook script that used for testing RF, KNN, LR, DT, SVM learning models on all metrics. 
 - [metrics_pnet_2gru.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_pnet_2gru.ipynb) is a jupyter notebook script that used for testing APSNet_GG (explained in Section IV Exp#2 in our paper) model on metrics of TPR, FPR, AUC, F1-score.
 - [metrics_pnet_2lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_2lstm.ipynb) is a jupyter notebook script that used for testing APSNet\_LL (explained in Section IV Exp#2 in our paper) model on metrics of TPR, FPR, AUC, F1-score.
 - [metrics_gru_lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_gru_lstm.ipynb) is a jupyter notebook script that used for testing APSNet\_GL (explained in Section IV Exp#2 in our paper) model on metrics of TPR, FPR, AUC, F1-score.
 - [metrics_pnet.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_pnet.ipynb.ipynb) is a jupyter notebook script that used for testing APSNet\_NRF (explained in Section IV Exp#3 in our paper) model on metrics of TPR, FPR, AUC, F1-score.

The jupyter notebook scripts include:
For Exp\#1:

- [APSNet/metrics_lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_lstm.ipynb)
- [APSNet/metrics_nn.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_nn.ipynb)
- [APSNet/metrics_rf_knn_lr_dt_svm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_rf_knn_lr_dt_svm.ipynb)
- [APSNet/metrics_apsnet.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_apsnet.ipynb)

These jupyter notebook scripts are used for testing LSTM, NN, RF, KNN, LR, DT, SVM and proposed APSNet learning models on all metrics, including TPR, FPR, F1-score, AUC-score, Precision and AUC-ROC curve. 

Figure 11.a and Table VIII are statistics obtained from printed AUC results of these models testing via setting N days lookahead with various values. 

Figure 11.b and Table VII are statistics obtained from TPR results of these models testing via setting N days lookahead with various values. 

Figure 11.c and Table VII are statistics obtained from FPR results of these models testing via setting N days lookahead with various values. 

Figure 11.d and Table IX are statistics that obtained from F1-score results of these models testing via setting N days lookahead with various values. 

Table V is obtained from all precision results of these models while predicting failed and healthy SSD in various days ahead.

Figure 12.a are statistics that obtained from ROC curves printed by [APSNet/metricsapsnet.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_apsnet.ipynb) via setting N days lookahead with 5, 7, 14, 30, 45, 60, 90 and 120.

Figure 12.b Figure 12.c and Figure 12.d compare statistics that obtained from ROC curves printed by [APSNet/metrics_lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_lstm.ipynb) and [APSNet/metrics_nn.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_nn.ipynb) with which are got from [APSNet/metrics_apsnet.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_apsnet.ipynb) by setting N with 7, 30 and 60. 

Table X are statistics that are recorded while training all modeling as described in Section~\ref{sec:traininig} and testing models as mentioned above in this artifact evaluation instruction.

For Exp#2:
- [APSNet/metrics_pnet_2gru.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_pnet_2gru.ipynb)
- [APSNet/metrics\_pnet\_2lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_2lstm.ipynb)
- [APSNet/metrics_gru_lstm.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_gru_lstm.ipynb)

These jupyter scripts are used for testing APSNet_GG, APSNet_LL, and APSNet_GL models on several metrics, including TPR, FPR, F1-score, and AUC-score. 
Figure 13.a, Figure 13.b, Figure 13.c and Figure 13.d are statistics obtained from printed AUC, TPR, FPR, and F1-score results respectively of these models testing via setting N days lookahead with various values.

For Exp#3:
- [APSNet/metrics_pnet.ipynb]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/metrics_pnet.ipynb.ipynb)

These jupyter notebook scripts are used for testing APSNet_NRF models on several metrics, including TPR, FPR, F1-score, and AUC-score. 
Figure 14.a, Figure 14.b, Figure 14.c and Figure 14.d are statistics obtained from printed AUC, TPR, FPR, and F1-score results respectively of these models testing via setting N days lookahead with various values.

## Experiment using Container Image
- Container Environment: Docker 20.10.14.
- The libraries and dependencies have already been setup into the container image.
- The source code and dataset used for evaluation has already been involved into the container image.
- The open-source docker image is save in our Zenodo deposit [https://doi.org/10.5281/zenodo.6634964](https://doi.org/10.5281/zenodo.6634964), because of the size limitation of single data package in github. 
- DOI: 10.5281/zenodo.6634964
- This image is released under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license.

### Hardware Requirement
2 GHz dual core x86 processor or better, 32 GB system memory, 40GB free hard drive space, and NVIDIA GPU which could support CUDA 11.1 and cudnn 8.0.5.

### Software Requirement
Docker 20.10.14

### Download Container Image and Files
Download the docker image gzip file and python file in local environment
```shell
wget https://zenodo.org/record/6634964/files/apsnet.tar.gz
```
### Load and Run Docker Image
First unzip our container’s filesystem apsnet.tar.gz:
```shell
gzip -d apsnet.tar.gz
```

Then import our contents from apsnet.tar to create a filesystem image:
```shell
docker import apsnet.tar apsnet/ubuntu:v1
```

Then use the image which is just created to create a docker container:
```shell
docker run --rm --gpus all -it apsnet/ubuntu:v1 /bin/bash
```                                                     
To use ssh, you can add parameter -p xxxx:22 to map the container's port 22 to the host's port xxxx, and then you can use ssh to connect containers.
```shell
docker run -p xxxx:22 --rm --gpus all -it apsnet/ubuntu:v1 /bin/bash
```  
```shell
ssh root@127.0.0.1 -p xxxx
``` 
### File Structure in Docker Image
- After login the container, the main code is shown in directory: /root/apsnet
  ![location](https://user-images.githubusercontent.com/37954782/163811354-8dbc202f-7d5d-470f-b3a2-225d3585ac13.png)
 - The the file structure in docker image is the same as using local environment as described in [/APSNet/README.md]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/README.md), except environment.yml, because all soft dependencies have been included in the container already.

### Initial Setup
Activate the experimental environment:

```shell
conda activate apsnet
```
### Evaluation Instructions
All the artifact instructions using container are the same as using the local environment that is described in described in [/APSNet/README.md]( https://github.com/SJTU-Storage-Lab/SC2023AE_APSNet/tree/main/README.md).
