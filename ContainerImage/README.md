# Docker Image
Our open-source docker image for APSNet experiment reproduction is save in Zenodo deposit https://doi.org/10.5281/zenodo.6467485 because of the size limitation of single data package in github. 
- DOI: 10.5281/zenodo.6467485
- This image is released under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license.
## Experiment using Container Image
- Container Environment: Docker 20.10.14.
- The libraries and dependencies have already been setup into the container image.
- The source code and dataset used for evaluation has already been involved into the container image.
### Download Container Image and Files
Download the docker image gzip file and python file in local environment
```shell
wget https://zenodo.org/record/6467485/files/apsnet.tar.gz?download=1
wget https://zenodo.org/record/6467485/files/data_prepare.py?download=1
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
After login the container, the main code is shown in directory: /root/apsnet
  ![location](https://user-images.githubusercontent.com/37954782/163811354-8dbc202f-7d5d-470f-b3a2-225d3585ac13.png)
 
 - The folder [model_src](model\_src) contains learning models source code including learning model structures, training and evaluation.
 - The folder [data](data) contains the files needed for training and evaluation.
 - The folder [trained_model](trained\_model) contains the well-trained APSNet, RF, LSTM, NN, DT, LR, SVM, KNN learning models, which are pre-trained and could be used for direct evalution.
 - The folder [loss](loss) contains the loss records while training different learning models with various epoches.
 - The folder [log](log) contains the inference evaluation results logs of APSNet, RF, LSTM, NN, DT, LR, SVM, KNN learning model.

### Initial Setup
Activate the experimental environment:

```shell
conda activate apsnet
```
### Train/Test scripts
Train/Test scripts for the experiment using docker Image is the same as **Train/Test Scripts** for **Experiment using Local Environment**.

### N days Lookahead Failure Prediction Experiment
- First, copy the data_prepare.py into container direction root/apsnet/
```shell
scp -P xxxx data_file_path_in_host root@127.0.0.1:/root/apsnet/ 
```
- The following steps are the same as which are shown in **N days Lookahead Failure Prediction Experiment** for **Experiment using Local Environment** that has been illustrated in [APSNet/README.md](https://github.com/YunfeiGu/APSNet/blob/main/README.md)
