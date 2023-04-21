# Model Source Code

## File description

### Part I: Baseline
The codes of baseline models contain training and testing process.

- [dt.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/dt.py):
  - Decision Tree
- [knn.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/knn.py):
  - K-Nearest Neighbor
- [lr.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/lr.py):
  - Linear Regression
- [lstm.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/lstm.py):
  - Long short-term memory
- [nn.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/nn.py):
  - Neural Networks 
- [rf.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/rf.py):
  - Random Forest
- [svm.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/svm.py):
  - Support Vector Machine

To run the code, using Random Forest as an example:
```shell
python rf.py
```

### Part II: APSNet

- [pnet.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/pnet.py):
  - To train the pseudo-siamese network part of apsnet
  - We have pre-trained and saved the  model in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
  - Trained pseudo-siamese network will be saved in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
- [apsnet.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/apsnet.py):
  - To test APSNet
  - A trained pseudo-siamese network is needed
- [pnet_2gru.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/pnet_2gru.py):
  - Pseudo-siamese network consisting of two GRUs
  - We have pre-trained and saved the  model in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
  - Trained pseudo-siamese network will be saved in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
- [pnet_2lstm.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/pnet_2lstm.py):
  - Pseudo-siamese network consisting of two LSTMs
  - We have pre-trained and saved the  model in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
  - Trained pseudo-siamese network will be saved in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
- [pnet_gru_lstm.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/pnet_gru_lstm.py):
  - Pseudo-siamese network consisting of a GRU and an LSTM
  - We have pre-trained and saved the  model in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)
  - Trained pseudo-siamese network will be saved in directory: [../trained_model](https://github.com/YunfeiGu/APSNet/tree/main/trained_model)

We have pre-trained and saved the pseudo-siamese network, so [apsnet.py](https://github.com/YunfeiGu/APSNet/blob/main/model_src/apsnet.py) can be run directly:
```shell
python apsnet.py
```

You can also train the pseudo-siamese network first, and then test it:
```shell
python pnet.py
python apsnet.py
```
