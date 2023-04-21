## SSD Aging BEC Dataset
The SSD aging BEC Dataset is composed of the following files:
- [BEC_aging_data/aging.npy](https://github.com/YunfeiGu/APSNet/blob/main/BEC_aging_data/aging.npy): contains 2048 aging time series. Please refer to our paper for detailed explanation on SSD aging BEC data, aging time series, and the process for SSD aging BEC data preprocessing.
- [badLoop7_4_45.npy](https://github.com/YunfeiGu/APSNet/blob/main/BEC_aging_data/badLoop7_4_45.npy) and [goodLoop7_4_45.npy](https://github.com/YunfeiGu/APSNet/blob/main/BEC_aging_data/goodLoop7_4_45.npy): contains the open-source normalized SSD aging BEC data over 20,000 SSD drives. These data are LUN fine-grained. This type of dataset represents the SSD wear-out degree corresponding to different P/E cycles. Since the SSD manufacturer NDA protocol, we open-source dataset corresponding to several certain P/E cycles. 

We hope that the released SSD Aging BEC Dataset will enable future studies to quantify the real-world SSD failure prediction research.
