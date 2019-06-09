# This repo is the open source of A convolutional neural network based auto features extraction method for tea classification with electronic tongue
Figure 2 shows the implementation process of this work. First, sensors response of the e-tongue was converted to time-frequency maps by STFT. Second, the CNN extracted features automatically with time-frequency maps as input. Finally, the features extraction and classification results were carried out under a general shallow CNN architecture.

![image](https://github.com/Shunzhange/auto-feature-extraction-method-for-e-tongue/blob/master/figure/overview%20structure.jpg)

The key idea behind our method is to transform the time series into time-frequency map by appropriate strategy so as to make full use of the advantages of CNN in images features extraction and pattern recognition. The structure of proposed features extraction method is shown in Figure 7

![image](https://github.com/Shunzhange/auto-feature-extraction-method-for-e-tongue/blob/master/figure/feature%20extraction.jpg)

## It is implemented in pytorch. Please follow the instructions (Anaconda with python3.6 installation is recommended)
pytroch==0.4.0
torchvision==0.1.8
pillow==4.2.1

## Other libraries
CUDA Version == 9.0.176
Cudnn Version == 7.4.1
Ubuntu 14.04 or 16.04

## traing the model
python main.py

## Testing on saved model
python inference.py 

# experiment results 

In terms of Hamming window, the best average classification accuracy 99.8% is acquired in Figure 8(b) when the window size is 128.

![image](https://github.com/Shunzhange/auto-feature-extraction-method-for-e-tongue/blob/master/figure/results.jpg)
