# Advanced-Deep-Learning-Project


## Results

Model (Val Set) | Bleu-4 | Meteor | Cider
------------ | ------------- | ------------- | -------------
Base | 0.228 | 0.215 | 0.710
Unfreeze 100 | 0.226 | 0.211 | 0.697
Unfreeze 100 + augmentation | 0.229 | 0.213 | 0.706
**Unfreeze 100 + augmentation 100 epochs** | **0.236** | **0.215** | **0.729**
Unfreeze 100 + augmentation + Adam + dropout | 0.212 | 0.203 | 0.637
Unfreeze 100 + augmentation + Adam + dropout 100 epochs| 0.222 | 0.209 | 0.669
Attention | 0.038 | 0.102 | 0.061
Attention + Dropout | | | 
Unfreeze all | 0.210 | 0.200 | 0.616
L2 Reg. + Dropout + Stacked LSTM |  |  |
Batch Norm (Base) (lr:2) | 0.219 | 0.209 | 0.69
Batch Norm (Base) (lr:3) | 0.212 | 0.208 | 0.661

Model (Test Set) | Bleu-4 | Meteor | Cider
------------ | ------------- | ------------- | -------------
Test: Base | 0.223 | 0.214| 0.707
**Test: U-100-DA-100e** | **0.237** | **0.218** | **0.727**

