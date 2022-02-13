'''
Modified implementation of LeNet-1 for MNIST inspired by [1] and heavily
influenced by [2]

Total Parameters: 3246


References:
[1] Y. Le Cun, B. Boser, J. S. Denker, R. E. Howard, W. Habbard, L. D. Jackel,
    and D. Henderson. 1990. Handwritten digit recognition with a
    back-propagation network. Advances in neural information processing
    systems 2. Morgan Kaufmann Publishers Inc.,
    San Francisco, CA, USA, 396–404.

[2] https://github.com/grvk/lenet-1

If you use this implementation in you work, please don't forget to mention the
original author, George Revkov.
'''

# Modified by Rishi Jha to match the experimental setup
# used in https://github.com/MadryLab/backdoor_data_poisoning

import torch
import torch.nn as nn

class LeNet1(nn.Sequential):
    
    def __init__(self):
        model_list = [
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(12 * 4 * 4, 10)
        ]

        super(LeNet1, self).__init__(*model_list)

        