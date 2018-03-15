# Build CNNs with TensorFlow

## FC
Classify MNIST by 2 layers of *fully connected layers(FC)*.

|type|node|
|----|----|
|data|784|
|fc|500|
|fc|10|
## LeNet-5
Classify MNIST by 6 layers LeNet-5  
Paper:*Gradient-based learning applied to document recognition*.

|No.|type|node|size|
|----|----|----|----|
|0|data|28x28x1|1x1|
|1|conv|28x28x32|5x5|
|2|pool|14x14x32|2x2|
|3|conv|10x10x64|5x5|
|4|pool|5x5x64|2x2|
|5|fc|512x1|1x1|
|6|fc|10x1|1x1|