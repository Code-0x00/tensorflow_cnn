# Build CNNs with TensorFlow

## FC
Classify MNIST by 2 layers of *fully connected layers(FC)*.  
|type|node|
|----|----|
|data|784|
|full connected layer|500|
|full connected layer|10|
## LeNet-5
Classify MNIST by 6 layers LeNet-5  
Paper:*Gradient-based learning applied to document recognition*.
|No.|type|node|
|----|----|----|
|0|data|28x28x1|
|1|conv layer|32|
|2|pool layer|32|
|3|conv layer|64|
|4|pool layer|64|
|5|fc layer|512|
|6|fc layer|10|