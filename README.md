# A Tensorflow implementation of the "Recurrent Attention Model"

The **Recurrent Attention Model** (RAM) is introduced in [1] & [2]. 

It is inspired by the way humans perceive their surroundings, i.e. focusing on selective parts of the 
environment to acquire information and combining it, instead of observing the scene in its entirety.

In [1], the performance of the model is demonstrated by calssifying the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
In contrast to the existing approaches that processes the whole image, the **RAM** uses the information of *glimpses* at selected locations. 
These *glimpses* are then perceived in a retina-like representation to classify the given symbols.

As suggested in [1], [2], the network is trained using the REINFORCE [3] learning rule. 
The baseline is trained by reducing the mean squared error between the baseline and the received reward.

*In contrast to the model introduced in [1], not only the mean, but also the standard deviation of the location policy is learned.*

The code is inspired by [3], [4] & [5].

## Installation
**Required packages:**
1. [Numpy](http://www.numpy.org/)
2. [Tensorflow](https://www.tensorflow.org/)
3. [OpenCv](https://opencv.org/) for evaluation
4. [Matplotlib](http://matplotlib.org/) for plotting
5. [H5Py](http://www.h5py.org/) for saving the trained network weights

## Usage
The parameters for the training are all defined in the configuration files 
`run_mnist.py` and `run_translated_mnist.py`.


After training, the network-model is 
saved. It can be loaded for further training or evaluation.

## Evaluation
During training information about the current losses, accuracy 
and the behavior of the location network can be gathered using `tensorboard`. 
```
tensorboard --logdir=./summary
```

To create images of the glimpses that the network uses after training, simply execute the evaluation script.
The first parameter is the name of the configuration file and the second is the path to the network model.
```
evaluate.py run_mnist ./model/
```

To plot the accuracy of the classification over the number of trained epochs use the plotting script. 
```
python plot.py ./results.json
```

## Classification of the standard MNIST dataset
To train the network on classifying the standard MNIST dataset, 
start the training via the corresponding configuration file:
```
python run_mnist.py
```

**Current Highscore:  97.97% +/- 0.14 accuracy on the MNIST test-dataset.**

The plot below shows the accuracy for the test-dataset over the number of trained epochs. 
![Example](./MNIST_Results/MNIST_accuracy.png)

Examples of the images and the corresponding glimpses used by the network are displayed in the table.
 
|Originial Image | Glimpse 1| Glimpse 3| Glimpse 5 |Glimpse 6|
|:--:|:--:|:--:|:--:|:--:|
|<img src="./MNIST_Results/Images/symbol_2.png" alt="Symbol0" width="140">|<img src="./MNIST_Results/Images/symbol_2_glimpse_0_zoom_1.png" alt="Glimpse0" width="140">|<img src="./MNIST_Results/Images/symbol_2_glimpse_2_zoom_1.png" alt="Glimpse0" width="140">|<img src="./MNIST_Results/Images/symbol_2_glimpse_4_zoom_1.png" alt="Glimpse3" width="140">|<img src="./MNIST_Results/Images/symbol_2_glimpse_5_zoom_1.png" alt="Glimpse6" width="140">|
|<img src="./MNIST_Results/Images/symbol_5.png" alt="Symbol1" width="140">|<img src="./MNIST_Results/Images/symbol_5_glimpse_0_zoom_1.png" alt="Glimpse0" width="140">|<img src="./MNIST_Results/Images/symbol_5_glimpse_2_zoom_1.png" alt="Glimpse0" width="140">|<img src="./MNIST_Results/Images/symbol_5_glimpse_4_zoom_1.png" alt="Glimpse3" width="140">|<img src="./MNIST_Results/Images/symbol_5_glimpse_5_zoom_1.png" alt="Glimpse6" width="140">|

## Classification of the translated MNIST dataset
In [1], the network is tested on non-centered digits. 
Therefore, the digits forming the MNIST dataset are incorporated into a
larger image patch and then randomly translated.  

To train the network on classifying the "translated" MNIST dataset, 
start the code via the corresponding configuration file:
```
python run_translated_mnist.py
```

**Current Highscore:  97.5% +/- 0.16 accuracy on the translated MNIST test-dataset.**

The plot below shows the accuracy for the test-dataset over the number of trained epochs. 
![Example](./MNIST_translated_Results/MNIST_translated_accuracy.png)

Examples of the images and the corresponding glimpses used by the network are displayed in the table.
 
|Originial Image | Glimpse 1| Glimpse 2| Glimpse 5|Glimpse 7|
|:--:|:--:|:--:|:--:|:--:|
|<img src="./MNIST_translated_Results/Images/symbol_2.png" alt="Symbol0" width="140">|<img src="./MNIST_translated_Results/Images/symbol_2_glimpse_0.gif" alt="Glimpse0" width="140">|<img src="./MNIST_translated_Results/Images/symbol_2_glimpse_1.gif" alt="Glimpse1" width="140">|<img src="./MNIST_translated_Results/Images/symbol_2_glimpse_4.gif" alt="Glimpse2" width="140">|<img src="./MNIST_translated_Results/Images/symbol_2_glimpse_6.gif" alt="Glimpse3" width="140">|
|<img src="./MNIST_translated_Results/Images/symbol_7.png" alt="Symbol0" width="140">|<img src="./MNIST_translated_Results/Images/symbol_7_glimpse_0.gif" alt="Glimpse0" width="140">|<img src="./MNIST_translated_Results/Images/symbol_7_glimpse_1.gif" alt="Glimpse1" width="140">|<img src="./MNIST_translated_Results/Images/symbol_7_glimpse_4.gif" alt="Glimpse2" width="140">|<img src="./MNIST_translated_Results/Images/symbol_7_glimpse_6.gif" alt="Glimpse3" width="140">|

--------
[1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves. "Recurrent models of visual attention." Advances in neural information processing systems. 2014.

[2] Ba, Jimmy, Volodymyr Mnih, and Koray Kavukcuoglu. "Multiple object recognition with visual attention." arXiv preprint arXiv:1412.7755 (2014).

[3] Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

[4] https://github.com/jlindsey15/RAM

[5] https://github.com/zhongwen/RAM

[6] http://torch.ch/blog/2015/09/21/rmva.html

