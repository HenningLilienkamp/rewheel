# rewheel
Another python based code library to train and employ machine learning models for image recognition.
This is a dummy codebase to demonstrate the following skills:
  - Object oriented programming in python
  - Deep learning with pytorch
  - Containerization with Docker
  - Version control with git

Current functionality is limited to classification of the MNIST handwritten digits dataset with a
Mulit Layer Perceptron model implemented in pytorch.

# Usage
The usage via the docker container is highly recommended:

# Step by step tutorial

[1] - Download the MNIST handwritten digits dataset from: https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/

[2] - Install docker: https://www.docker.com/

[2.5] - OPTIONAL: Install NVIDIA container toolkit to enable cuda GPU support: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

[3] - Get the rewheel docker container: 

`docker pull lilienka/python311-rewheel`

[4] - Launch the container:

`run -v /path/to/MNIST/data/:/MNIST_data/ --gpus all -it -p 8888:8888 python311-rewheel` (skip "--gpus all" if you skipped step 2.5)

[5] - Launch jupyter notebook:

`jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`

[6] - Explore rewheel via the jupyter notebook MNIST_classification_tutorial.ipynb in the docker container

