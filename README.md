# JAX implementation of Separable PINN
### [Project Page](https://jwcho5576.github.io/spinn.github.io/) | [Paper](https://arxiv.org/abs/2306.15969)

[Junwoo Cho](https://github.com/jwcho5576)\*, 
[Seungtae Nam](https://github.com/stnamjef)\*, 
[Hyunmo Yang](https://github.com/extremebird),
[Youngjoon Hong](https://www.youngjoonhong.com/), 
[Seok-Bae Yun](https://seokbaeyun.wordpress.com/), 
[Eunbyun Park](https://silverbottlep.github.io/)&dagger;\
*Equal contribution, &dagger;Corresponding author.\
\
Conference on Neural Information Processing Systems (NeurIPS 2023)\
**Spotlight presentation**


https://github.com/stnamjef/SPINN/assets/94037424/46c4846a-680b-418c-80ef-2760d78402c4


# Architecture overview
![architecture](https://github.com/stnamjef/SPINN/assets/94037424/e0669832-d0ec-47f2-bb18-43d38490e6b6)


* SPINN consists of multiple MLPs, each of which takes an individual 1-dimensional coordinate as an input.
* The output is constructed by a simple product and summation.



# Environment Setup
#### 0. If you're using Google Colab, just run the code

#### 1. Install Docker and NVIDIA Container Toolkit
* please follow the official [document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for installation.
* if you already installed both of them, please skip this part.

#### 2. Build the docker image
* run the command below at "/your/path/to/SPINN".
* don't forget to include the dot at the end.
```
docker build -t spinn_environment .
```

#### 3. Run the docker image
* run the command below at "/your/path/to/SPINN".
```
docker run -it -v $(pwd):/workspace -p 8888:8888 --gpus all --ipc host --name spinn spinn_environment:latest
```

#### 4. (for demo code) Launch Jupyter and run the code
* run the command below inside the container.
```
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```

# Training
Run the command below:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python <EQUATIONnd>.py
```
You can also use our configurations by running the script files:
```
bash ./scripts/<EQUATIONnd_MODEL>.sh
```

# Navier-Stokes Reference Data
Find the original NS data from here: https://github.com/PredictiveIntelligenceLab/CausalPINNs/tree/main/data

# Citation

```
@article{cho2023separable,
  title={Separable Physics-Informed Neural Networks},
  author={Cho, Junwoo and Nam, Seungtae and Yang, Hyunmo and Yun, Seok-Bae and Hong, Youngjoon and Park, Eunbyung},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
