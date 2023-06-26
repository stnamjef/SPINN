# Jax implementation of Separable PINN
### [Arxiv](https://arxiv.org/abs/2211.08761)

[Junwoo Cho](https://github.com/jwcho5576)\*, 
[Seungtae Nam](https://github.com/stnamjef)\*, 
[Hyunmo Yang](https://github.com/extremebird),
[Youngjoon Hong](https://www.youngjoonhong.com/), 
[Seok-Bae Yun](https://seokbaeyun.wordpress.com/), 
[Eunbyun Park](https://silverbottlep.github.io/)&dagger;\
*Equal contribution, &dagger;Corresponding author.\
The Symbiosis of Deep Learning and Differential Equations (DLDE), NeurIPS 2022 Workshop.

# Architecture overview
![architecture](./assets/architecture.png)

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

#### 4. Launch Jupyter and run the code
* run the command below inside the container.
```
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```

# Training 
* you can run each experiment by running ```<EQUATION_Nd>.py```.
* to disable the memory preallocation, assign the environment variable ```XLA_PYTHON_CLIENT_PREALLOCATE``` to ```false```.
```
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python <EQUATION_Nd>.py --model=<MODEL> --equation=<EQUATION_Nd>
```
* you can also reproduce with our configurations by running the scripts ```SPINN/scripts/<EQUATION_Nd_MODEL>.sh```
* configurations   
```--data_dir```: directory to the reference data if needed   
```--model```: model (PINN or SPINN)   
```--equation```: name of the equation and the spatio-temporal dimension   
```--nc```: number of collocation points sampled from each axis   
```--nc_test```: number of test points sampled from each axis   
```--seed```: random seed   
```--lr```: learning rate   
```--epochs```: training epochs   
```--mlp```: type of MLP (mlp or modified_mlp)   
```--n_layers```: depth of each MLP   
```--features```: width of each MLP   
```--r```: rank of SPINN   
```--out_dim```: output dimension (channel size) of the model   
```--pos_enc```: size of the positional encoding (0 if not used)   
```--log_iter```: logging every ... epoch   
```--plot_iter```: visualize the solution every ... epoch   
```--a1``` ```--a2``` ```--a3```: (HELMHOLTZ EQUATION) frequency in the manufactured solution $\sin(a_1\pi x)+\sin(a_2\pi y)+\sin(a_3\pi z)$   
```--k```: (KLEIN_GORDON_EQUATION) temporal frequency of the manufactured solution   
```--nt```: (NAVIER_STOKES_EQUATION) number of sampled collocation points in the temporal domain   
```--nxy```: (NAVIER_STOKES_EQUATION 3D) number of sampled collocation points in the spatial domain   
```--offset_num```: (NAVIER_STOKES_EQUATION 3D) number of input grid set   
```--offset_iter```: (NAVIER_STOKES_EQUATION 3D) change the grid set every ... epoch   
```--marching_steps```: (NAVIER_STOKES_EQUATION 3D) number of time window   
```--step_idx```: (NAVIER_STOKES_EQUATION 3D) index of the time window   
```--lbda_c```: (NAVIER_STOKES_EQUATION 3D and 4D) weighting factor for incompressible condition loss   
```--lbda_ic```: (NAVIER_STOKES_EQUATION 3D and 4D) weighting factor for initial condition loss   




# Example (Klein-Gordon Eq.)
\\<!--#### Please visit our [project page](https://jwcho5576.github.io/spinn/) for more examples.-->

https://user-images.githubusercontent.com/47411051/217729201-7e0c2a1d-6d13-4352-9bd6-5054d8ead37d.mp4

# Citation

```
@inproceedings{choseparable,
  title={Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks},
  author={Cho, Junwoo and Nam, Seungtae and Yang, Hyunmo and Yun, Seok-Bae and Hong, Youngjoon and Park, Eunbyung},
  booktitle={The Symbiosis of Deep Learning and Differential Equations II}
}
```
