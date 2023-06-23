for i in 0 1 2 3 4 5 6 7 8 9
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python navier_stokes3d.py --data_dir=./data/navier_stokes --model=spinn --equation=navier_stokes3d --nt=32 --nxy=256 --seed=111 --lr=0.002 --epochs=100000 --mlp=modified_mlp --n_layers=3 --features=128 --r=128 --out_dim=2 --pos_enc=5 --offset_num=8 --offset_iter=100 --marching_steps=10 --step_idx=$i --log_iter=1000 --plot_iter=10000
done