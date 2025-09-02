set -x

n_gpu=4
bsz=32
grad_acc=1
seed=42
ts=1000000
ws=10000
lr=3e-4
dropout=0.1
fp_bits=4096
fp_level=3
test_bsz_multi=8
eps=100

# Will automatically load the 4 datasets from huggingface.
# See line 185-197 of 3d_molt5/utils/model_utils.py
data_dir=place_holder

# Download the pre-trained checkpoint from huggingface, and set the ckpt_path to .bin file, e.g.
# wget https://huggingface.co/QizhiPei/3d-molt5-base/resolve/main/pytorch_model.bin
ckpt_path=your_pretrained_ckpt_path

pre=/home/user_name/3d_molt5
setting=ft_3d_molm_generalist_fp
exp_name=$(date +%y%m%d)_${setting}_inst_bits${fp_bits}_level${fp_level}_gpu${n_gpu}_bsz${bsz}_acc${grad_acc}_lr${lr}_eps${eps}_dp${dropout}_seed${seed}
save_dir=${pre}/checkpoints/3dt5/${setting}/${exp_name}

mkdir -p ${save_dir}/.hydra
echo "placeholder" > ${save_dir}/.hydra/config.yaml

echo "=============== start training ==============="


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=${n_gpu} ../main.py \
    task=${setting} \
    seed=${seed} \
    hydra.run.dir=${save_dir} \
    optim.batch_size=${bsz} \
    optim.grad_acc=${grad_acc} \
    optim.total_steps=${ts} \
    optim.warmup_steps=${ws} \
    optim.base_lr=${lr} \
    logging.wandb=true \
    logging.wandb_project=${setting} \
    logging.wandb_group=${exp_name} \
    logging.wandb_run_name=${exp_name} \
    model.dropout=${dropout} \
    logging.every_steps=50 \
    checkpoint.every_epochs=1 \
    pred.every_epochs=1 \
    data.num_workers=8 \
    fp_bits=${fp_bits} \
    fp_level=${fp_level} \
    model.random_init=false \
    model.checkpoint_path=${ckpt_path} \
    optim.test_bsz_multi=${test_bsz_multi} \
    optim.epochs=${eps} \
    optim.lr_scheduler=linear \
    eval.every_steps=100000000 \
    data.data_dir=${data_dir} \
    inst_format=true