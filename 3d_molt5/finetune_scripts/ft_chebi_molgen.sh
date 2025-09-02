set -x

n_gpu=1
bsz=96
grad_acc=2
seed=42
ts=150000
ws=4500
lr=3e-4
dropout=0.1
fp_bits=4096
fp_level=3
test_bsz_multi=8
eps=-1

data_dir=QizhiPei/e3fp-chebi-molgen
# Download the pre-trained checkpoint from huggingface, and set the ckpt_path to .bin file, e.g.
# wget https://huggingface.co/QizhiPei/3d-molt5-base/resolve/main/pytorch_model.bin
ckpt_path=your_pretrained_ckpt_path

pre=/home/user_name/3d_molt5
setting=ft_chebi_molgen_fp
exp_name=$(date +%y%m%d)_${setting}_bits${fp_bits}_level${fp_level}_gpu${n_gpu}_bsz${bsz}_acc${grad_acc}_lr${lr}_eps${eps}_dp${dropout}_seed${seed}
save_dir=${pre}/checkpoints/3dt5/${setting}/${exp_name}

mkdir -p ${save_dir}/.hydra
echo "placeholder" > ${save_dir}/.hydra/config.yaml

echo "=============== start training ==============="


CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=${n_gpu} --master_port=29502 ../main.py \
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
    logging.every_steps=10 \
    checkpoint.every_steps=1000 \
    pred.every_steps=1000 \
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
    inst_format=true \