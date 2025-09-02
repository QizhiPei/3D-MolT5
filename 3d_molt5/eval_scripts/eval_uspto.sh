setting=ft_uspto_fp

bsz=96
grad_acc=4
test_bsz_multi=8

data_dir=QizhiPei/e3fp-uspto-50k
# Download the corresponding checkpoint from huggingface, and set the ckpt_path to .bin file, e.g.
# wget https://huggingface.co/QizhiPei/3d-molt5-base-uspto-retro/resolve/main/pytorch_model.bin
ckpt_path=your_ckpt_path

CUDA_VISIBLE_DEVICES=0 python ../main.py \
    predict_only=true \
    task=ft_uspto_fp \
    hydra.run.dir=logs_eval/${setting} \
    data.data_dir=${data_dir} \
    optim.batch_size=${bsz} \
    optim.grad_acc=${grad_acc} \
    optim.test_bsz_multi=${test_bsz_multi} \
    model.random_init=false \
    model.checkpoint_path=${ckpt_path} \
    inst_format=true

