setting=ft_molinst_react_fp

bsz=96
grad_acc=4
test_bsz_multi=8

# Download the corresponding checkpoint from huggingface, and set the ckpt_path to .bin file, e.g.
# wget https://huggingface.co/QizhiPei/3d-molt5-base-mol-instructions-react/resolve/main/pytorch_model.bin
ckpt_path=your_ckpt_path

for data_dir in QizhiPei/e3fp-mol-instructions-reagent-prediction QizhiPei/e3fp-mol-instructions-forward-reaction-prediction QizhiPei/e3fp-mol-instructions-retrosynthesis
do
    echo ">>> Running ${data_dir} ..."
    CUDA_VISIBLE_DEVICES=0 python ../main.py \
        predict_only=true \
        task=ft_molinst_react_fp \
        hydra.run.dir=logs_eval/${setting} \
        data.data_dir=${data_dir} \
        optim.batch_size=${bsz} \
        optim.grad_acc=${grad_acc} \
        optim.test_bsz_multi=${test_bsz_multi} \
        model.random_init=false \
        model.checkpoint_path=${ckpt_path} \
        inst_format=true
done