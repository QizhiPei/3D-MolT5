# Download the corresponding checkpoint from huggingface, and set the ckpt_path to .bin file, e.g.
# wget https://huggingface.co/QizhiPei/3d-molt5-base-3d-molm-generalist/resolve/main/pytorch_model.bin
ckpt_path=your_ckpt_path
test_bsz_multi=8

declare -A settings

settings[ft_pubchem_cap_fp]="QizhiPei/e3fp-pubchem-cap 96 4"
settings[ft_pubchem_des_fp]="QizhiPei/e3fp-pubchem-des 96 4"
settings[ft_pubchem_com_fp]="QizhiPei/e3fp-pubchem-com 96 4"
settings[ft_pqc_prop_fp]="QizhiPei/e3fp-pubchemqc-prop 256 1"

for setting in "${!settings[@]}"; do
    vals=(${settings[$setting]})
    data_dir=${vals[0]}
    bsz=${vals[1]}
    grad_acc=${vals[2]}

    echo ">>> Running ${setting} ... data_dir: ${data_dir}, bsz: ${bsz}, grad_acc: ${grad_acc}"
    CUDA_VISIBLE_DEVICES=0 python ../main.py \
        predict_only=true \
        task=${setting} \
        hydra.run.dir=logs_eval/${setting} \
        data.data_dir=${data_dir} \
        optim.batch_size=${bsz} \
        optim.grad_acc=${grad_acc} \
        optim.test_bsz_multi=${test_bsz_multi} \
        model.random_init=false \
        model.checkpoint_path=${ckpt_path} \
        inst_format=true
done