$env:PYTHONUTF8=1
$env:WANDB_MODE="online"
uv run python .\cs336_basics\main_train.py `
    --train_data_path .\data\TinyStoriesV2-GPT4-train.bin `
    --valid_data_path .\data\TinyStoriesV2-GPT4-valid.bin `
    --run_name "silu_bs32_lr6e4" `
    --vocab_size 10000 `
    --num_layers 4  --num_heads 8 --d_model 512 --d_ff 1344 `
    --max_iters 7000 `
    --batch_size 32 `
    --context_length 256 `
    --lr 6e-4 `
    --min_lr 6e-5 `
    --warmup_iters 700 `
    --max_norm 1.0 `
    --seed 42 `
    --ffn_type 'silu' `
    --out_dir model_result\TinyStories_silu `
    --wandb_project "cs336-pretraining-TinyStories-Ablations" `
    --device cuda

