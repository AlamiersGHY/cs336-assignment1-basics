$env:PYTHONUTF8=1
$env:WANDB_MODE="online"
uv run python .\cs336_basics\main_train.py `
    --train_data_path .\data\TinyStoriesV2-GPT4-train.bin `
    --valid_data_path .\data\TinyStoriesV2-GPT4-valid.bin `
    --run_name "test" `
    --vocab_size 10000 `
    --num_layers 2  --num_heads 4 --d_model 128 --d_ff 512 `
    --max_iters 600 `
    --batch_size 32 `
    --context_length 64 `
    --lr 6e-4 `
    --min_lr 6e-5 `
    --warmup_iters 20 `
    --max_norm 1.0 `
    --seed 42 `
    --out_dir model_result\TinyStories_test `
    --wandb_project "cs336-pretraining-TinyStories-Ablations" `
    --device cpu

