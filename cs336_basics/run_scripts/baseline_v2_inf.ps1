$env:PYTHONUTF8=1
uv run python .\cs336_basics\inference.py `
    --checkpoint_path .\model_result\TinyStories_baseline_v2\ckpt_final.pt `
    --tokenizer_dir .\data\TinyStoriesV2-GPT4-train `
    --vocab_size 10000 `
    --num_layers 4  --num_heads 8 --d_model 512 --d_ff 1344 --rope_theta 10000.0 `
    --batch_size 32 `
    --context_length 256 `
    --temperature 0.8 `
    --top_p 0.9 `
    --max_new_tokens 100 `
    --device cuda

