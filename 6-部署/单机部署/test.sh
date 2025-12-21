curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "./model/sft_merge",
        "prompt": "你好，请介绍一下你自己",
        "max_tokens": 100
    }'