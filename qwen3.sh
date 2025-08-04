CUDA_VISIBLE_DEVICES=2 vllm serve /work/Qwen/Qwen3-Reranker-8B/ \
 --port 8000 \
 --served-model-name qwen \
 --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' >qwen3.log 2>&1 & 