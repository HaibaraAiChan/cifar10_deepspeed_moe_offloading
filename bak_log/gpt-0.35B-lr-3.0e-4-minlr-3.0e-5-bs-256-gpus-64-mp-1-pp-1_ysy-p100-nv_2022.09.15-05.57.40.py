[2022-09-15 05:57:41,508] [WARNING] [runner.py:178:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2022-09-15 05:57:43,644] [INFO] [runner.py:504:main] cmd = /usr/bin/python3 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 /home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py --override-lr-scheduler --adam-beta1 0.9 --adam-beta2 0.95 --tensor-model-parallel-size 1 --moe-expert-parallel-size 1 --num-experts 1 --moe-loss-coeff 0.01 --moe-train-capacity-factor 1.0 --moe-eval-capacity-factor 1.0 --moe-min-capacity 4 --init-method-std 0.014 --lr-decay-tokens 260000000000 --lr-warmup-tokens 375000000 --micro-batch-size 4 --exit-duration-in-mins 30000000 --global-batch-size 256 --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 2048 --max-position-embeddings 2048 --train-tokens 300000000000 --train-samples 439453125 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine --split 98,2,0 --log-interval 10 --eval-interval 100 --eval-iters 10 --save-interval 1000 --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 --num-workers 0 --fp16 --load /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1 --save /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1 --tensorboard-queue-size 1 --log-timers-to-tensorboard --log-batch-size-to-tensorboard --log-validation-ppl-to-tensorboard --tensorboard-dir /home/cc/Megatron-DeepSpeed/examples/MoE/output/tensorboard/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1_ysy-p100-nv_2022.09.15-05.57.40 --checkpoint-activations --vocab-file ./data/the_pile_public_merged_nopreprocessing/gpt2-vocab.json --merge-file ./data/the_pile_public_merged_nopreprocessing/gpt2-merges.txt --data-path /data/the_pile_public_merged_nopreprocessing/pile_text_document --data-impl mmap --deepspeed --deepspeed_config ds_config_gpt_gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1.json --pipeline-model-parallel-size 1 --deepspeed-activation-checkpointing
[2022-09-15 05:57:44,795] [INFO] [launch.py:136:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2022-09-15 05:57:44,795] [INFO] [launch.py:143:main] nnodes=1, num_local_procs=4, node_rank=0
[2022-09-15 05:57:44,795] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2022-09-15 05:57:44,795] [INFO] [launch.py:156:main] dist_world_size=4
[2022-09-15 05:57:44,795] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [92m[OKAY][0m
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [92m[YES][0m ...... [92m[OKAY][0m
cpu_adagrad ............ [92m[YES][0m ...... [92m[OKAY][0m
fused_adam ............. [92m[YES][0m ...... [92m[OKAY][0m
fused_lamb ............. [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [92m[OKAY][0m
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [92m[YES][0m ...... [92m[OKAY][0m
cpu_adagrad ............ [92m[YES][0m ...... [92m[OKAY][0m
fused_adam ............. [92m[YES][0m ...... [92m[OKAY][0m
fused_lamb ............. [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [92m[OKAY][0m
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [92m[YES][0m ...... [92m[OKAY][0m
cpu_adagrad ............ [92m[YES][0m ...... [92m[OKAY][0m
fused_adam ............. [92m[YES][0m ...... [92m[OKAY][0m
fused_lamb ............. [92m[YES][0m ...... [92m[OKAY][0m
sparse_attn ............ [92m[YES][0m ...... [92m[OKAY][0m
transformer ............ [92m[YES][0m ...... [92m[OKAY][0m
stochastic_transformer . [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [92m[OKAY][0m
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [92m[YES][0m ...... [92m[OKAY][0m
cpu_adagrad ............ [92m[YES][0m ...... [92m[OKAY][0m
fused_adam ............. [92m[YES][0m ...... [92m[OKAY][0m
fused_lamb ............. [92m[YES][0m ...... [92m[OKAY][0m
sparse_attn ............ [92m[YES][0m ...... [92m[OKAY][0m
transformer ............ [92m[YES][0m ...... [92m[OKAY][0m
stochastic_transformer . [92m[YES][0m ...... [92m[OKAY][0m
sparse_attn ............ [92m[YES][0m ...... [92m[OKAY][0m
transformer ............ [92m[YES][0m ...... [92m[OKAY][0m
stochastic_transformer . [92m[YES][0m ...... [92m[OKAY][0m
sparse_attn ............ [92m[YES][0m ...... [92m[OKAY][0m
transformer ............ [92m[YES][0m ...... [92m[OKAY][0m
stochastic_transformer . [92m[YES][0m ...... [92m[OKAY][0m
async_io ............... [92m[YES][0m ...... [92m[OKAY][0m
utils .................. [92m[YES][0m ...... [92m[OKAY][0m
quantizer .............. [92m[YES][0m ...... [92m[OKAY][0m
async_io ............... [92m[YES][0m ...... [92m[OKAY][0m
utils .................. [92m[YES][0m ...... [92m[OKAY][0m
quantizer .............. [92m[YES][0m ...... [92m[OKAY][0m
transformer_inference .. [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
async_io ............... [92m[YES][0m ...... [92m[OKAY][0m
utils .................. [92m[YES][0m ...... [92m[OKAY][0m
quantizer .............. [92m[YES][0m ...... [92m[OKAY][0m
async_io ............... [92m[YES][0m ...... [92m[OKAY][0m
utils .................. [92m[YES][0m ...... [92m[OKAY][0m
quantizer .............. [92m[YES][0m ...... [92m[OKAY][0m
DeepSpeed general environment info:
torch install path ............... ['/home/cc/.local/lib/python3.6/site-packages/torch']
torch version .................... 1.9.1+cu111
torch cuda version ............... 11.1
torch hip version ................ None
nvcc version ..................... 11.1
deepspeed install path ........... ['/home/cc/.local/lib/python3.6/site-packages/deepspeed']
deepspeed info ................... 0.7.3+a691ec60, a691ec60, master
deepspeed wheel compiled w. ...... torch 1.9, cuda 11.1
transformer_inference .. [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
transformer_inference .. [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/cc/.local/lib/python3.6/site-packages/torch']
torch version .................... 1.9.1+cu111
torch cuda version ............... 11.1
torch hip version ................ None
nvcc version ..................... 11.1
deepspeed install path ........... ['/home/cc/.local/lib/python3.6/site-packages/deepspeed']
deepspeed info ................... 0.7.3+a691ec60, a691ec60, master
deepspeed wheel compiled w. ...... torch 1.9, cuda 11.1
transformer_inference .. [92m[YES][0m ...... [92m[OKAY][0m
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/cc/.local/lib/python3.6/site-packages/torch']
torch version .................... 1.9.1+cu111
torch cuda version ............... 11.1
torch hip version ................ None
nvcc version ..................... 11.1
deepspeed install path ........... ['/home/cc/.local/lib/python3.6/site-packages/deepspeed']
deepspeed info ................... 0.7.3+a691ec60, a691ec60, master
deepspeed wheel compiled w. ...... torch 1.9, cuda 11.1
DeepSpeed general environment info:
torch install path ............... ['/home/cc/.local/lib/python3.6/site-packages/torch']
torch version .................... 1.9.1+cu111
torch cuda version ............... 11.1
torch hip version ................ None
nvcc version ..................... 11.1
deepspeed install path ........... ['/home/cc/.local/lib/python3.6/site-packages/deepspeed']
deepspeed info ................... 0.7.3+a691ec60, a691ec60, master
deepspeed wheel compiled w. ...... torch 1.9, cuda 11.1
**** Git info for Megatron: git_hash=5e0f373 git_branch=main ****
using world size: 4, data-parallel-size: 4, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.95
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  aml_data_download_path .......................... None
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_in_cpu ............................... False
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  compression_training ............................ False
  consumed_train_samples .......................... 0
  consumed_train_tokens ........................... 0
  consumed_valid_samples .......................... 0
  contigious_checkpointing ........................ False
  cpu_optimizer ................................... False
  cpu_torch_adam .................................. False
  create_moe_param_group .......................... False
  curriculum_learning ............................. False
  custom_token_counting ........................... False
  data_impl ....................................... mmap
  data_parallel_size .............................. 4
  data_path ....................................... ['/data/the_pile_public_merged_nopreprocessing/pile_text_document']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  deepscale ....................................... False
  deepscale_config ................................ None
  deepspeed ....................................... True
  deepspeed_activation_checkpointing .............. True
  deepspeed_config ................................ ds_config_gpt_gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1.json
  deepspeed_mpi ................................... False
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  ds_inference .................................... False
  ds_pipeline_enabled ............................. True
  embedding_path .................................. None
  enable_expert_tensor_parallelism ................ False
  encoder_seq_length .............................. 2048
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... 30000000
  exit_interval ................................... None
  expert_interval ................................. 2
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 256
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hidden_size_teacher ............................. None
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference ....................................... False
  init_method_std ................................. 0.014
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kd .............................................. False
  kd_alpha_ce ..................................... 1
  kd_beta_ce ...................................... 1
  kd_temp ......................................... 1.0
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1
  load_teacher .................................... None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... True
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_num_zeros_in_grad ........................... False
  log_optimizer_states_to_tensorboard ............. False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... True
  log_validation_ppl_to_tensorboard ............... True
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0003
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_decay_tokens ................................. 260000000000
  lr_warmup_fraction .............................. None
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  lr_warmup_tokens ................................ 375000000
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 2048
  memory_centric_tiled_linear ..................... False
  merge_file ...................................... ./data/the_pile_public_merged_nopreprocessing/gpt2-merges.txt
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 3e-05
  mlp_type ........................................ standard
  mmap_warmup ..................................... False
  moe_eval_capacity_factor ........................ 1.0
  moe_expert_parallel_size ........................ 1
  moe_loss_coeff .................................. 0.01
  moe_min_capacity ................................ 4
  moe_token_dropping .............................. True
  moe_train_capacity_factor ....................... 1.0
  mos ............................................. False
  no_load_lr_state ................................ False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_pipeline_parallel ............................ False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_attention_heads_teacher ..................... None
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... [1]
  num_experts_teacher ............................. [1]
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_layers_teacher .............................. None
  num_workers ..................................... 0
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... True
  params_dtype .................................... torch.float16
  partition_activations ........................... False
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 1
  profile_backward ................................ False
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  remote_device ................................... none
  reset_attention_mask ............................ False
  reset_iteration ................................. False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1
  save_interval ................................... 1000
  scatter_gather_tensors_in_pipeline .............. True
  scattered_embeddings ............................ False
  seed ............................................ 1234
  seq_length ...................................... 2048
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 98,2,0
  split_transformers .............................. False
  synchronize_each_layer .......................... False
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. /home/cc/Megatron-DeepSpeed/examples/MoE/output/tensorboard/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1_ysy-p100-nv_2022.09.15-05.57.40
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1
  tile_factor ..................................... 1
  titles_data_path ................................ None
  tokenizer_type .................................. GPT2BPETokenizer
  topk ............................................ 1
  train_iters ..................................... None
  train_samples ................................... 439453125
  train_tokens .................................... 300000000000
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  use_pin_memory .................................. False
  use_tutel ....................................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... ./data/the_pile_public_merged_nopreprocessing/gpt2-vocab.json
  weight_decay .................................... 0.1
  world_size ...................................... 4
  zero_allgather_bucket_size ...................... 0.0
  zero_contigious_gradients ....................... False
  zero_reduce_bucket_size ......................... 0.0
  zero_reduce_scatter ............................. False
  zero_stage ...................................... 1.0
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 16
> building GPT2BPETokenizer tokenizer ...
**** Git info for Megatron: git_hash=5e0f373 git_branch=main ****
**** Git info for Megatron: git_hash=5e0f373 git_branch=main ****
**** Git info for Megatron: git_hash=5e0f373 git_branch=main ****
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> initializing torch distributed ...
[2022-09-15 05:57:47,081] [INFO] [comm.py:635:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
WARNING: TensorBoard writing requested but is not available (are you using PyTorch 1.1.0 or later?), no TensorBoard logs will be written.
> initializing tensor model parallel with size 1
> initializing pipeline model parallel with size 1
> setting random seeds to 1234 ...
[2022-09-15 05:57:47,261] [INFO] [checkpointing.py:234:model_parallel_cuda_manual_seed] > initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
[W ProcessGroupNCCL.cpp:1569] Rank 2 using best-guess GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
> compiling dataset index builder ...
[W ProcessGroupNCCL.cpp:1569] Rank 1 using best-guess GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
[W ProcessGroupNCCL.cpp:1569] Rank 3 using best-guess GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
make: Entering directory '/home/cc/Megatron-DeepSpeed/megatron/data'
g++ -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -I/usr/include/python3.6m -I/home/cc/.local/lib/python3.6/site-packages/pybind11/include helpers.cpp -o helpers.cpython-36m-x86_64-linux-gnu.so
make: Leaving directory '/home/cc/Megatron-DeepSpeed/megatron/data'
>>> done with dataset index builder. Compilation time: 5.171 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF scaled_upper_triang_masked_softmax.o.d -DTORCH_EXTENSION_NAME=scaled_upper_triang_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/TH -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.6m -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++14 -O3 -c /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/scaled_upper_triang_masked_softmax.cpp -o scaled_upper_triang_masked_softmax.o 
[2/3] /usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=scaled_upper_triang_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/TH -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.6m -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_60,code=sm_60 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -std=c++14 -c /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/scaled_upper_triang_masked_softmax_cuda.cu -o scaled_upper_triang_masked_softmax_cuda.cuda.o 
[3/3] c++ scaled_upper_triang_masked_softmax.o scaled_upper_triang_masked_softmax_cuda.cuda.o -shared -L/home/cc/.local/lib/python3.6/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda_cu -ltorch_cuda_cpp -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o scaled_upper_triang_masked_softmax_cuda.so
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF scaled_masked_softmax.o.d -DTORCH_EXTENSION_NAME=scaled_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/TH -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.6m -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++14 -O3 -c /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/scaled_masked_softmax.cpp -o scaled_masked_softmax.o 
[2/3] /usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=scaled_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/TH -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.6m -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_60,code=sm_60 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -std=c++14 -c /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/scaled_masked_softmax_cuda.cu -o scaled_masked_softmax_cuda.cuda.o 
[3/3] c++ scaled_masked_softmax.o scaled_masked_softmax_cuda.cuda.o -shared -L/home/cc/.local/lib/python3.6/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda_cu -ltorch_cuda_cpp -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o scaled_masked_softmax_cuda.so
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/build/build.ninja...
Building extension module fused_mix_prec_layer_norm_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF layer_norm_cuda.o.d -DTORCH_EXTENSION_NAME=fused_mix_prec_layer_norm_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/TH -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.6m -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++14 -O3 -c /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda.cpp -o layer_norm_cuda.o 
[2/3] /usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=fused_mix_prec_layer_norm_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/TH -isystem /home/cc/.local/lib/python3.6/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.6m -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_60,code=sm_60 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -maxrregcount=50 -gencode arch=compute_80,code=sm_80 -std=c++14 -c /home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu -o layer_norm_cuda_kernel.cuda.o 
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In function ‘void cuda_layer_norm(at::Tensor*, at::Tensor*, at::Tensor*, at::Tensor*, int, int, c10::IntList, at::Tensor*, at::Tensor*, double)’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:224: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:247: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:272: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:296: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                        ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:359: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:414: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                              ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:119: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:142: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                              ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:167: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:191: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                               ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:258: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                  ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:317: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                             ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:131: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                   ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:154: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                          ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:179: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                   ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:203: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                           ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:274: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                  ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:337: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                 ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:151: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:174: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                              ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:199: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:227: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                   ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:294: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:353: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                 ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:166: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:189: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                             ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:214: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:246: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:317: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                             ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:725:380: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                            ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In function ‘void cuda_layer_norm_gradient(at::Tensor*, at::Tensor*, at::Tensor*, at::Tensor*, int, int, c10::IntList, at::Tensor*, at::Tensor*, double, at::Tensor*, at::Tensor*, at::Tensor*)’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:224: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:247: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:272: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:333: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                             ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:389: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                     ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:438: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:489: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:550: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:120: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                        ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:143: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                               ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:168: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                        ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:233: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                         ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:293: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                     ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:342: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                      ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:397: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                             ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:462: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:132: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                    ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:155: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                           ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:180: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                    ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:249: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                         ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:313: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                         ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:362: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                          ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:421: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:490: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:152: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                        ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:175: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                               ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:200: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                        ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:265: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                         ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:325: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                     ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:378: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                          ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:433: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:498: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:167: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:190: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                              ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:215: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                       ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:284: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                            ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:348: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                            ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:405: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                     ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:464: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:533: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = float; U = float; V = float]’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:562:   required from here
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:801:97: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = float; U = float; V = c10::Half]’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:474:   required from here
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:801:97: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = float; U = float; V = c10::BFloat16]’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:502:   required from here
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:801:97: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = c10::Half; U = float; V = c10::Half]’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:510:   required from here
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:801:97: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = c10::BFloat16; U = float; V = c10::BFloat16]’:
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:833:545:   required from here
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:770:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:783:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/cc/Megatron-DeepSpeed/megatron/fused_kernels/layer_norm_cuda_kernel.cu:801:97: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/home/cc/.local/lib/python3.6/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
[3/3] c++ layer_norm_cuda.o layer_norm_cuda_kernel.cuda.o -shared -L/home/cc/.local/lib/python3.6/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda_cu -ltorch_cuda_cpp -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o fused_mix_prec_layer_norm_cuda.so
Loading extension module fused_mix_prec_layer_norm_cuda...
[W ProcessGroupNCCL.cpp:1569] Rank 0 using best-guess GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
>>> done with compiling and loading fused kernels. Compilation time: 216.525 seconds
time to initialize megatron (seconds): 184.960
[after megatron is initialized] datetime: 2022-09-15 06:01:28 
building GPT model ...
[2022-09-15 06:01:29,045] [INFO] [utils.py:827:see_memory_usage] Before Building Model
[2022-09-15 06:01:29,046] [INFO] [utils.py:832:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 0.0 GB         Max_CA 0 GB 
[2022-09-15 06:01:29,046] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 12.95 GB, percent = 10.3%
SEED_LAYERS=False BASE_SEED=1234 SEED_FN=None
Using topology: {ProcessCoord(pipe=0, data=0, model=0): 0, ProcessCoord(pipe=0, data=1, model=0): 1, ProcessCoord(pipe=0, data=2, model=0): 2, ProcessCoord(pipe=0, data=3, model=0): 3}
[2022-09-15 06:01:29,165] [INFO] [module.py:366:_partition_layers] Partitioning pipeline stages with method type:transformer
stage=0 layers=31
     0: _to_float16
     1: EmbeddingPipe
     2: <lambda>
     3: ParallelTransformerLayerPipe
     4: ParallelTransformerLayerPipe
     5: ParallelTransformerLayerPipe
     6: ParallelTransformerLayerPipe
     7: ParallelTransformerLayerPipe
     8: ParallelTransformerLayerPipe
     9: ParallelTransformerLayerPipe
    10: ParallelTransformerLayerPipe
    11: ParallelTransformerLayerPipe
    12: ParallelTransformerLayerPipe
    13: ParallelTransformerLayerPipe
    14: ParallelTransformerLayerPipe
    15: ParallelTransformerLayerPipe
    16: ParallelTransformerLayerPipe
    17: ParallelTransformerLayerPipe
    18: ParallelTransformerLayerPipe
    19: ParallelTransformerLayerPipe
    20: ParallelTransformerLayerPipe
    21: ParallelTransformerLayerPipe
    22: ParallelTransformerLayerPipe
    23: ParallelTransformerLayerPipe
    24: ParallelTransformerLayerPipe
    25: ParallelTransformerLayerPipe
    26: ParallelTransformerLayerPipe
    27: <lambda>
    28: MixedFusedLayerNorm
    29: EmbeddingPipe
    30: float16_to_fp32
  loss: CrossEntropy
[2022-09-15 06:01:29,282] [INFO] [utils.py:827:see_memory_usage] After Building Model
[2022-09-15 06:01:29,283] [INFO] [utils.py:832:see_memory_usage] MA 0.68 GB         Max_MA 0.7 GB         CA 0.74 GB         Max_CA 1 GB 
[2022-09-15 06:01:29,284] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 12.99 GB, percent = 10.3%
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 355919872
setting training iterations to 1716613
> learning rate decay style: cosine
DeepSpeed is enabled.
[2022-09-15 06:01:29,287] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed info: version=0.7.3+a691ec60, git-hash=a691ec60, git-branch=master
[2022-09-15 06:01:29,584] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2022-09-15 06:01:29,585] [INFO] [logging.py:68:log_dist] [Rank 0] Removing param_group that has no 'params' in the client Optimizer
[2022-09-15 06:01:29,585] [INFO] [logging.py:68:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2022-09-15 06:01:29,595] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed Basic Optimizer = {basic_optimizer.__class__.__name__}
[2022-09-15 06:01:29,596] [INFO] [logging.py:68:log_dist] [Rank 0] Creating fp16 optimizer with dynamic loss scale
[2022-09-15 06:01:29,655] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed Final Optimizer = FusedAdam
[2022-09-15 06:01:29,656] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2022-09-15 06:01:29,656] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed LR Scheduler = <megatron.learning_rates.AnnealingLR object at 0x7f6cf8d658d0>
[2022-09-15 06:01:29,656] [INFO] [logging.py:68:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0, 0.0], mom=[(0.9, 0.95), (0.9, 0.95)]
[2022-09-15 06:01:29,657] [INFO] [config.py:987:print] DeepSpeedEngine configuration:
[2022-09-15 06:01:29,658] [INFO] [config.py:991:print]   activation_checkpointing_config  {
    "partition_activations": false, 
    "contiguous_memory_optimization": false, 
    "cpu_checkpointing": false, 
    "number_checkpoints": null, 
    "synchronize_checkpoint_boundary": false, 
    "profile": false
}
[2022-09-15 06:01:29,658] [INFO] [config.py:991:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2022-09-15 06:01:29,658] [INFO] [config.py:991:print]   amp_enabled .................. False
[2022-09-15 06:01:29,658] [INFO] [config.py:991:print]   amp_params ................... False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   autotuning_config ............ {
    "enabled": false, 
    "start_step": null, 
    "end_step": null, 
    "metric_path": null, 
    "arg_mappings": null, 
    "metric": "throughput", 
    "model_info": null, 
    "results_dir": null, 
    "exps_dir": null, 
    "overwrite": true, 
    "fast": true, 
    "start_profile_step": 3, 
    "end_profile_step": 5, 
    "tuner_type": "gridsearch", 
    "tuner_early_stopping": 5, 
    "tuner_num_trials": 50, 
    "model_info_path": null, 
    "mp_size": 1, 
    "max_train_batch_size": null, 
    "min_train_batch_size": 1, 
    "max_train_micro_batch_size_per_gpu": 1.024000e+03, 
    "min_train_micro_batch_size_per_gpu": 1, 
    "num_tuning_micro_batch_sizes": 3
}
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   bfloat16_enabled ............. False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   checkpoint_tag_validation_enabled  True
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   checkpoint_tag_validation_fail  False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f6d01754518>
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   communication_data_type ...... None
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   curriculum_enabled ........... False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   curriculum_params ............ {'curriculum_type': 'seqlen', 'min_difficulty': 80, 'max_difficulty': 2048, 'schedule_type': 'fixed_linear', 'schedule_config': {'total_curriculum_step': 220277, 'difficulty_step': 8}}
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   dataloader_drop_last ......... False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   disable_allgather ............ False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   dump_state ................... False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   dynamic_loss_scale_args ...... {'init_scale': 2048, 'scale_window': 500, 'delayed_shift': 2, 'min_scale': 1}
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   eigenvalue_enabled ........... False
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   eigenvalue_gas_boundary_resolution  1
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   eigenvalue_layer_num ......... 0
[2022-09-15 06:01:29,659] [INFO] [config.py:991:print]   eigenvalue_max_iter .......... 100
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   eigenvalue_stability ......... 1e-06
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   eigenvalue_tol ............... 0.01
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   eigenvalue_verbose ........... False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   elasticity_enabled ........... False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   flops_profiler_config ........ {
    "enabled": false, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   fp16_auto_cast ............... False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   fp16_enabled ................. True
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   fp16_master_weights_and_gradients  False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   global_rank .................. 0
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   gradient_accumulation_steps .. 16
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   gradient_clipping ............ 1.0
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   gradient_predivide_factor .... 1.0
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   initial_dynamic_scale ........ 2048
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   load_universal_checkpoint .... False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   loss_scale ................... 0
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   memory_breakdown ............. False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   monitor_config ............... <deepspeed.monitor.config.DeepSpeedMonitorConfig object at 0x7f6d017546d8>
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   nebula_config ................ {
    "enabled": false, 
    "persistent_storage_path": null, 
    "persistent_time_interval": 100, 
    "num_of_version_in_retention": 2, 
    "enable_nebula_load": true, 
    "load_path": null
}
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   optimizer_legacy_fusion ...... False
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   optimizer_name ............... None
[2022-09-15 06:01:29,660] [INFO] [config.py:991:print]   optimizer_params ............. None
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   pld_enabled .................. False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   pld_params ................... False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   prescale_gradients ........... True
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   scheduler_name ............... None
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   scheduler_params ............. None
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   sparse_attention ............. None
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   sparse_gradients_enabled ..... False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   steps_per_print .............. 10
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   train_batch_size ............. 256
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   train_micro_batch_size_per_gpu  4
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   wall_clock_breakdown ......... False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   world_size ................... 4
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   zero_allow_untested_optimizer  False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   zero_config .................. stage=0 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=True offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   zero_enabled ................. False
[2022-09-15 06:01:29,661] [INFO] [config.py:991:print]   zero_optimization_stage ...... 0
[2022-09-15 06:01:29,661] [INFO] [config.py:983:print_user_config]   json = {
    "train_batch_size": 256, 
    "train_micro_batch_size_per_gpu": 4, 
    "steps_per_print": 10, 
    "zero_optimization": {
        "stage": 0, 
        "elastic_checkpoint": true
    }, 
    "gradient_clipping": 1.0, 
    "prescale_gradients": true, 
    "fp16": {
        "enabled": true, 
        "loss_scale": 0, 
        "loss_scale_window": 500, 
        "hysteresis": 2, 
        "min_loss_scale": 1, 
        "initial_scale_power": 11
    }, 
    "bf16": {
        "enabled": false
    }, 
    "curriculum_learning": {
        "enabled": false, 
        "curriculum_type": "seqlen", 
        "min_difficulty": 80, 
        "max_difficulty": 2.048000e+03, 
        "schedule_type": "fixed_linear", 
        "schedule_config": {
            "total_curriculum_step": 2.202770e+05, 
            "difficulty_step": 8
        }
    }, 
    "wall_clock_breakdown": false
}
[2022-09-15 06:01:29,663] [INFO] [engine.py:87:__init__] CONFIG: micro_batches=16 micro_batch_size=4
[2022-09-15 06:01:29,720] [INFO] [engine.py:145:__init__] RANK=0 STAGE=0 LAYERS=31 [0, 31) STAGE_PARAMS=355919872 (355.920M) TOTAL_PARAMS=355919872 (355.920M) UNIQUE_PARAMS=355919872 (355.920M)
[2022-09-15 06:01:29,734] [WARNING] [engine.py:2572:load_checkpoint] Unable to find latest file at /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1/latest, if trying to load latest checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint.
WARNING: could not find the metadata file /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1 
    will not load any checkpoints and will start from random
[2022-09-15 06:01:29,734] [WARNING] [engine.py:2572:load_checkpoint] Unable to find latest file at /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1/latest, if trying to load latest checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint.
[2022-09-15 06:01:29,734] [WARNING] [engine.py:2572:load_checkpoint] Unable to find latest file at /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1/latest, if trying to load latest checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint.
[2022-09-15 06:01:29,734] [WARNING] [engine.py:2572:load_checkpoint] Unable to find latest file at /home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1/latest, if trying to load latest checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint.
time (ms) | load-checkpoint: 0.92
[after model, optimizer, and learning rate scheduler are built] datetime: 2022-09-15 06:01:29 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      439453125
    validation: 43947520
    test:       2560
> building train, validation, and test datasets for GPT ...
 > building dataset index ...
Dataset does not exist: /data/the_pile_public_merged_nopreprocessing/pile_text_document
Dataset does not exist: /data/the_pile_public_merged_nopreprocessing/pile_text_document
Path should be a basename that both .idx and .bin can be appended to get full filenames.
Dataset does not exist: /data/the_pile_public_merged_nopreprocessing/pile_text_document
Path should be a basename that both .idx and .bin can be appended to get full filenames.
Path should be a basename that both .idx and .bin can be appended to get full filenames.
 > finished creating indexed dataset in 0.000048 seconds
Dataset does not exist: /data/the_pile_public_merged_nopreprocessing/pile_text_document
Path should be a basename that both .idx and .bin can be appended to get full filenames.
Traceback (most recent call last):
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 295, in <module>
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 151, in pretrain
    train_valid_test_dataset_provider)
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 1186, in build_train_valid_test_data_iterators
    train_val_test_num_samples)
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 259, in train_valid_test_datasets_provider
    skip_warmup=(not args.mmap_warmup))
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 41, in build_train_valid_test_datasets
    seq_length, seed, skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 88, in _build_train_valid_test_datasets
    skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 134, in get_indexed_dataset_
    indexed_dataset.sizes.shape[0]))
AttributeError: 'NoneType' object has no attribute 'sizes'
Traceback (most recent call last):
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 295, in <module>
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 151, in pretrain
    train_valid_test_dataset_provider)
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 1186, in build_train_valid_test_data_iterators
    train_val_test_num_samples)
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 259, in train_valid_test_datasets_provider
    skip_warmup=(not args.mmap_warmup))
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 41, in build_train_valid_test_datasets
    seq_length, seed, skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 88, in _build_train_valid_test_datasets
    skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 134, in get_indexed_dataset_
Traceback (most recent call last):
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 295, in <module>
    indexed_dataset.sizes.shape[0]))
AttributeError: 'NoneType' object has no attribute 'sizes'
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 151, in pretrain
Traceback (most recent call last):
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 295, in <module>
    train_valid_test_dataset_provider)
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 1186, in build_train_valid_test_data_iterators
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 151, in pretrain
    train_valid_test_dataset_provider)
  File "/home/cc/Megatron-DeepSpeed/megatron/training.py", line 1186, in build_train_valid_test_data_iterators
    train_val_test_num_samples)
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 259, in train_valid_test_datasets_provider
    skip_warmup=(not args.mmap_warmup))
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 41, in build_train_valid_test_datasets
    train_val_test_num_samples)
  File "/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py", line 259, in train_valid_test_datasets_provider
    seq_length, seed, skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 88, in _build_train_valid_test_datasets
    skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 134, in get_indexed_dataset_
    skip_warmup=(not args.mmap_warmup))
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 41, in build_train_valid_test_datasets
    seq_length, seed, skip_warmup)
    indexed_dataset.sizes.shape[0]))
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 88, in _build_train_valid_test_datasets
AttributeError: 'NoneType' object has no attribute 'sizes'
    skip_warmup)
  File "/home/cc/Megatron-DeepSpeed/megatron/data/gpt_dataset.py", line 134, in get_indexed_dataset_
    indexed_dataset.sizes.shape[0]))
AttributeError: 'NoneType' object has no attribute 'sizes'
[2022-09-15 06:01:31,103] [INFO] [launch.py:286:sigkill_handler] Killing subprocess 56268
[2022-09-15 06:01:31,124] [INFO] [launch.py:286:sigkill_handler] Killing subprocess 56269
[2022-09-15 06:01:31,135] [INFO] [launch.py:286:sigkill_handler] Killing subprocess 56270
[2022-09-15 06:01:31,147] [INFO] [launch.py:286:sigkill_handler] Killing subprocess 56271
[2022-09-15 06:01:31,147] [ERROR] [launch.py:292:sigkill_handler] ['/usr/bin/python3', '-u', '/home/cc/Megatron-DeepSpeed/examples/MoE/../../pretrain_gpt.py', '--local_rank=3', '--override-lr-scheduler', '--adam-beta1', '0.9', '--adam-beta2', '0.95', '--tensor-model-parallel-size', '1', '--moe-expert-parallel-size', '1', '--num-experts', '1', '--moe-loss-coeff', '0.01', '--moe-train-capacity-factor', '1.0', '--moe-eval-capacity-factor', '1.0', '--moe-min-capacity', '4', '--init-method-std', '0.014', '--lr-decay-tokens', '260000000000', '--lr-warmup-tokens', '375000000', '--micro-batch-size', '4', '--exit-duration-in-mins', '30000000', '--global-batch-size', '256', '--num-layers', '24', '--hidden-size', '1024', '--num-attention-heads', '16', '--seq-length', '2048', '--max-position-embeddings', '2048', '--train-tokens', '300000000000', '--train-samples', '439453125', '--lr', '3.0e-4', '--min-lr', '3.0e-5', '--lr-decay-style', 'cosine', '--split', '98,2,0', '--log-interval', '10', '--eval-interval', '100', '--eval-iters', '10', '--save-interval', '1000', '--weight-decay', '0.1', '--clip-grad', '1.0', '--hysteresis', '2', '--num-workers', '0', '--fp16', '--load', '/home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1', '--save', '/home/cc/Megatron-DeepSpeed/examples/MoE/output/checkpoint/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1', '--tensorboard-queue-size', '1', '--log-timers-to-tensorboard', '--log-batch-size-to-tensorboard', '--log-validation-ppl-to-tensorboard', '--tensorboard-dir', '/home/cc/Megatron-DeepSpeed/examples/MoE/output/tensorboard/gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1_ysy-p100-nv_2022.09.15-05.57.40', '--checkpoint-activations', '--vocab-file', './data/the_pile_public_merged_nopreprocessing/gpt2-vocab.json', '--merge-file', './data/the_pile_public_merged_nopreprocessing/gpt2-merges.txt', '--data-path', '/data/the_pile_public_merged_nopreprocessing/pile_text_document', '--data-impl', 'mmap', '--deepspeed', '--deepspeed_config', 'ds_config_gpt_gpt-0.35B-lr-3.0e-4-minlr-3.0e-5-bs-256-gpus-64-mp-1-pp-1.json', '--pipeline-model-parallel-size', '1', '--deepspeed-activation-checkpointing'] exits with return code = 1
