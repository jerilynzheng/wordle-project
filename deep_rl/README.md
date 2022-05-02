## Advantage Actor Critic Model

### Shell Environment

Use W&B offline model.

```shell
export WANDB_MODE=offline
```

### Train

Train initial model with 100-word list.

```shell
python a2c_train.py --network_name=SumChars \
  --env=WordleEnv100-v0 --epoch_len=100 --batch_size=64 \
  --max_epochs=100 --lr=0.0001 \
  --save_model=model_WordleEnv100-v0.pth
```

### Evaluate

Evaluate with 100-word list.

```shell
python a2c_play.py \
  --env=WordleEnv100-v0 lightning_logs/version_32/checkpoints/epoch=3-step=25599.ckpt \
  evaluate
```

Evaluate with 1000-word list.

```shell
python a2c_play.py \
  --env=WordleEnv1000-v0 lightning_logs/version_32/checkpoints/epoch=3-step=25599.ckpt \
  evaluate
```

### Export model

If `--export_model` option wasn't given when calling a2c_train.py, use the following to export model from checkpoint.

```shell
python a2c_play.py --model_path=model_WordleEnv100-v0 \
  lightning_logs/version_32/checkpoints/epoch=3-step=25599.ckpt \
  export
```

### Train with pre-trained model

Train new model with 1000-word list, using the pre-trained model with 100-word list.

```shell
python a2c_train.py --network_name=SumChars \
  --env=WordleEnv1000-v0 --epoch_len=100 --batch_size=64 \
  --max_epochs=20 --lr=0.0001 \
  --load_model=model_WordleEnv100-v0 \
  --save_model=model_WordleEnv1000-v0
```

