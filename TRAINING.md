# Training Your Own MistralMail Model

## Data Preparation

1. Export your emails to JSONL format:
```python
# Each line should be a JSON object with a "text" field
{"text": "Email content here..."}
```

2. Place in `data/emails_training.jsonl`

## Training Script

Run the training with:
```bash
python train.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --data_path data/emails_training.jsonl \
  --output_dir models/your-model \
  --num_epochs 1 \
  --batch_size 4
```

## Hardware Requirements

- Minimum: RTX 3090 (24GB VRAM)
- Recommended: RTX 4090/5090 (32GB VRAM)
- Training time: ~3-6 hours for 100k emails

## Hyperparameters

Tested configuration:
- LoRA rank: 32
- LoRA alpha: 64
- Learning rate: 2e-4
- Batch size: 4
- Gradient accumulation: 4
