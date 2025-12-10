# MistralMail - Personal Email AI Assistant

A fine-tuned Mistral 7B model trained on personal email data to generate responses in your unique writing style.

## 🚀 Features

- **Personalized Writing Style**: Trained on 100,000 personal emails
- **Efficient LoRA Fine-tuning**: Only 336MB adapter vs 14GB full model
- **Web Interface**: Simple Flask-based UI for testing
- **Docker Support**: Easy deployment with GPU acceleration
- **Fast Inference**: 4-bit quantization for efficient generation

## 📊 Training Details

- **Base Model**: Mistral-7B-v0.1
- **Training Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Dataset**: 100,000 personal emails
- **Training Time**: ~6 hours on RTX 5090
- **Final Loss**: 0.14
- **LoRA Rank**: 32
- **Learning Rate**: 2e-4

## 🛠️ Installation

### Prerequisites
- NVIDIA GPU with 16GB+ VRAM
- Docker with NVIDIA Container Toolkit
- CUDA 12.1+

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/kylefoxaustin/mistral-mail.git
cd mistral-mail
```

2. Download or place your trained model in `models/mistral-editorial-final/`

3. Build and run with Docker:
```bash
docker build -t mistral-mail .
docker run -it --gpus all -p 8081:8081 mistral-mail
```

4. Open http://localhost:8081 in your browser

## 🏋️ Training Your Own Model

See [TRAINING.md](TRAINING.md) for detailed instructions on fine-tuning Mistral on your own email dataset.

## 📁 Project Structure
```
mistral-mail/
├── app.py                 # Flask web interface
├── train.py              # Training script
├── inference.py          # Standalone inference
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── models/              # Model weights (not in repo)
│   └── mistral-editorial-final/
└── data/                # Training data (not in repo)
```

## ⚠️ Privacy Notice

This project is designed to train on personal email data. Never share your trained model publicly as it may generate text containing personal information from your training data.

## 🔧 Configuration

Edit `config.yaml` to adjust:
- Generation parameters (temperature, top_p, etc.)
- Model paths
- Training hyperparameters

## 📝 License

MIT License - See [LICENSE](LICENSE) file

## 👨‍💻 Maintainer

**Kyle Fox** - Austin, TX
- GitHub: [@kylefoxaustin](https://github.com/kylefoxaustin)
- Project: [MistralMail](https://github.com/kylefoxaustin/mistral-mail)

## 🙏 Acknowledgments

- Built with [Mistral-7B](https://mistral.ai/)
- Uses [PEFT](https://github.com/huggingface/peft) for LoRA
- Inspired by personal productivity needs

---

**Note**: This model is trained on personal data and should not be shared publicly.
