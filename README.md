# Hunyuan3D Telegram Bot (Kaggle GPU)

Telegram interface for Hunyuan3D shape-only generation. The bot runs inside a manually started Kaggle notebook session; no Google Drive, hosted backend, or always-on server is required.

## Supported inputs

- **Image → 3D:** upload a reference image.
- **Text → 3D:** send a prompt. The official HunyuanDiT text-to-image stage creates a conditioning image, then Hunyuan3D-DiT generates the mesh.

Both paths use the same shape-only pipeline and return a GLB file. Hunyuan3D-Paint is intentionally not loaded.

## Architecture

```
Telegram user
    │ image or text
    ▼
Telegram polling bot (inside Kaggle)
    │ sequential job queue (one GPU generation at a time)
    ▼
Text input ──► HunyuanDiT text-to-image ──┐
                                         ├─► Hunyuan3D-DiT shape generation
Image input ──────────────────────────────┘
    │
    ▼
Trimesh GLB export + validation
    │
    ▼
Telegram document reply
```

## Kaggle setup

1. Start a Kaggle notebook with **GPU T4 x2** enabled.
2. Clone/install the official [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) repository using its current official installation instructions. The shape-only path does not need the Paint custom rasterizer.
3. Clone this repository in the notebook:
   ```bash
   !git clone https://github.com/lk0024709-ux/Telegram-bot-with-py.git
   %cd Telegram-bot-with-py
   !pip install -r requirements.txt
   ```
4. Add the Telegram bot token as a Kaggle Secret named `TELEGRAM_BOT_TOKEN` (recommended), or set it in the notebook environment. Never commit the token.
5. Start the bot:
   ```python
   from kaggle_runtime import run_telegram_bot
   run_telegram_bot()
   ```
6. Keep the notebook session running while using the bot. Stop the notebook to stop the bot.

## Telegram commands

- `/start` — instructions
- `/help` — help
- Send a photo — image-to-3D
- Send a text prompt — text-to-3D

## Runtime notes

- Jobs are processed sequentially to avoid simultaneous GPU inference and VRAM contention.
- The bot and model run in the same Kaggle session; there is no public webhook or separate cloud server.
- Kaggle working storage is temporary. Generated GLBs are sent to Telegram immediately and local job files are deleted afterward.
- T4 x2 does not automatically mean a single model is split across both GPUs. This implementation targets one CUDA device by default; multi-GPU model sharding is not assumed.
- Text-to-3D requires the official HunyuanDiT text-to-image module and its model weights. The runtime reports a clear startup error if this optional module is unavailable.
- The bot does not use Google Drive.

## Configuration

Environment variables:

| Name | Default | Purpose |
|---|---:|---|
| `TELEGRAM_BOT_TOKEN` | required | Telegram bot token |
| `HUNYUAN_SHAPE_MODEL` | `tencent/Hunyuan3D-2` | Shape model repository |
| `HUNYUAN_SHAPE_SUBFOLDER` | `hunyuan3d-dit-v2-0` | Shape model subfolder |
| `HUNYUAN_T2I_MODEL` | `Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled` | Text-to-image model |
| `CUDA_DEVICE` | `cuda` | Device |
| `SHAPE_STEPS` | `30` | Shape inference steps |
| `SHAPE_GUIDANCE_SCALE` | `5.0` | Shape guidance scale |
| `OCTREE_RESOLUTION` | `256` | Mesh extraction resolution |
| `MAX_PROMPT_LENGTH` | `1000` | Prompt limit |

## License

This integration code is provided under MIT. Hunyuan3D code and model weights are subject to Tencent's own license terms; review those terms before use.
