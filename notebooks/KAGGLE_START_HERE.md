# Kaggle launcher

This project is designed to run in the same Kaggle notebook session as the Hunyuan3D models.

## Cell 1 — Enable accelerator

In Kaggle notebook settings choose **GPU T4 x2** and enable Internet if your model weights still need to download.

## Cell 2 — Install the bot dependency

```python
!pip install -q python-telegram-bot==21.10 Pillow>=10.0 trimesh>=4.0
```

## Cell 3 — Get this integration code

```python
!git clone https://github.com/lk0024709-ux/Telegram-bot-with-py.git
%cd Telegram-bot-with-py
```

If you are using the feature branch before it is merged, clone it with:

```python
!git clone -b feature/hunyuan3d-kaggle-telegram https://github.com/lk0024709-ux/Telegram-bot-with-py.git
%cd Telegram-bot-with-py
```

## Cell 4 — Add the bot token

Recommended: create a Kaggle Secret named `TELEGRAM_BOT_TOKEN`. Do not paste the token into a notebook cell or commit it to GitHub.

## Cell 5 — Start the bot

Run after the official Hunyuan3D-2 package is installed and the model dependencies are available:

```python
from kaggle_runtime import run_telegram_bot
# If your existing Hunyuan3D-DiT pipeline variable is named `pipeline`:
run_telegram_bot(shape_pipeline=pipeline)
# Otherwise call run_telegram_bot() and the runtime loads the shape model.
```

This loads the HunyuanDiT text-to-image pipeline on GPU 1 (when available) and the Hunyuan3D shape pipeline on GPU 0. Text prompts first produce a conditioning image; uploaded photos skip that stage. Both paths then run shape-only Hunyuan3D-DiT and send a GLB to Telegram.

**Important:** The bot is not a hosted service. Keep this Kaggle session running. Stopping or timing out the session stops Telegram polling. No Google Drive is used.
