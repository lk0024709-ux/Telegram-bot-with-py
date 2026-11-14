"""Telegram polling runtime for Hunyuan3D inside a Kaggle notebook.

Run after the official Tencent Hunyuan3D-2 repository is installed:
    from kaggle_runtime import run_telegram_bot
    run_telegram_bot()

The runtime intentionally uses a single sequential queue because GPU inference
must not overlap on the same shape-generation pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

LOG = logging.getLogger("hunyuan3d.telegram")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1000"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "3"))
SHAPE_STEPS = int(os.getenv("SHAPE_STEPS", "30"))
SHAPE_GUIDANCE_SCALE = float(os.getenv("SHAPE_GUIDANCE_SCALE", "5.0"))
OCTREE_RESOLUTION = int(os.getenv("OCTREE_RESOLUTION", "256"))
SHAPE_MODEL = os.getenv("HUNYUAN_SHAPE_MODEL", "tencent/Hunyuan3D-2")
SHAPE_SUBFOLDER = os.getenv("HUNYUAN_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-0")
T2I_MODEL = os.getenv(
    "HUNYUAN_T2I_MODEL",
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
)


def _get_bot_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        return token
    # Kaggle Secrets are optional; environment variables also work.
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        token = UserSecretsClient().get_secret("TELEGRAM_BOT_TOKEN").strip()
    except Exception:
        token = ""
    if not token:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN. Add it to Kaggle Secrets "
            "or set it as an environment variable."
        )
    return token


def _load_models(shape_pipeline: Any = None, text_to_image: Any = None, background_remover: Any = None) -> tuple[Any, Any, Any]:
    """Load HunyuanDiT text-to-image and Hunyuan3D shape pipelines.

    With Kaggle T4 x2, text-to-image defaults to cuda:1 and shape generation
    to cuda:0. This is two independent model placements, not model sharding.
    """
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.text2image import HunyuanDiTPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Enable Kaggle GPU acceleration.")

    gpu_count = torch.cuda.device_count()
    shape_device = os.getenv("CUDA_DEVICE", "cuda:0")
    default_t2i_device = "cuda:1" if gpu_count > 1 else shape_device
    t2i_device = os.getenv("T2I_DEVICE", default_t2i_device)

    if text_to_image is None:
        LOG.info("Loading text-to-image model on %s", t2i_device)
        text_to_image = HunyuanDiTPipeline(T2I_MODEL, device=t2i_device)

    if shape_pipeline is None:
        LOG.info(
            "Loading Hunyuan3D shape model %s/%s on %s",
            SHAPE_MODEL,
            SHAPE_SUBFOLDER,
            shape_device,
        )
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            SHAPE_MODEL,
            subfolder=SHAPE_SUBFOLDER,
            use_safetensors=True,
            device=shape_device,
        )

    if background_remover is None:
        background_remover = BackgroundRemover()
    LOG.info("Both generation pipelines are ready.")
    return text_to_image, shape_pipeline, (background_remover, export_to_trimesh)


def _generate_glb(
    *,
    prompt: str | None,
    image_path: str | None,
    text_to_image: Any,
    shape_pipeline: Any,
    helpers: tuple[Any, Any],
) -> tuple[str, str]:
    """Generate one shape-only GLB. Returns (file_path, source_description)."""
    import torch
    background_remover, export_to_trimesh = helpers

    if bool(prompt) == bool(image_path):
        raise ValueError("Provide exactly one of prompt or image_path.")

    if prompt:
        LOG.info("Generating conditioning image from text prompt.")
        conditioning_image = text_to_image(prompt)
        source_description = "Text-to-3D"
    else:
        conditioning_image = Image.open(image_path).convert("RGBA")
        source_description = "Image-to-3D"

    # Match the official Hunyuan3D shape-only path: remove the background
    # before passing the conditioning image to Hunyuan3D-DiT.
    try:
        conditioning_image = background_remover(conditioning_image.convert("RGB"))
    except Exception as exc:
        LOG.warning("Background removal failed; using original image: %s", exc)
        conditioning_image = conditioning_image.convert("RGBA")

    generator = torch.Generator().manual_seed(
        int.from_bytes(os.urandom(4), "big")
    )
    outputs = shape_pipeline(
        image=conditioning_image,
        num_inference_steps=SHAPE_STEPS,
        guidance_scale=SHAPE_GUIDANCE_SCALE,
        generator=generator,
        octree_resolution=OCTREE_RESOLUTION,
        num_chunks=200000,
        output_type="mesh",
    )
    mesh = export_to_trimesh(outputs)[0]

    output_dir = tempfile.mkdtemp(prefix="hunyuan3d_")
    output_path = str(Path(output_dir) / "generated_model.glb")
    mesh.export(output_path, include_normals=False)

    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 100:
        raise RuntimeError("GLB export failed or produced an empty file.")

    # Basic parse check before sending the artifact to Telegram.
    import trimesh
    loaded = trimesh.load(output_path, force="scene")
    if loaded is None or len(loaded.geometry) == 0:
        raise RuntimeError("Exported GLB could not be parsed as a 3D scene.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path, source_description


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Hunyuan3D Generator is ready.\n\n"
        "• Send a photo for Image → 3D\n"
        "• Send a text prompt for Text → 3D\n\n"
        "The output is a shape-only GLB. Generation runs on the active Kaggle GPU."
    )


async def _help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _start(update, context)


async def _enqueue(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    prompt: str | None = None,
    image_path: str | None = None,
) -> None:
    queue: asyncio.Queue = context.application.bot_data["job_queue"]
    if queue.full():
        if image_path:
            Path(image_path).unlink(missing_ok=True)
        await update.effective_message.reply_text(
            "The generation queue is full. Please try again in a few minutes."
        )
        return

    await queue.put(
        {
            "chat_id": update.effective_chat.id,
            "prompt": prompt,
            "image_path": image_path,
        }
    )
    position = queue.qsize()
    await update.effective_message.reply_text(
        f"Request queued (position {position}). I will send your GLB here when it is ready."
    )


async def _photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.photo:
        return
    photo = message.photo[-1]
    telegram_file = await context.bot.get_file(photo.file_id)
    fd, local_path = tempfile.mkstemp(prefix="hunyuan_input_", suffix=".jpg")
    os.close(fd)
    try:
        await telegram_file.download_to_drive(custom_path=local_path)
        await _enqueue(update, context, image_path=local_path)
    except Exception:
        Path(local_path).unlink(missing_ok=True)
        LOG.exception("Could not download Telegram image.")
        await message.reply_text("I couldn't download that image. Please send it again.")


async def _text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return
    prompt = message.text.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        await message.reply_text(
            f"Prompt is too long. Maximum length: {MAX_PROMPT_LENGTH} characters."
        )
        return
    if not prompt:
        await message.reply_text("Please send a description of the 3D model.")
        return
    await _enqueue(update, context, prompt=prompt)


async def _queue_worker(application: Application) -> None:
    queue: asyncio.Queue = application.bot_data["job_queue"]
    text_to_image = application.bot_data["text_to_image"]
    shape_pipeline = application.bot_data["shape_pipeline"]
    helpers = application.bot_data["helpers"]

    while True:
        job = await queue.get()
        output_path: str | None = None
        try:
            await application.bot.send_message(
                chat_id=job["chat_id"],
                text="Generation started. This can take several minutes on Kaggle GPU.",
            )
            output_path, mode = await asyncio.to_thread(
                _generate_glb,
                prompt=job["prompt"],
                image_path=job["image_path"],
                text_to_image=text_to_image,
                shape_pipeline=shape_pipeline,
                helpers=helpers,
            )
            with open(output_path, "rb") as model_file:
                await application.bot.send_document(
                    chat_id=job["chat_id"],
                    document=model_file,
                    filename="hunyuan3d_model.glb",
                    caption=f"{mode} complete — shape-only GLB",
                    read_timeout=180,
                    write_timeout=180,
                )
        except Exception as exc:
            LOG.exception("Generation job failed.")
            await application.bot.send_message(
                chat_id=job["chat_id"],
                text=f"Generation failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if output_path:
                output_file = Path(output_path)
                output_dir = output_file.parent
                output_file.unlink(missing_ok=True)
                try:
                    output_dir.rmdir()
                except OSError:
                    LOG.warning("Could not remove temporary output directory: %s", output_dir)
            if job["image_path"]:
                Path(job["image_path"]).unlink(missing_ok=True)
            queue.task_done()


async def _post_init(application: Application) -> None:
    application.bot_data["job_queue"] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    application.bot_data["worker_task"] = application.create_task(
        _queue_worker(application),
        name="hunyuan3d-generation-queue",
    )


async def _post_shutdown(application: Application) -> None:
    task = application.bot_data.get("worker_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def run_telegram_bot(*, shape_pipeline: Any = None, text_to_image: Any = None, background_remover: Any = None) -> None:
    """Start polling; pass already-loaded notebook models to avoid loading them twice."""
    token = _get_bot_token()
    text_to_image, shape_pipeline, helpers = _load_models(
        shape_pipeline=shape_pipeline,
        text_to_image=text_to_image,
        background_remover=background_remover,
    )
    application = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    application.bot_data["text_to_image"] = text_to_image
    application.bot_data["shape_pipeline"] = shape_pipeline
    application.bot_data["helpers"] = helpers
    application.add_handler(CommandHandler("start", _start))
    application.add_handler(CommandHandler("help", _help))
    application.add_handler(MessageHandler(filters.PHOTO, _photo))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _text)
    )
    LOG.info("Starting Telegram polling. Keep this Kaggle session active.")
    application.run_polling(drop_pending_updates=True)
