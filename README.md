# edit-batch

Batch image processing with [FLUX.2 [klein 9B] Q4_K_M GGUF](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF), [HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image), [Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit), or [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).

Run the same prompt across many images with dynamic prompting, static and dynamic reference images, support for multiple generations and step count modification along with multiple text encoders and loras. Optionally use HiDream-O1-Image, Boogu-Image-0.1-Edit, or Qwen-Image-2.1 as the backend model.

Repos, GGUFs, and other dependencies are cached under `~/.cache/flux-batch/`.

- Output directory (`-o`) is created automatically if it doesn't exist.
- The prompt file is reread at each image, so you can modify it mid-batch.
- The ref file (`-rf`) is also reread on each iteration — changes are detected by content comparison, and images are only reloaded when the file actually changes.

```
edit-batch -i "*.jpg" -o new/ -p prompt.txt
```


- The prompt file can contain multiple lines. Each line generates a separate image per input, cycling through prompts sequentially. Lines starting with `#` are skipped as comments.
- When multiple prompt lines are present, `--count` is automatically overridden to match the number of lines.

**Inline reference images (`ref:<path>`):**

Prompt lines can embed reference image paths directly using `ref:<path>` syntax. These are extracted, loaded as reference images, and removed from the prompt text before it reaches the model. This composes with `-r` and `-rf` — all sources contribute to the reference pool.

```
# prompt.txt
Make the character wear the outfit ref:/tmp/hat.png ref:/tmp/jacket.png
Put the ball on the table ref:/tmp/ball.png
```

```bash
# No -r needed — refs are embedded in the prompt
edit-batch -i "*.jpg" -o out/ -p prompt.txt
```

**Inline aspect ratio (`ratio:<W>:<H>`):**

Prompt lines can embed an aspect ratio override using `ratio:<W>:<H>` syntax. This overrides `--ratio` from the command line for that prompt line.

```
# prompt.txt
Wide cinematic shot ratio:16:9
Square format ratio:1:1
```

**Inline steps (`steps:<int>`):**

Prompt lines can embed a step count override using `steps:<int>` syntax. This overrides `--steps` from the command line for that prompt line.

```
# prompt.txt
Quick draft steps:4
High quality render steps:20
```

Here's an example. This is 960 frames. 

Look at how frame-matched it is with the original. I even have a nice little peek-a-boo window to compare


https://github.com/user-attachments/assets/08e5e7ea-d49c-4cbd-9391-237583cf7396

A version posted on YouTube: https://youtu.be/8TdXRJQygyY?si=RCVXLUbkEInJ_Z0r

I did this for over 5,000 frames in the video, automatically with edit-batch. Here's a few up close!
<table>
  <tr>
    
<td><img width="1440" height="1072" alt="03884" src="https://github.com/user-attachments/assets/eb75541e-6e08-44e6-9d0a-ed8b52146e46" /></td>
<td><img width="1440" height="1072" alt="03534" src="https://github.com/user-attachments/assets/5f1fc3f1-e6f8-41cc-a959-4f4714ddeedc" /></td></tr><tr>
<td><img width="1440" height="1072" alt="03431" src="https://github.com/user-attachments/assets/d243e9b4-359d-4587-9b9c-6346392cb351" /></td>
<td><img width="1440" height="1072" alt="03575" src="https://github.com/user-attachments/assets/54129fcd-5101-4c45-850e-aca1aed59d14" /></td>
</tr>
</table>



### Text Encoders

By default, edit-batch uses the censored FLUX text encoder. Alternative text encoders can be enabled:

- `--exact`: Loads [dx8152/Flux2-Klein-9B-Consistency](https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency) LoRA for improved consistency
- `--nsfw`: Uses [ponpoke/flux2-klein-9b-uncensored-text-encoder](https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder) text encoder for unrestricted content

```bash
# Use exact encoder (LoRA)
edit-batch --exact -i "*.jpg" -o out/ -p prompt.txt

# Use uncensored encoder
edit-batch --nsfw -i "*.jpg" -o out/ -p prompt.txt
```

### HiDream-O1-Image Mode

Use `--model hidream` to run [HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image), a natively unified 8B image generation model. This is text-to-image only (no img2img conditioning) — input image globs provide size reference but aren't used for content.

```bash
# Basic text-to-image
edit-batch --model hidream -o out/ -p prompt.txt --width 1024 --height 1024

# Use dev model (28-step distilled variant)
edit-batch --model hidream --model-type dev -o out/ -p prompt.txt

```

You'll need the HiDream repo dependencies installed (`pip install -r /path/to/HiDream-O1-Image/requirements.txt`) and `transformers>=4.57.1`.

**HiDream-only flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `flux` | Model to use: `flux`, `hidream`, `boogu`, or `qwen-image-2.1` |
| `--model-type` | `full` | `full` (25 steps, guidance 5.0) or `dev` (28 steps, guidance 0.0) |
| `--guidance-scale` | * | Guidance scale (5.0 full, 0.0 dev) |
| `--seed` | `42` | Random seed for reproducibility |

### Boogu-Image-0.1-Edit-Turbo GGUF Mode

Use `--model boogu` to run [Boogu-Image-0.1-Edit-Turbo](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-Turbo), a 10B unified image editing model with DMD few-step inference.

When no `--boogu-path` is given, the script automatically:
1. Clones the [Boogu-Image](https://github.com/BOOGU-Project/Boogu-Image) repo on first run
2. Downloads the GGUF-quantized transformer (Q4_1, ~6.9 GiB) from HuggingFace
3. Loads the Qwen3VL MLLM in 4-bit via bitsandbytes to save VRAM

The pipeline runs in **DMD turbo mode** (4 steps, no CFG by default). When the default guidance scales (`--text-guidance-scale 5.0 --image-guidance-scale 1.0`) are unchanged, the script auto-adjusts both to 1.0 for DMD.

```bash
# Basic image editing
edit-batch --model boogu -i "*.jpg" -o out/ -p prompt.txt

# Text-to-image only (no input image)
edit-batch --model boogu -o out/ -p prompt.txt --width 1024 --height 1024
```

**Boogu flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--boogu-path` | — | Path to local full-precision Boogu-Image-0.1-Edit directory (skips GGUF auto-setup) |
| `--text-guidance-scale` | `5.0` | Text guidance scale; auto-set to 1.0 for DMD turbo if defaults unchanged |
| `--image-guidance-scale` | `1.0` | Image guidance scale |

**Requirements:** `bitsandbytes` (installed automatically) for 4-bit MLLM quantization. ~12 GiB VRAM needed for 1024² generation.

### Qwen-Image-2.1 Mode

Use `--model qwen-image-2.1` to run [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), a unified 7B text-to-image generation and image editing model with native RGBA transparency and up to 10 reference images for multi-subject composition.

Requires a diffusers release with `QwenImage21Pipeline` (upstream PR [#14804](https://github.com/huggingface/diffusers/pull/14804), not yet in pip releases) and `transformers>=5.17`:

```bash
pip install -U "git+https://github.com/huggingface/diffusers" "transformers>=5.17" accelerate
```

```bash
# Image editing (input image is the edit target)
edit-batch --model qwen-image-2.1 -i "*.jpg" -o out/ -p prompt.txt

# Text-to-image
edit-batch --model qwen-image-2.1 -o out/ -p prompt.txt --width 1024 --height 1024

# Multi-reference composition: input + -r/-rf refs + inline ref: all become condition images
edit-batch --model qwen-image-2.1 -i "photo.jpg" -r "refs/*.jpg" -o out/ -p prompt.txt
```

**Qwen notes:**
- Defaults to **40 inference steps** (`-s` overrides); the model is meant to be sampled without CFG guidance (`true_cfg_scale=1.0`).
- The input image and every reference (from `-r`/`-rf`, inline `ref:`, or `--cumulative` outputs) are passed together as condition images to the pipeline's `image` argument (capped at 10).
- Use the transparency prompt format for RGBA output: `This is an RGBA image with transparency. <desc>. The image has alpha channel and the background is transparent.`
- ~16 GiB VRAM needed at 2048² native resolution; `enable_model_cpu_offload()` is used by default like the other backends.

### Skeleton ControlNet Mode

Use `--skeleton` to guide generation with detected human poses or custom skeleton images:

```bash
# Auto-detect pose from input images and apply skeleton control
edit-batch --skeleton -in "*.jpg" -out pose_out/ -p prompt.txt

# Pure skeleton lines (no Canny edges)
edit-batch --skeleton --skeleton-mode skeleton -in "*.jpg" -out pose_out/ -p prompt.txt

# Use pre-generated skeleton image (reused for all inputs)
edit-batch --skeleton --skeleton-image pose.png -in "*.jpg" -out out/ -p prompt.txt

# Use globbed skeleton images paired with input images
edit-batch --skeleton --skeleton-image "poses/*.png" -in "*.jpg" -out out/ -p prompt.txt

# Adjust control strength (0.0-1.0)
edit-batch --skeleton --skeleton-strength 0.8 -in "*.jpg" -out out/ -p prompt.txt
```

**Skeleton modes:**
- `auto` (default): Renders skeleton lines, then applies Canny edge detection
- `skeleton`: Pose skeleton lines only
- `edges`: Skeleton lines + Canny edges on skeleton
- `canny`: Canny edge detection on original input image

### Required Flags

| Flag | Description |
|------|-------------|
| `-p` / `--prompt` | Prompt file |
| `-o` / `--out` | Output directory |

### Optional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--in` | — | Glob pattern for input images (e.g. `"*.jpg"`) |
| `-c` / `--count` | `1` | Number of images to make |
| `--cumulative` | false | Chain generations: each output becomes a reference for the next (boogu: added as `input_images`; other models: tracked in memory) |
| `-d` / `--device` | `cuda` | Torch device |
| `-s` / `--steps` | `4` | Inference steps |
| `-r` / `--ref` | — | Reference image(s) for style/content. Supports glob patterns; matched files cycle in cadence with input images |
| `--skeleton` | — | Enable skeleton ControlNet mode for pose-guided generation |
| `--skeleton-mode` | `auto` | Skeleton extraction mode: `auto`, `skeleton`, `edges`, `canny` |
| `--skeleton-strength` | `0.6` | ControlNet conditioning scale (0.0-1.0) |
| `--skeleton-image` | — | Pre-generated skeleton/control image (supports glob patterns) |
| `--controlnet` | `InstantX/FLUX.1-dev-Controlnet-Canny` | ControlNet model |
| `--exact` | false | Use [exact](https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency) text encoder (LoRA) |
| `--nsfw` | false | Use [uncensored](https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder) text encoder + anatomy fixer LoRA |
| `--nsfw-lora` | — | Custom LoRA URL/path for --nsfw (default: Klein anatomy fixer from CivitAI) |
| `--lora-strength` | `2.5` | LoRA strength (recommended range: 1.0-3.0) |
| `--scale` | `1.0` | Output size multiplier relative to input (e.g. 0.5 for half, 2.0 for double) |
| `--max-width` | — | Maximum width constraint; overrides `--scale`, maintains aspect ratio |
| `--max-height` | — | Maximum height constraint; overrides `--scale`, maintains aspect ratio |
| `--ratio` | — | Target aspect ratio (e.g. `16:9`, `4:3`); applied after scale/max constraints, then reclamped to max bounds |
| `--offset` | `0` | Start reading prompt file from this line (default: 0) |
| `-rf` / `--ref-file` | — | File listing reference images (one per line); re-read each iteration like `--prompt`, reloads images only on content change |
| `--shuf` | false | Shuffle input file order randomly |
| `-nc` | false | No Clobber — skip existing outputs |
| `--model` | `flux` | Model backend: `flux`, `hidream`, `boogu`, or `qwen-image-2.1` |

| `--model-type` | `full` | HiDream variant: `full` (25 steps) or `dev` (28 steps) |
| `--guidance-scale` | * | HiDream guidance scale (5.0 full, 0.0 dev) |
| `--text-guidance-scale` | `5.0` | Text guidance scale for Boogu |
| `--image-guidance-scale` | `1.0` | Image guidance scale for Boogu |
| `--seed` | `42` | Random seed for generation |

