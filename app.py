from flask import Flask, request, jsonify
from pathlib import Path
from loguru import logger
from mllm.dialoggen_demo import DialogGen
from hydit.config import get_args
from hydit.inference import End2End

app = Flask(__name__)

# 预加载模型
args = get_args()

# Set default values for arguments
args.infer_steps = 10
args.enhance = False
args.sampler = 'dpmms'

models_root_path = Path(args.model_root)
if not models_root_path.exists():
    raise ValueError(f"`models_root` not exists: {models_root_path}")

# Load models
gen = End2End(args, models_root_path)
enhancer = None

logger.info("Models preloaded and ready for inference.")

def inferencer():
    # 返回预加载的模型
    return args, gen, enhancer

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Update args with the prompt
    args, gen, enhancer = inferencer()
    args.prompt = prompt

    if enhancer:
        logger.info("Prompt Enhancement...")
        success, enhanced_prompt = enhancer(args.prompt)
        if not success:
            return jsonify({"error": "Prompt is not compliant"}), 400
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
    else:
        enhanced_prompt = None

    # Run inference
    logger.info("Generating images...")
    height, width = args.image_size
    results = gen.predict(args.prompt,
                          height=height,
                          width=width,
                          seed=args.seed,
                          enhanced_prompt=enhanced_prompt,
                          negative_prompt=args.negative,
                          infer_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          batch_size=args.batch_size,
                          src_size_cond=args.size_cond,
                          use_style_cond=args.use_style_cond,
                          sampler=args.sampler
                          )
    images = results['images']

    # Save images
    save_dir = Path('results')
    save_dir.mkdir(exist_ok=True)
    # Find the first available index
    all_files = list(save_dir.glob('*.png'))
    if all_files:
        start = max([int(f.stem) for f in all_files]) + 1
    else:
        start = 0

    saved_paths = []
    for idx, pil_img in enumerate(images):
        save_path = save_dir / f"{idx + start}.png"
        pil_img.save(save_path)
        saved_paths.append(str(save_path))
        logger.info(f"Saved to {save_path}")

    return jsonify({"images": saved_paths})

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=5004)