import os
import torch
import gradio as gr
import soundfile as sf
from pytorch_lightning import seed_everything

from src.tools.training_utils import get_restore_step
from src.utilities.model.model_util import instantiate_from_config
from src.tools.configuration import Configuration
from src.tools.download_manager import get_checkpoint_path

def load_model(
    model_name: str = "genau-l-full-hq-data",
    config_yaml_path: str = None,
    checkpoint_path: str = None,
):
    assert torch.cuda.is_available(), "CUDA is not available."

    if config_yaml_path is None:
        config_yaml_path = get_checkpoint_path(f"{model_name}_config")
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)
    
    print("checkpoint_path", checkpoint_path)
    configuration = Configuration(config_yaml_path)
    config_dict = configuration.get_config()
    
    if checkpoint_path is not None:
        config_dict["reload_from_ckpt"] = checkpoint_path

    exp_name = os.path.basename(config_yaml_path.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml_path))
    log_path = config_dict['logging']["log_directory"]
    
    if "reload_from_ckpt" in config_dict and config_dict["reload_from_ckpt"]:
        resume_from_checkpoint = config_dict["reload_from_ckpt"]
    else:
        # Otherwise try to load the latest
        ckpt_folder = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")
        if not os.path.exists(ckpt_folder):
            raise RuntimeError(f"No checkpoint directory found at {ckpt_folder}")
        restore_step, _ = get_restore_step(ckpt_folder)
        resume_from_checkpoint = os.path.join(ckpt_folder, restore_step)

    config_dict["model"]["params"]["ckpt_path"] = resume_from_checkpoint

    latent_diffusion = instantiate_from_config(config_dict["model"])
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    return latent_diffusion, config_dict


def infer_gradio(
    prompt: str,
    seed: int,
    cfg_weight: float,
    n_cand: int,
    ddim_steps: int
):
    """
    Inference function called by Gradio's interface.
    Returns a WAV audio object (data, sr) to play in the Gradio UI.
    """
    seed_everything(seed)

    saved_wav_path = latent_diffusion.text_to_audio(
        prompt=prompt,
        ddim_steps=ddim_steps,
        unconditional_guidance_scale=cfg_weight,
        n_gen=n_cand,
        use_ema=True
    )

    data, sr = sf.read(saved_wav_path)
    return (sr, data)


latent_diffusion, config_yaml = load_model(
    model_name="genau-l-full-hq-data",  # or whichever default model you want
    config_yaml_path=None,      # or path to your .yaml if you have it
    checkpoint_path=None        # or a direct path to a .ckpt file
)

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Audio Demo")

    with gr.Row():
        prompt_input = gr.Textbox(
            lines=2, 
            label="Prompt", 
            placeholder="Type your text prompt here...",
            value="A calm piano melody."
        )

    with gr.Accordion("Advanced Parameters", open=False):
        seed_slider = gr.Number(
            value=0,
            label="Random Seed"
        )
        cfg_slider = gr.Slider(
            minimum=0.1, maximum=10.0, value=4.0, step=0.1,
            label="Classifier-Free Guidance Weight"
        )
        n_cand_slider = gr.Slider(
            minimum=1, maximum=4, value=1, step=1,
            label="Number of Candidates"
        )
        ddim_steps_slider = gr.Slider(
            minimum=10, maximum=500, value=100, step=10,
            label="DDIM Steps"
        )

    generate_button = gr.Button("Generate Audio")

    audio_output = gr.Audio(type="numpy", label="Generated Audio")

    generate_button.click(
        fn=infer_gradio,
        inputs=[prompt_input, seed_slider, cfg_slider, n_cand_slider, ddim_steps_slider],
        outputs=[audio_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
