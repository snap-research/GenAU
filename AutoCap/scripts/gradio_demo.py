import gradio as gr
import torch
from pytorch_lightning import seed_everything
from src.utilities.model.model_utils import setup_seed
from src.tools.configuration import Configuration
from src.tools.download_manager import get_checkpoint_path
from src.utilities.audio.audio_processing import process_wavform

def load_model(
    model_name="autocap-full", 
    config_path=None,
    checkpoint_path=None,
    use_cpu=False,
    seed=0
):
    """
    Loads AutoCap model + config, sets seeds, returns (model, config).
    Modify this according to your actual environment if needed.
    """

    if config_path is None:
        config_path = get_checkpoint_path(f"{model_name}_config")
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)

    configuration = Configuration(config_path)
    config = configuration.get_config()

    # Set seeds
    config["training"]["seed"] = seed
    setup_seed(config["training"]["seed"])
    seed_everything(config["training"]["seed"])

    # Insert checkpoint path into config
    if checkpoint_path is not None:
        config["training"]["pretrain_path"] = checkpoint_path

    # Ensure we have a checkpoint to load
    assert config["training"]["pretrain_path"] is not None, "Please provide a valid checkpoint."

    # Create model on device
    device = "cpu" if use_cpu else "cuda"
    # 'config["target"]' should be the AutoCap class or a function that returns the model
    model = config["target"](config).to(device)
    model.eval()

    return model, config

def transcribe(
    model_state,
    audio_file,
    title,
    video_caption,
    description,
    beam_size,
    use_cpu=False
):
    """
    Args:
      model_state (dict): {'model': AutoCapModel, 'config': config_dict}
      audio_file (tuple): (sample_rate, data) from Gradio if 'type="numpy"'
      title, video_caption, description: Optional metadata
      beam_size (int): for model.generate(...)
      use_cpu (bool): whether to force CPU

    Returns:
      str: The generated transcription
    """
    if audio_file is None:
        return "No audio provided."

    # Extract model + config from the dictionary
    model = model_state["model"]
    config = model_state["config"]

    # Convert audio (sample_rate, numpy array) -> Torch tensor
    sr, data = audio_file
    waveform = torch.from_numpy(data).float().unsqueeze(0)  # shape = [1, samples]
    waveform = process_wavform(waveform, sr, resampling_rate=16000, duration=10)
    print("waveform", waveform.shape)
    waveform = torch.from_numpy(waveform).float()
    device = "cpu" if use_cpu else "cuda"
    waveform = waveform.to(device)

    # Model expects audio=..., text=..., meta=...
    # Prepare meta dictionary if needed
    meta = model.get_meta_dict({
        "title":        [title],
        "video_caption":[video_caption],
        "description":  [description]
    })

    with torch.no_grad():
        # Pass arguments as expected by your model
        caption = model.generate(
            samples=waveform,  
            meta=meta,
            num_beams=beam_size
        )

    return caption[0]


def main():
    # 3.1) Load model + config once
    model, config = load_model(
        model_name="autocap-full",
        config_path=None,       # or path/to/your_config.yaml
        checkpoint_path=None,   # or path/to/your_checkpoint.ckpt
        use_cpu=False,
        seed=0
    )

    model_state = {"model": model, "config": config}

    with gr.Blocks() as demo:
        gr.Markdown("## Audio-to-Text Demo for AutoCap")

        with gr.Row():
            audio_input = gr.Audio(type="numpy", label="Upload/Record Audio")
        
        with gr.Accordion("Optional Metadata", open=False):
            title_box = gr.Textbox(label="Title")
            video_caption_box = gr.Textbox(label="Video Caption")
            description_box = gr.Textbox(label="Description")

        beam_size_slider = gr.Slider(
            minimum=1, maximum=10, step=1, value=2,
            label="Beam Size"
        )

        transcribe_btn = gr.Button("Transcribe")
        output_text = gr.Textbox(label="Generated Caption")

        # 3.4) Link button to inference
        transcribe_btn.click(
            fn=transcribe,
            inputs=[
                gr.State(model_state),
                audio_input,
                title_box,
                video_caption_box,
                description_box,
                beam_size_slider,
                gr.State(False)  # use_cpu=False in this example
            ],
            outputs=output_text
        )

    # 3.5) Launch Gradio
    demo.launch(server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main()
