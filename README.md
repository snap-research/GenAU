<div style="text-align: center;">
  <img src="assets/logo.png" alt="Logo" style="width: 50px; vertical-align: middle;">
  <span style="font-size: 28px; vertical-align: middle;">Taming Data and Transformers for Audio Generation</span>
</div>
</br>
This is the official GitHub repository of the paper Taming Data and Transformers for Audio Generation.

**[Taming Data and Transformers for Audio Generation](https://snap-research.github.io/GenAU)**
</br>
[Moayed Haji-Ali](https://tsaishien-chen.github.io/),
[Willi Menapace](https://www.willimenapace.com/),
[Aliaksandr Siarohin](https://aliaksandrsiarohin.github.io/aliaksandr-siarohin-website/),
[Guha Balakrishnan](https://www.guhabalakrishnan.com),
[Sergey Tulyakov](http://www.stulyakov.com/)
[Vicente Ordonez](https://vislang.ai/),
</br>
*Arxiv 2024*

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://snap-research.github.io/GenAU) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/taming-data-and-transformers-for-audio/audio-captioning-on-audiocaps)](https://paperswithcode.com/sota/audio-captioning-on-audiocaps?p=taming-data-and-transformers-for-audio) 


# Introduction

<div align="justify">
<div>
<img src="assets/framework.jpg" width="1000" />
</div>
</br>
Generating ambient sounds and effects is a challenging problem due to data scarcity and often insufficient caption quality, making it difficult to employ large-scale generative models for the task. In this work, we tackle the problem by introducing two new models. First, we propose AutoCap, a high-quality and efficient automatic audio captioning model. We show that by leveraging metadata available with the audio modality, we can substantially improve the quality of captions. AutoCap reaches a CIDEr score of 83.2, marking a 3.2% improvement from the best available captioning model at four times faster inference speed. We then use AutoCap to caption clips from existing datasets, obtaining 761, 000 audio clips with high-quality captions, forming the largest available audio-text dataset. Second, we propose GenAu, a scalable transformer-based audio generation architecture that we scale up to 1.25B parameters and train with our new dataset. When compared to state-of-the-art audio generators, GenAu obtains significant improvements of 15.7% in FAD score, 22.7% in IS, and 13.5% in CLAP score, indicating significantly improved quality of generated audio compared to previous works. This shows that the quality of data is often as important as its quantity. Besides, since AutoCap is fully automatic, new audio samples can be added to the training dataset, unlocking the training of even larger generative models for audio synthesis. For more details, please visit our <a href='https://snap-research.github.io/GenAU'>project webpage</a>.
</div> 
<br>


# Updates
- **2024.06.28**: Paper and code released!

# TODOs
- [ ] Add GenAU Gradio demo
- [ ] Add AutoCap Gradio demo

# Setup
Initialize a [conda](https://docs.conda.io/en/latest) environment named genau by running:
```
conda env create -f environment.yaml
conda activate genau
```
# Dataset Preparation 
See [Dataset Preparation](./dataset_preperation/README.md) for details on downloading and preparing the AutoCap dataset, as well as more information on organizing your custom dataset.

# Audio Captioning (AutoCap)
See [GenAU](./AutoCap/README.md) README for details on inference, training, and evaluating our audio captioner AutoCAP.

# Audio Generation (GenAU)
See [GenAU](./GenAU/README.md) README for details on inference, training, finetuning, and evaluating our audio generator GenAU.


## Citation
If you find this paper useful in your research, please consider citing our work:
```
TODO
```
