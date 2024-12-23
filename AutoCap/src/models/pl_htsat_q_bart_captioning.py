# Author: Moayed Haji Ali
# Email: mh155@rice.edu

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl

from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from laion_clap import CLAP_Module
import logging
from loguru import logger

from src.modules.audio_encoder.audio_encoder_config import AudioEncoderConfig
from src.modules.audio_encoder.audio_encoder import AudioEncoderModel
from src.modules.qformer.Qformer import BertConfig, BertLMHeadModel
from src.tools.optim_utils import cosine_lr, get_optimizer, cosine_lr_scheduler
from src.utilities.model.model_utils import set_logger, decode_output
from src.tools.evaluation import evaluate_metrics
from src.tools.io import load_file, load_json, write_json
from src.tools.download_manager import get_checkpoint_path

import time
import shutil
import numpy as np
# disable tokenizer parallelism
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SpecialTokensEmbeddings(nn.Module):
    def __init__(self, model, base_vocab_size, base_tokenizer, special_tokens):
        super().__init__()
        self.hidden_size = model.config.hidden_size
        self.special_tokens = special_tokens
        self.base_vocab_size = base_vocab_size
        self.base_tokenizer = base_tokenizer
        self.model = model

        self.special_token_embed = nn.Embedding(len(special_tokens), self.hidden_size)

        # initialize with bos and eos tokens 
        start_special_tokens_idx = [i for i in range(len(special_tokens)) if not "/" in special_tokens[i]]
        end_special_tokens_idx = [i for i in range(len(special_tokens)) if "/" in special_tokens[i]]
        with torch.no_grad():
            self.special_token_embed.weight[start_special_tokens_idx] = torch.nn.Parameter((model.encoder.embed_tokens.weight[base_tokenizer.bos_token_id]).clone().detach())
            self.special_token_embed.weight[end_special_tokens_idx] = torch.nn.Parameter((model.encoder.embed_tokens.weight[base_tokenizer.eos_token_id]).clone().detach())

    def forward(self, token_ids):
        # Create a mask for special tokens
        special_tokens_mask = token_ids >= self.base_vocab_size
        special_tokens_ids = token_ids[special_tokens_mask] - self.base_vocab_size
        special_tokens_embeds = self.special_token_embed(special_tokens_ids)

        token_ids = torch.where(special_tokens_mask, self.base_tokenizer.pad_token_id, token_ids)
        embeddings = self.model.encoder.embed_tokens(token_ids)

        embeddings.view(-1, embeddings.shape[-1])[special_tokens_mask.view(-1)] = special_tokens_embeds
        return embeddings


class AutoCap(pl.LightningModule):
    @classmethod
    def init_text_Qformer(cls, config, num_query_token, text_width,
                           input_vid2tex_query_embed=True,
                           vocab_size=50265,
                           max_position_embeddings=1024):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.vocab_size = vocab_size
        encoder_config.max_position_embeddings = max_position_embeddings
        encoder_config.num_hidden_layers = config['text_qformer']['num_hidden_layers']
        encoder_config.num_attention_heads = config['text_qformer']['num_attention_heads']
        encoder_config.encoder_width = text_width
        encoder_config.hidden_size = config['text_qformer']['hidden_size']
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = config['text_qformer']['add_cross_attention']
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)

        if input_vid2tex_query_embed:
            return Qformer
        else:
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            return Qformer, query_tokens


    @classmethod
    def init_audio_Qformer(cls, config, num_query_token,
                           vision_width,
                           vocab_size=50265,
                           max_position_embeddings=1024):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.vocab_size = vocab_size
        encoder_config.max_position_embeddings = max_position_embeddings
        encoder_config.num_hidden_layers = config['audio_qformer']['num_hidden_layers']
        encoder_config.encoder_width = vision_width
        encoder_config.num_attention_heads = config['audio_qformer']['num_attention_heads']
        encoder_config.hidden_size = config['audio_qformer']['hidden_size']
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = config['audio_qformer']['add_cross_attention']
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def set_audio_encoder_requires_grad(self, freeze=False):
        for name, param in self.encoder.named_parameters():
            if "fc1" not in name:
                param.requires_grad = not freeze
    
    def set_qformer_requires_grad(self, freeze_audio=False, freeze_text=False):
        if self.use_audio_qformer:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = not freeze_audio

        if self.use_text_qformer:
            for name, param in self.text_Qformer.named_parameters():
                param.requires_grad = not freeze_text
    
    def set_decoder_requires_grad(self, freeze=False, freeze_embed_layer=False):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = not freeze
        
        # embedding layer
        self.decoder.model.shared.weight.requires_grad = not freeze_embed_layer
        
        # Note: Adding grad hook messes up with the optimizer first and second momentum due to the enforced zero gradients leading to divergance, instead we added an embedding layer for the special tokens
        # if freeze and self.add_grad_hook and freeze_embed_layer: 
        #     # Unfreeze embeddings for newly added tokens only
        #     print("[INFO] adding grad hook for tokens")
        #     if self.extended_seq_length > 0:
        #         self.decoder.model.encoder.embed_positions.weight.requires_grad = True
        #         def freeze_prev_pos_embeds(grad):
        #             if not freeze:
        #                 return grad
                    
        #             mask = torch.zeros_like(grad)
        #             mask[-self.extended_seq_length:] = 1
        #             return grad * mask
        #         self.decoder.model.encoder.embed_positions.weight.register_hook(freeze_prev_pos_embeds)
            
        #     if self.num_added_special_tokens > 0:
        #         self.decoder.model.encoder.embed_tokens.weight.requires_grad = True
        #         def freeze_prev_tokens(grad):
        #             if not freeze:
        #                 return grad
                    
        #             mask = torch.zeros_like(grad)
        #             mask[-self.num_added_special_tokens:] = 1
        #             return grad * mask
        #         self.decoder.model.encoder.embed_tokens.weight.register_hook(freeze_prev_tokens)
            
    
    def __init__(self, config):
        super(AutoCap, self).__init__()

        self.config = config
        self.num_added_special_tokens = 0
        self.extended_seq_length = 0
        self.exclude_metrics = config['training'].get('exclude_metrics', [])
        self.meta_keys = config['model']['meta_keys']
        self.resize_token_embeds = config['model'].get('resize_token_embeds', False)

        # encoder
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],
                                            audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)

        # bart decoder
        decoder_name = config["text_decoder_args"]["name"]
        decoder_pretrained = config["text_decoder_args"]["pretrained"]
        freeze_decoder = config["text_decoder_args"]["freeze"]
        freeze_embed_layer = config["text_decoder_args"]["freeze_embed_layer"]

        if decoder_pretrained:
            self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        else:
            bart_config = BartConfig.from_pretrained(decoder_name)
            self.decoder = BartForConditionalGeneration(config=bart_config)

        self.set_decoder_requires_grad(freeze=freeze_decoder, freeze_embed_layer=freeze_embed_layer)

        self.enc_to_dec_proj = nn.Linear(encoder_config.hidden_size, self.decoder.config.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        
        self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
        self.text_vocab_size = len(self.tokenizer)
        
        self.tokenizer_max_length = config['tokenizer']['max_length']
        self.max_prompt_len = config['meta']['max_prompt_len']
        
        self.use_audio_qformer = config['model']['use_audio_qformer']
        self.use_text_qformer = config['model']['use_text_qformer']
        self.use_clap_embeds = config['model'].get('use_clap_embeds', False)
        self.use_meta = config['model']['meta_input']
        self.add_special_tokens = config['model'].get('add_special_tokens', True)
        self.resample_rate = config['data_args']['preprocessing']['audio']['sampling_rate']
        self.eval_beam_sizes = config['model'].get('eval_beam_sizes', [1, 2, 3])
        
        # audio qformer
        if self.use_audio_qformer:
            self.audio_position_embedding = nn.Embedding(1024, self.decoder.config.hidden_size) # audio is always encoded to 1024
            self.audio_position_embedding.weight = torch.nn.Parameter(self.decoder.model.encoder.embed_positions.weight.clone().detach())

            # set them to learnable
            if not config['audio_qformer'].get('freeze_pos_embeds', False):
                self.audio_position_embedding.weight.requires_grad = True

            self.audio_Qfromer_layernorm_embedding = nn.LayerNorm(self.decoder.config.hidden_size)

            self.num_audio_query_token = config['audio_qformer']['num_audio_query_token']
            self.audio_Qformer, self.audio_query_tokens = \
                        self.init_audio_Qformer(config=config,
                                                num_query_token=self.num_audio_query_token,
                                                vision_width=self.decoder.config.hidden_size)

            # disable word and pos embeddings as we are not gonna use them
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

            if config['audio_qformer']['frozen_audio_Qformer']:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                logging.info('Audio Qformer is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                logging.info('Audio Qformer is not frozen')
            logging.info('Initializing Audio Qformer Done')
            self.audio_qformer_proj = nn.Linear(config['audio_qformer']['hidden_size'], self.decoder.config.hidden_size)
        # text qformer
        if self.use_text_qformer:
            logging.info('Initializing Text Qformer')
            self.text_prompt_position_embedding = nn.Embedding(1024, self.decoder.config.hidden_size) 
            # initialize with bart embeddings
            self.text_prompt_position_embedding.weight = torch.nn.Parameter(self.decoder.model.encoder.embed_positions.weight.clone().detach())
            
            # set them to learnable
            if not config['text_qformer'].get('freeze_pos_embeds', False):
                self.text_prompt_position_embedding.weight.requires_grad = True

            self.text_Qfromer_layernorm_embedding = nn.LayerNorm(self.decoder.config.hidden_size)
            # self.text_prompt_position_embedding.weight.data.normal_(mean=0.0, std=0.02) # TODO: fix the std
            self.num_text_query_token = config['text_qformer']['num_text_query_token']
            self.input_vid2tex_query_embed = config['text_qformer']['input_audio2tex_query_embed']
            self.detach_video_query_embed = config['text_qformer']['detach_video_query_embed']
            if self.input_vid2tex_query_embed:
                # assert num_video_query_token == num_text_query_token
                self.text_Qformer = self.init_text_Qformer(
                    config=config,
                    num_query_token=self.num_text_query_token, \
                    text_width=self.decoder.config.hidden_size, \
                    input_vid2tex_query_embed=self.input_vid2tex_query_embed
                )
                self.audio_proj_to_query = nn.Linear(self.decoder.config.hidden_size, config['text_qformer']['hidden_size'])
            else:
                self.text_Qformer, self.text_query_tokens = self.init_text_Qformer(
                    config=config,
                    num_query_token=self.num_text_query_token, \
                    text_width=self.decoder.config.hidden_size, \
                    input_vid2tex_query_embed=self.input_vid2tex_query_embed
                )

            # disable embds layers that we are not gonna use
            self.text_Qformer.cls = None
            self.text_Qformer.bert.embeddings.position_embeddings = None

            # initalize with bart word embeddings
            self.text_Qformer.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(self.decoder.model.encoder.embed_tokens.weight.clone().detach())

            # set to learnable
            if not config['text_qformer'].get('freeze_word_embeds', False):
                self.text_Qformer.bert.embeddings.word_embeddings.weight.requires_grad = True

            for layer in self.text_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            

            if config['text_qformer']['frozen_text_Qformer']:
                for name, param in self.text_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.text_prompt_position_embedding.named_parameters():
                    param.requires_grad = False
                if not self.input_vid2tex_query_embed:
                    self.text_query_tokens.requires_grad = False
                logging.info('Text Qformer is frozen')
            else:
                for name, param in self.text_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.text_prompt_position_embedding.named_parameters():
                    param.requires_grad = True
                if not self.input_vid2tex_query_embed:
                    self.text_query_tokens.requires_grad = True
                logging.info('Text Qformer is not frozen')
            
            # initalize tokenizer
            # self.text_qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.text_qformer_tokenizer = BartTokenizer.from_pretrained(decoder_name) # use same tokenizer as bart model (since we copied its word embeddings)
            self.text_vocab_size = len(self.text_qformer_tokenizer)
            self.text_qformer_proj = nn.Linear(config['text_qformer']['hidden_size'], self.decoder.config.hidden_size)

        # add calp embeddings 
        if self.use_clap_embeds:
            config['clap_embeds']['ckpt'] = get_checkpoint_path('music_speech_audioset_epoch_15_esc_89.98', local_ckpt_path=config['clap_embeds']['ckpt'])
            self.clap_model = CLAP_Module(amodel=config['clap_embeds'].get('model', 'HTSAT-base'), enable_fusion=False, device=self.device)
            self.clap_model.load_ckpt(config['clap_embeds']['ckpt']) # laod default checkpoint 

            for name, param in self.clap_model.named_parameters():
                param.requires_grad = False

            # add projection layer
            self.audio_clap_resampler = torchaudio.transforms.Resample(self.resample_rate, 48000).to(self.device)
            self.clap_proj = nn.Linear(config['clap_embeds']['embed_dim'], self.decoder.config.hidden_size)
            self.clap_layernorm = nn.LayerNorm(self.decoder.config.hidden_size)


        # add special tokens
        if 'tokenizer' in config and 'special_tokens' in config['tokenizer'] and self.add_special_tokens:
            print("[INFO] adding special tokens", config['tokenizer']['special_tokens'])
            # increase vocab size
            if self.use_text_qformer:
                self.text_qformer_tokenizer.add_special_tokens({'additional_special_tokens':config['tokenizer']['special_tokens']})
                
                if self.resize_token_embeds:
                    self.text_Qformer.resize_token_embeddings(len(self.text_qformer_tokenizer)) # it will be trainable anyway # TODO: deactivate
            else:
                self.tokenizer.add_special_tokens({'additional_special_tokens':config['tokenizer']['special_tokens']})
                if self.resize_token_embeds:
                    self.decoder.resize_token_embeddings(len(self.tokenizer))  # it will be trainable anyway # TODO: deactivate
                    self.text_vocab_size = len(self.tokenizer)
            
            self.num_added_special_tokens = len(config['tokenizer']['special_tokens'])
            self.special_tokens_embed = SpecialTokensEmbeddings(model=self.decoder.model,
                                                                base_vocab_size=self.text_vocab_size,
                                                                base_tokenizer=self.tokenizer,
                                                                special_tokens=config['tokenizer']['special_tokens'],
                                                                )
            # training only these tokens and blocking the gradients on the others messes up the gradient momentun and the optimization, instead we we will use a separate embed layer initialized with the bos, and eos tokens
            

        logging.info('Initializing Text Qformer Done')
        # handle longer input seq
        audio_max_tokens = 1024 if not self.use_audio_qformer else self.num_audio_query_token
        text_max_tokens = self.max_prompt_len if self.use_meta else 0
        if self.use_text_qformer:
            if self.input_vid2tex_query_embed:
                text_max_tokens = audio_max_tokens
            else:
                text_max_tokens = self.num_text_query_token
        
        # extra token for CLAP, two for offsent in bart, two for bos and eos, and the rest because why not
        self.decoder = self.adjust_max_pos_embeds(self.decoder, audio_max_tokens+text_max_tokens+20+self.use_clap_embeds) 
        

        # dropout kayer 
        self.audio_features_dropout = nn.Dropout(p=config['model']['audio_features_dropout_p'])
        self.text_features_dropout = nn.Dropout(p=config['model']['text_features_dropout_p'])

        # load bert model for bertscore, skip for efficient gpu memory usage
        self.bert_model = None
        # self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # for name, param in self.bert_model.named_parameters():
        #     param.requires_grad = False

        # update deocder freezed components to exclude newly added tokens and positional encoding 
        self.set_decoder_requires_grad(freeze=freeze_decoder, freeze_embed_layer=freeze_embed_layer) 

        if 'pretrain_path' in config['training']:
            self.load_from_checkpoint(config['training']["pretrain_path"])


    def load_from_checkpoint(self, ckpt_path, ignore_keys=['clap_model']):
        main_logger = logger.bind(indent=1)
        pretrain_checkpoint = torch.load(ckpt_path)
        state_dict = pretrain_checkpoint.get("state_dict", pretrain_checkpoint.get('model', None))
        
        # skip clap key
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    main_logger.info("[INFO] Deleting key {} from state_dict.".format(k))
                    del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        main_logger.info(
            f"[INFO] Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            main_logger.info(f"[WARNING] Missing Keys: {missing}")
        if len(unexpected) > 0:
            main_logger.info(f"[WARNING] Unexpected Keys: {unexpected}")

    
    def adjust_max_pos_embeds(self, model, new_max_pos):
        # Current maximum position embeddings
        current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape

        self.extended_seq_length = new_max_pos - current_max_pos
        if current_max_pos >= new_max_pos:
            return model
        
        # Extend the position embeddings
        new_pos_embed = model.model.encoder.embed_positions.weight.new_empty(new_max_pos, embed_size)
        torch.nn.init.normal_(new_pos_embed, mean=0, std=model.config.init_std)

        new_pos_embed[:current_max_pos] = model.model.encoder.embed_positions.weight
        last_embed = model.model.encoder.embed_positions.weight[-1] # copying last embed for initalization
        for pos in range(current_max_pos, new_max_pos):
            new_pos_embed[pos] = last_embed

        # Replace old position embeddings with the new one
        model.model.encoder.embed_positions.weight = torch.nn.Parameter(new_pos_embed)
        return model


    def encode_audioQformer_audio(self, audio_embeds):
        device = audio_embeds.device
        # input shape b,c,t,h,w
        # audio: B x 1024 x 768
        batch_size,num_tokens,_ = audio_embeds.size()
        with self.maybe_autocast():

            # add frame_pos embedding
            position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            audio_position_embeddings = self.audio_position_embedding(position_ids)
            frame_hidden_state = audio_position_embeddings + audio_embeds

            # layer norm
            frame_hidden_state = self.audio_Qfromer_layernorm_embedding(frame_hidden_state)

            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            audio_query_tokens = self.audio_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state
            # TODO: double check that removing the projection is alright

            atts_llama = torch.ones(audio_hidden.size()[:-1], dtype=torch.long).to(device)
        return audio_hidden, atts_llama
    
    def encode_textQformer_prompt(self, meta, audio_embeds):
        # audi_embeds: B x 1024 x 768
        # self.tokenizer.pad_token = self.llama_tokenizer.bos_token
        # Description: We add positional encoder to the word embeddings (initalized with bart), add learned positional encoder (initalized with bart), then pass it to Qformer
        raise "handling special tokens is not Implemented yet"
        device = audio_embeds.device
        meta_txt_tokenized = self.text_qformer_tokenizer(meta,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_prompt_len,
                                return_tensors="pt",
                                add_special_tokens=True) # TODO: Consider disabling
        
        meta_txt_tokenized_ids = meta_txt_tokenized["input_ids"].to(self.device)
        prompt_atts = meta_txt_tokenized['attention_mask']
        prompt_atts = prompt_atts.to(device)

        # encode prompts
        prompt_embeds = self.text_Qformer.bert.embeddings.word_embeddings(meta_txt_tokenized_ids)
        batch_size, num_tokens, _ = prompt_embeds.shape

        # add positional encoding
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.text_prompt_position_embedding(position_ids)
        prompt_embeds = position_embeds + prompt_embeds 

        # layer norm
        prompt_embeds = self.text_Qfromer_layernorm_embedding(prompt_embeds)

        if self.input_vid2tex_query_embed:
            audio_embeds = self.audio_proj_to_query(audio_embeds)
            if not self.detach_video_query_embed:
                query_embeds = audio_embeds.type(prompt_embeds.dtype)
            else:
                query_embeds = audio_embeds.detach().type(prompt_embeds.dtype)
        else:
            query_embeds = self.text_query_tokens.expand(batch_size, -1, -1)
        
        # print("query_embeds", query_embeds.shape)
        # print("prompt_embeds", query_embeds.shape)
        text_query_output = self.text_Qformer.bert(
            query_embeds=query_embeds,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_atts, 
            return_dict=True,
            )
        prompt_embeds = text_query_output.last_hidden_state
        prompt_atts = torch.ones(prompt_embeds.size()[:-1], dtype=torch.long).to(device)

        return prompt_embeds, prompt_atts
    
    @property
    def device(self):
        return list(self.parameters())[0].device

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward_encoder(self, audios):
        outputs = self.encoder(audios)
        print("HTSAT embedding", outputs.last_hidden_state.shape)
        outputs = self.enc_to_dec_proj(outputs.last_hidden_state)

        # dropout 
        outputs = self.audio_features_dropout(outputs)
        return outputs

    def combine_attn_masks(self, mask_1, n_1, mask_2, n_2):
        if len(mask_1.shape) != len(mask_2.shape):
            if len(mask_1.shape) == 1:
                mask_1 = mask_1.unsqueeze(1).expand(-1, n_1)
            if len(mask_2.shape) == 1:
                mask_2 = mask_2.unsqueeze(1).expand(-1, n_2)
        
        if len(mask_1.shape) == 1:
            return torch.cat([mask_1, mask_2], dim=0)
        else:
            return torch.cat([mask_1, mask_2], dim=1)
    
    def add_bos_embds(self, embeds, attn_mask):
        batch_size = embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                            dtype=torch.long).to(embeds.device) * self.tokenizer.bos_token_id
        bos_embed = self.get_bart_token_embeds(bos)
        bos_atts = attn_mask[:, :1]

        attn_mask = self.combine_attn_masks(bos_atts, 1, attn_mask, embeds.shape[1])
        embeds = torch.cat([bos_embed, embeds], dim=1).to(embeds.device)

        return embeds, attn_mask
        

    def forward_decoder(self, text, audio, meta, encoder_outputs):
        attn_mask = torch.ones(encoder_outputs.size()[:-1], dtype=torch.long, device=encoder_outputs.device)

        if self.use_audio_qformer:
            encoder_outputs, attn_mask = self.encode_audioQformer_audio(encoder_outputs)
            encoder_outputs = self.audio_qformer_proj(encoder_outputs)
        
        encoded_meta = None
        if self.use_meta and any(meta) and self.use_text_qformer:
            encoded_meta, meta_attn_mask = self.encode_textQformer_prompt(meta=meta, audio_embeds=encoder_outputs) # TODO: skip attention mask for now
            encoded_meta = self.text_qformer_proj(encoded_meta)
        elif self.use_meta and any(meta):
            encoded_meta, meta_attn_mask = self.encode_meta(meta, return_dict=False, add_bos=False, add_eos=True)

        if encoded_meta is not None:
            # dropout on text
            encoded_meta = self.text_features_dropout(encoded_meta)
            attn_mask = self.combine_attn_masks(attn_mask, encoder_outputs.shape[1], meta_attn_mask, encoded_meta.shape[1])
            encoder_outputs = torch.cat([encoder_outputs, encoded_meta], dim=1).to(encoder_outputs.device)

        # add bos and eos embeds 
        encoder_outputs, attn_mask = self.add_bos_embds(encoder_outputs, attn_mask)
        
        encoder_outputs = self.decoder.model.encoder(
                input_ids=None,
                inputs_embeds=encoder_outputs,
                attention_mask=attn_mask,
                return_dict=True
            )["last_hidden_state"]

        # clap
        if self.use_clap_embeds:
            clap_audio = self.audio_clap_resampler(audio)
            clap_embeddings = self.clap_model.get_audio_embedding_from_data(clap_audio, use_tensor=True)
            clap_embeddings = self.clap_proj(clap_embeddings)
            clap_embeddings = self.clap_layernorm(clap_embeddings)
            attn_mask = self.combine_attn_masks(torch.ones((clap_embeddings.shape[0], 1), dtype=torch.long).to(self.device), 1, attn_mask, encoder_outputs.shape[1])
            encoder_outputs = torch.cat([clap_embeddings.unsqueeze(1), encoder_outputs], dim=1)
            
            
        text = self.tokenizer(text,
                              padding='longest',
                              truncation=True,
                              max_length=self.tokenizer_max_length,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
        )

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=(encoder_outputs,),
            return_dict=True
        )
        lm_logits = decoder_outputs["logits"]
        loss = self.loss_fct(lm_logits.view(-1, self.text_vocab_size), decoder_targets.view(-1))
        return loss

    def forward(self, audio, text, meta):
        # meta is a dict of additional meta text
        audio_embeds = self.forward_encoder(audio)
        loss = self.forward_decoder(text, audio, meta, audio_embeds)

        return loss

    # self.tokenizer.bos_token_id
    def get_bart_token_embeds(self, token_ids):
        return self.decoder.model.encoder.embed_tokens(token_ids) 

    def encode_meta(self, meta, add_pos_embeds=False, return_dict=False, add_bos=False, add_eos=False):
        """obtain the token embedding of metadata with the default bart model"""
        # Description: We tokenize the text (add bos, eos), embed the tokens with bart (pretrained and frozen), (optionally, we add positional encoding and layer norm
        
        # TODO: IMPORTANT: combining the tokens of different norm from audio and language might results om a situation where the layernorm reducsthe influence of one of them
        meta_txt_tokenized = self.tokenizer(meta,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_prompt_len,
                                return_tensors="pt",
                                add_special_tokens=add_bos and add_eos)
        input_ids = meta_txt_tokenized["input_ids"].to(self.device)
        attn_mask = meta_txt_tokenized['attention_mask'].to(self.device)
        # handle padding, bos and eos
        if add_bos and not add_eos:
            bos_token_id = self.tokenizer.bos_token_id
            bos_tokens = torch.full((input_ids.size(0), 1), bos_token_id, dtype=torch.long, device=self.device)
            input_ids = torch.cat([bos_tokens, input_ids], dim=1)

            bos_mask = torch.ones((attn_mask.size(0), 1), dtype=torch.long, device=self.device)
            attn_mask = torch.cat([bos_mask, attn_mask.to(bos_mask.device)], dim=1)

        
        if add_eos and not add_bos:
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id

            # add one padding token to make sure all sentences has at least one pad token to replace
            input_ids = torch.cat([input_ids, torch.full((input_ids.size(0), 1), pad_token_id, dtype=torch.long, device=self.device)], dim=1)
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=torch.long, device=self.device)], dim=1)

            pad_positions = (input_ids == pad_token_id).long().argmax(dim=1).to(self.device).contiguous()
            input_ids.scatter_(1, pad_positions.unsqueeze(1), eos_token_id)

            attn_mask = attn_mask.to(pad_positions.device)
            attn_mask.scatter_(1, pad_positions.unsqueeze(1), 1)
            

        meta_txt_tokenized_ids = input_ids.to(self.device)
        
        # new tokenizer
        if self.add_special_tokens:
            encoded_meta = self.special_tokens_embed(meta_txt_tokenized_ids) 
        else:
            encoded_meta = self.decoder.model.encoder.embed_tokens(meta_txt_tokenized_ids)
        
        if add_pos_embeds:
            embed_pos = self.decoder.model.encoder.embed_positions(meta_txt_tokenized_ids)
            embed_pos = embed_pos.to(encoded_meta.device)
            encoded_meta = encoded_meta + embed_pos
            encoded_meta = self.decoder.model.encoder.layernorm_embedding(encoded_meta)

        if return_dict:
            if isinstance(encoded_meta, dict):
                return encoded_meta
            ret = {}
            ret['last_hidden_state'] = encoded_meta
            ret['attn_mask'] = attn_mask
            return ret
        else:
            return encoded_meta, attn_mask

        
    def generate(self,
                 samples,     
                 meta=[""], 
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=30,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):

        encoder_outputs = self.forward_encoder(samples)
        print("audio_embeds", encoder_outputs.shape)
        attn_mask = torch.ones(encoder_outputs.size()[:-1], dtype=torch.long, device=encoder_outputs.device)

        if self.use_audio_qformer:
            encoder_outputs, attn_mask = self.encode_audioQformer_audio(encoder_outputs)
            encoder_outputs = self.audio_qformer_proj(encoder_outputs)

        encoded_meta = None
        if self.use_meta and any(meta) and self.use_text_qformer:
            encoded_meta, meta_attn_mask = self.encode_textQformer_prompt(meta=meta, audio_embeds=encoder_outputs)
            encoded_meta = self.text_qformer_proj(encoded_meta)
        elif self.use_meta and any(meta):
            encoded_meta, meta_attn_mask = self.encode_meta(meta, return_dict=False, add_bos=False, add_eos=True) # TODO: this ignores prompt attention. Fix later

        if encoded_meta is not None:
            encoded_meta = self.text_features_dropout(encoded_meta)
            attn_mask = self.combine_attn_masks(attn_mask, encoder_outputs.shape[1], meta_attn_mask, encoded_meta.shape[1])
            encoder_outputs = torch.cat([encoder_outputs, encoded_meta], dim=1)

        # add bos and eos embeds 
        encoder_outputs, attn_mask = self.add_bos_embds(encoder_outputs, attn_mask)

        if self.use_clap_embeds:
            clap_audio = self.audio_clap_resampler(samples)
            clap_embeddings = self.clap_model.get_audio_embedding_from_data(clap_audio, use_tensor=True)
            clap_embeddings = self.clap_proj(clap_embeddings)
            clap_embeddings = self.clap_layernorm(clap_embeddings)
            # attn_mask = self.combine_attn_masks(attn_mask, encoder_outputs.shape[1], torch.ones((clap_embeddings.shape[0], 1), dtype=torch.long).to(self.device), 1)
            attn_mask = self.combine_attn_masks(torch.ones((clap_embeddings.shape[0], 1), dtype=torch.long).to(self.device), 1, attn_mask, encoder_outputs.shape[1])
            encoder_outputs = torch.cat([clap_embeddings.unsqueeze(1), encoder_outputs], dim=1)
            

        # Encoder pass
        encoder_outputs = self.decoder.model.encoder(
            input_ids=None,
            attention_mask=attn_mask,
            head_mask=None,
            inputs_embeds=encoder_outputs,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True)
        
        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        input_ids[:, 0] = self.decoder.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)

        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=attn_mask,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.decoder.generate(input_ids=None,
                                            attention_mask=attn_mask,
                                            decoder_input_ids=input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            encoder_outputs=encoder_outputs,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            inputs_embeds=None,
                                            decoder_inputs_embeds=None,
                                            use_cache=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            max_length=max_length,
                                            min_length=min_length,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty)

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions



    # pl functions 
    def get_meta_dict(self, batch, meta_keys=None, drop_keys=[]):
        ordered_keys = ['labels', 'video_caption', 'title', 'subtitle', 'videollama_caption']
        
        # prepare meta
        if not self.use_meta:
            return ""

        if meta_keys is None:
           meta_keys = self.meta_keys
        
        # pad meta
        batch_size = len(batch['title'])
        if 'original_captions' in batch:
            meta_subtitle = batch['original_captions'] if batch['original_captions'] else batch['automatic_captions']
        else:
            meta_subtitle = [""] * batch_size
        for key in ordered_keys:
            batch[key] = batch.get(key, [""] * batch_size)
        
        
        if self.add_special_tokens:
            meta = {
                "labels": [f"<LBL> {t} </LBL>" if not 'labels' in drop_keys else "<LBL>  </LBL>" for t in batch['labels'] ],
                "video_caption" : [ f"<HQVC> {t} </HQVC>" if not 'video_caption' in drop_keys else "<HQVC>  </HQVC>" for t in batch['video_caption'] ],
                "title": [f"<TITLE> {t} </TITLE>" if not 'title' in drop_keys else "<TITLE>  </TITLE>" for t in batch['title'] ],
                "description": [f"<DESC> {t} </DESC>" if not 'description' in drop_keys else "<DESC>  </DESC>" for t in batch['description']],
                "subtitle": [f"<SUB> {t} </SUB>" if not 'subtitle' in drop_keys else "<SUB>  </SUB>" for t in meta_subtitle ],
                "videollama_caption": [f"<AVC> {t} </AVC>" if not 'videollama_caption' in drop_keys else "<AVC>  </AVC>" for t in batch['videollama_caption']],
            }
        else:
            meta = {
                "labels": [f"Label {t}" if not 'labels' in drop_keys else "" for t in batch['labels'] ],
                "video_caption" : [ f"Caption {t}" if not 'video_caption' in drop_keys else "" for t in batch['video_caption'] ],
                "title": [f"Title {t}" if not 'title' in drop_keys else "" for t in batch['title'] ],
                "description": [f"Description {t}" if not 'description' in drop_keys else "" for t in batch['description']],
                "subtitle": [f"Subtitle {t}" if not 'subtitle' in drop_keys else "" for t in meta_subtitle ],
                "videollama_caption": [f"<AVC> {t} </AVC>" if not 'videollama_caption' in drop_keys else "" for t in batch['videollama_caption']],
            }

        # filter
        filtered_meta = {key: meta[key] for key in meta_keys if key in meta}

        # combine to a single string
        ordered_keys = [k for k in ordered_keys if k in filtered_meta.keys()]
        meta_processed = []
        if filtered_meta:
            first_key = list(filtered_meta.keys())[0]
            for i in range(len(filtered_meta[first_key])):
                meta_processed.append(' \n '.join(filtered_meta[k][i] for k in ordered_keys))

            return meta_processed 
        else:
            return ""

    def training_step(self, batch, batch_idx):
        audio = batch['waveform'].squeeze(1)
        text = batch['gt_audio_caption'] # list of captions
        audio = audio.to(self.device)

        # prepare meta
        if 'meta_keys' in self.config['model'].keys():
            # drop meta keys p% of the time 
            if 'drop_meta_p' in self.config['model'].keys():
                drop_meta_keys_list = [k for k in self.config['model']['meta_keys'] if np.random.rand() < self.config['model']['drop_meta_p']]
            else:
                drop_meta_keys_list = []
            meta = self.get_meta_dict(batch, meta_keys=self.config['model']['meta_keys'], drop_keys=drop_meta_keys_list)
            loss = self.forward(audio, text, meta=meta)
        else:
            loss = self.forward(audio, text)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log_dict({"train/loss": loss.cpu().item(),
                    "lr":lr},
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True)
        return loss


    @torch.no_grad()
    def validation_step(self, batch_data, batch_idx, dataloader_idx=0):
        val_logger = logger.bind(indent=1)
        val_outputs = {}
        with torch.no_grad():
            # test for different beams 
            for beam_size in self.eval_beam_sizes:
                audios = batch_data['waveform'].squeeze(1) 
                caption_dict = batch_data['gt_audio_caption'] # expects a list of N captions for each sample. B x N x tokens
                audio_names = batch_data['fname']

                # audios, caption_dict, audio_names, audio_ids = batch_data
                # move data to GPU
                audios = audios.to(self.device)
                texts = [captions[0] for captions in caption_dict]

                # prepare meta
                if self.use_meta:
                    meta = self.get_meta_dict(batch_data, meta_keys=self.config['model']['meta_keys'])
                else:
                    meta = ""
                val_loss = self.forward(audios, texts, meta=meta)
                output = self.generate(samples=audios,
                                    meta=meta,
                                    num_beams=beam_size)
                
                self.log_dict({f"{self.val_loaders_labels[dataloader_idx]}/loss": val_loss.cpu().item()},
                        prog_bar=True,
                        logger=True,
                        on_step=True,
                        on_epoch=True)

                self.val_outputs[dataloader_idx][beam_size]['y_hat_all'].extend(output)
                self.val_outputs[dataloader_idx][beam_size]['ref_captions_dict'].extend(caption_dict)
                self.val_outputs[dataloader_idx][beam_size]['file_names_all'].extend(audio_names)


    @torch.no_grad()
    def predict_step(self, batch_data, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            # test for different beams 
            for beam_size in self.eval_beam_sizes:
                audios = batch_data['waveform'].squeeze(1) 
                audio_paths = batch_data['fname']

                # audios, caption_dict, audio_names, audio_ids = batch_data
                # move data to GPU
                audios = audios.to(self.device)

                # prepare meta
                if self.use_meta:
                    meta = self.get_meta_dict(batch_data, meta_keys=self.config['model']['meta_keys'])
                else:
                    meta = ""
                output = self.generate(samples=audios,
                                    meta=meta,
                                    num_beams=beam_size)
        # save predicted caption if caption_store_key is provided
        try:
            if hasattr(self, "caption_store_key"):
                for caption, fname in zip(output, audio_paths):
                    json_path = f"{'.'.join(fname.split('.')[:-1])}.json"
                    data = load_json(json_path)
                    data[self.caption_store_key] = caption
                    write_json(data, json_path)
                    print("[INFO] saved predicted caption for:", json_path)
                    if hasattr(self, "audio_save_path"):
                        shutil.copy(fname, self.audio_save_path)
                        shutil.copy(json_path, self.audio_save_path)
        except Exception as e:
            print("[ERROR] while saving:", e)
              


    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_start_time = time.time()
    
    def on_train_epoch_end(self):
        self.log("time/train_epoch", time.time() - self.train_start_time, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # reset buffer 
        self.val_outputs = []
        for idx in range(len(self.val_loaders_labels)):
            val_loader_out = {}
            for beam_size in self.eval_beam_sizes:
                val_loader_out[beam_size] = {}
                val_loader_out[beam_size]['y_hat_all'] = []
                val_loader_out[beam_size]['ref_captions_dict'] = []
                val_loader_out[beam_size]['file_names_all'] = []
            
            self.val_outputs.append(val_loader_out)
        
        self.val_start_time = time.time()
        
    
    def on_validation_epoch_end(self):
        
        # place a barrier to ensure that all ranks reach the gather operation
        torch.distributed.barrier()
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, self.val_outputs)
        torch.distributed.barrier()
        
        metrics_log = {}
        # all ranks should excute the blocks to avoid deadlock 
        self.gathered_output_dict = []
        for idx in range(len(self.val_loaders_labels)):
            val_loader_out = {}
            for beam_size in self.eval_beam_sizes:
                val_loader_out[beam_size] = {}
                val_loader_out[beam_size]['y_hat_all'] = sum([g_out[idx][beam_size]['y_hat_all'] for g_out in gathered], [])
                val_loader_out[beam_size]['ref_captions_dict'] = sum([g_out[idx][beam_size]['ref_captions_dict'] for g_out in gathered], [])
                val_loader_out[beam_size]['file_names_all'] = sum([g_out[idx][beam_size]['file_names_all'] for g_out in gathered], [])
            
            self.gathered_output_dict.append(val_loader_out)

        
        val_logger = logger.bind(indent=1)
        for split, dataloader_outputs in zip(self.val_loaders_labels, self.gathered_output_dict):
            val_logger.info(f"[INFO] evaluating metrics for split: {split}")
            for beam_size in self.eval_beam_sizes:
                val_logger.info(f"[INFO] evaluating metrics for split: {split}, beam: {beam_size}")
                y_hat_all = dataloader_outputs[beam_size]['y_hat_all']
                ref_captions_dict = dataloader_outputs[beam_size]['ref_captions_dict']
                file_names_all = dataloader_outputs[beam_size]['file_names_all']
                captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all,
                                                    self.log_output_dir, self.current_epoch, beam_size=beam_size)
                
                if len(captions_pred) == 0 or len(captions_gt) == 0:
                    continue
                
                try:
                    metrics = evaluate_metrics(captions_pred, captions_gt, nb_reference_captions=5, bert_model=self.bert_model, exclude_metrics=self.exclude_metrics)

                    
                    def get_score(metrics, key):
                        if key in metrics:
                            return float(metrics[key]['score'])
                        else:
                            return 0

                    spider = get_score(metrics, 'spider')
                    cider = get_score(metrics, 'cider')
                    spice = get_score(metrics, 'spice')
                    bleu_1 = get_score(metrics, 'bleu_1')
                    bleu_4 = get_score(metrics, 'bleu_4')
                    rouge_l = get_score(metrics, 'rouge_l')
                    meteor = get_score(metrics, 'meteor')

                    val_logger.info(f'Cider: {cider:7.4f}')
                    val_logger.info(
                        f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}')
                    
                    metrics_log = {f"{split}/spider_beam_{beam_size}" : spider,
                                f"{split}/cider_beam_{beam_size}":cider,
                                f"{split}/spice_beam_{beam_size}":spice,
                                f"{split}/bleu_1_beam_{beam_size}":bleu_1,
                                    f"{split}/bleu_4_beam_{beam_size}":bleu_4,
                                    f"{split}/rouge_l_beam_{beam_size}":rouge_l,
                                    f"{split}/meteor_beam_{beam_size}":meteor }
                    if 'bert_score' in metrics:
                        bert_score = metrics.pop('bert_score')
                        metrics_log[f"{split}/bertscore_beam_{beam_size}"] = bert_score
                        val_logger.info(f"Bert score {bert_score}")

                    self.log_dict(metrics_log,
                                prog_bar=True,
                                logger=True,
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True)

                    for metric, values in metrics.items():
                        val_logger.info(f'beam search (size {beam_size}): {metric:<7s}: {values["score"]:7.4f}')
                except Exception as e:
                    print("Error while calculating the metrics.")
                    metrics_log = {}
                    
        self.log("time/val_epoch", time.time() - self.val_start_time, on_step=False, on_epoch=True, logger=True)
        return metrics_log


    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(),
                              lr=self.config["optim_args"]["lr"],
                              betas=self.config["optim_args"]["betas"],
                              eps=self.config["optim_args"]["eps"],
                              momentum=self.config["optim_args"]["momentum"],
                              weight_decay=self.config["optim_args"]["weight_decay"],
                              optimizer_name=self.config["optim_args"]["optimizer_name"])

        # setup dataloader as it is needed for the scheduler
        print("[INFO] warmup_length", self.config["optim_args"]["warmup_epochs"] * self.train_loader_len)
        print("[INFO] base_lr", self.config["optim_args"]["lr"])
        scheduler = cosine_lr_scheduler(optimizer,
                              base_lr=self.config["optim_args"]["lr"],
                              warmup_length=self.config["optim_args"]["warmup_epochs"] * self.train_loader_len,
                              steps= self.train_loader_len * self.config["step"]["epochs"])

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",
                "frequency": 1
            }
        }
    

