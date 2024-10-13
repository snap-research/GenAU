import os
import pandas as pd
from librosa.filters import mel as librosa_mel_fn
import random
from torch.utils.data import Dataset
import torch.nn.functional
import torch
import numpy as np
import torchaudio
from moviepy.editor import VideoFileClip
from torchvision import transforms as T
import src.utilities.audio as Audio
import numpy as np
from torch.utils.data.dataloader import default_collate
import shutil
import re
from src.tools.io import load_file, write_json, load_json
from src.tools.torch_utils import spectral_normalize_torch, random_uniform
from src.tools.training_utils import build_dataset_json_from_list
import gc
import librosa
import threading

class VideoAudioDataset(Dataset):
    def __init__(
        self,
        config=None,
        load_video=True,
        load_audio=True,
        keep_audio_files=True,
        video_transform=None,
        target_frame_cnt=10,
        split="train",
        waveform_only=False,
        add_ons=[],
        dataset_json=None,
        sample_single_caption=True,
        augment_p=0.0,
        limit_data_percentage = None,
        cache_dir=None
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.target_frame_cnt = target_frame_cnt
        self.config = config
        self.split = split
        self.pad_wav_start_sample = 0  # If none, random choose
        self.load_video = load_video
        self.load_audio = load_audio
        self.keep_audio_files = keep_audio_files
        self.sample_single_caption = sample_single_caption
        self.limit_data_percentage = config['data'].get('limit_data_percentage', False) 
        self.trim_wav = False
        self.waveform_only = waveform_only
        self.augment_p = augment_p
        self.add_ons = [eval(x) for x in add_ons]
        self.consistent_start_time = config['data'].get('consistent_start_time', False)
        
        self.cache_dir = config['data'].get('cache_dir', None)
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        print("[INFO] Add-ons:", self.add_ons)
        self.obtained_samples = 0

        # transforms
        if video_transform is None:
            video_transform = T.Compose(
                [
                    # T, H, W, C
                    T.ToTensor()
                ]
            )
        self.video_transform = video_transform
        self.build_setting_parameters()

        # For an external dataset
        if dataset_json is not None:
            self.data = dataset_json["data"]
            self.dataset_name = "TEST"
            self.id2label, self.index_dict, self.num2label = {}, {}, {}
        else:
            self.metadata_root = load_json(self.config['data']["metadata_root"])
            self.dataset_name = self.config["data"][self.split]
            assert split in self.config["data"].keys(), (
                "The dataset split %s you specified is not present in the config. You can choose from %s"
                % (split, self.config["data"].keys())
            )
            self.retrieve_paths()
            
        
        if split=='train' and self.limit_data_percentage:
            print(f"[INFO] limiting data to only {self.limit_data_percentage} of the total data {len(self.data)}")
            num_datapoints = int(len(self.data) * self.limit_data_percentage)
            
            # fix the seed to make sure we select the same data.
            np.random.seed(42)
            selected_idx = np.random.randint(0, len(self.data), size=num_datapoints)
            
            # select 
            self.video_json_paths = np.asarray(self.video_json_paths)[selected_idx]
            self.data = np.asarray(self.data)[selected_idx]
            self.datasets_of_datapoints = np.asarray(self.datasets_of_datapoints)[selected_idx]

        self.build_dsp()
        
        if 'keys_synonyms' in config['data']:
            self.keys_synonyms = config['data']['keys_synonyms']
        else:
            self.keys_synonyms = {
                "gt_audio_caption": ["audiocaps_gt_captions", "gt_audio_caption",  "caption", "gt_caption", "gt_captions", 'best_model_w_meta_pred_caption',  "wavcaps_caption"], 
                "tags": ["keywords", "tags"], 
            }
        print("[INFO] Dataset initialize finished")

    def filter_text(self, text):
        filtered_text = re.sub(r'[^\x00-\x7F\u00A0-\u00FF]+', '', text).strip()
        return filtered_text

    def get_data_from_keys(self, data, key, default_value=None):
        """
        Check for each possible key and return the value if found
        """
        if key not in self.keys_synonyms:
            return data.get(key, default_value)
        possible_keys = self.keys_synonyms[key]
        for key in possible_keys:
            if key in data:
                return data[key]
        return default_value  # Or return a default value if none of the keys are found
    
    
    def default_sample(self):
        data = {
                "dataset_name": "UNK",
                "json_path": "UNK",
                "fname": "UNK",  # list
                "waveform": "" if (not self.load_audio) else torch.zeros(1, int(self.sampling_rate * self.duration)),
                # "waveform": torch.zeros(1, int(self.sampling_rate * self.duration)),
                # tensor, [batchsize, t-steps, f-bins]
                "stft": "" if self.waveform_only else torch.zeros(int(self.duration * 100), 512),
                # tensor, [batchsize, t-steps, mel-bins]
                "log_mel_spec": "" if self.waveform_only else torch.zeros(int(self.duration * 100), 64),
                "duration": self.duration,
                "sampling_rate": self.sampling_rate,
                "random_start_sample_in_original_audio_file": -1,
                "labels": "UNK",

                # # video 
                "frames": "",
                
                # additional meta data
                "title": "UNK",
                "url": "UNK",
                "description": "UNK",
                "original_captions": "UNK",
                "automatic_captions": "UNK",
                "gt_audio_caption": "UNK" if self.sample_single_caption else ["UNK"] * 5,
                "video_caption": "UNK",
                "videollama_caption": "UNK",
                "text": "UNK" if self.sample_single_caption else ["UNK"] * 5
            }
            
        return data
        
    def __getitem__(self, index, augment=True):
        
        retries = 0
        max_retries = 1
        
        while retries < max_retries:
            try:
                if '.json' in self.data[index]:
                    dataset_name = self.datasets_of_datapoints[index]
                    absolute_file_path = self._relative_path_to_absolute_path([self.data[index]], dataset_name)[0]
                    if not os.path.exists(absolute_file_path):
                        print(f"file {absolute_file_path} does not exists. Retying..")
                        index = random.randint(0, len(self.data) - 1)
                        retries += 1
                        continue
                else:
                    dataset_name = absolute_file_path = ""
                    
                (
                    index,
                    fname,
                    video_frames,
                    waveform,
                    stft,
                    log_mel_spec,
                    _,  # the one-hot representation of the audio class
                    (datum, mix_datum),
                    random_start,
                ) = self.feature_extraction(index)
                
                data = {
                    "dataset_name": dataset_name,
                    "json_path": absolute_file_path,
                    "fname": fname,  # list
                    "waveform": "" if (not self.load_audio) else waveform.float(),
                    # tensor, [batchsize, t-steps, f-bins]
                    "stft": "" if (stft is None) else stft.float(),
                    # tensor, [batchsize, t-steps, mel-bins]
                    "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),
                    "duration": self.duration,
                    "sampling_rate": self.sampling_rate,
                    "random_start_sample_in_original_audio_file": -1 if random_start is None else random_start,
                    "labels": ', '.join(datum.get('labels', [])),

                    # # video 
                    "frames": video_frames if self.load_video else "",
                    
                    # additional meta data
                    "title": self.filter_text(datum.get('title', '')),
                    "url": self.filter_text(datum.get('url', '')),
                    "description": self.filter_text(self.get_sample_description(datum)),
                    "original_captions": self.filter_text(datum.get('original_captions', '')),
                    "automatic_captions": self.filter_text(datum.get('automatic_captions', '')),
                    "gt_audio_caption": self.get_sample_caption(datum, index=index),
                    "video_caption": datum.get('panda70m_caption_0000', '').replace("<unk>", "").strip(),
                    "videollama_caption": datum.get('videollama_caption_0000', ''),
                }
                
                # select one caption if multiple exists
                if isinstance(data['gt_audio_caption'], list) and len(data['gt_audio_caption']) > 0 and self.sample_single_caption:
                    idx = np.random.randint(len(data['gt_audio_caption']))
                    data['gt_audio_caption'] = data['gt_audio_caption'][idx]

                
                for add_on in self.add_ons:
                    data.update(add_on(self.config, data, self.data[index]))
                
                # augment data
                if augment and np.random.rand() < self.augment_p:
                    data = self.pair_augmentation(data)
                
                data['text'] = data['gt_audio_caption']
                
                self.obtained_samples += 1
                
                if self.obtained_samples % 20 == 0:
                    gc.collect()
                return data
            except Exception as e:
                if '.json' in self.data[index]:
                    dataset_name = self.datasets_of_datapoints[index]
                    file_path = self._relative_path_to_absolute_path([self.data[index]], dataset_name)[0]
                else:
                    file_path = ""
                    
                index = random.randint(0, len(self.data) - 1)
                retries += 1
                print("[ERROR, videoaudio_dataset] error while loading", file_path,  e)
                continue
        return self.default_sample()

    def text_to_filename(self, text):
        return text.replace(" ", "_").replace("'", "_").replace('"', "_")

    def get_dataset_root_path(self, dataset):
        assert dataset in self.metadata_root.keys()
        return self.metadata_root[dataset]

    def get_dataset_metadata_path(self, dataset, key):
        # key: train, test, val, class_label_indices
        try:
            if dataset in self.metadata_root["metadata"]["path"].keys():
                return self.metadata_root["metadata"]["path"][dataset][key]
        except KeyError as e:
            print("Error:", e)
            raise ValueError(
                '[ERROR, videoaudio_dataset] Dataset %s does not metadata "%s" specified' % (dataset, key)
            )

    def __len__(self):
        return len(self.data)
    
    def replace_extension(self, path, new_ext):
        return f"{'/'.join(path.split('.')[:-1])}.{new_ext}"
    
    
    def feature_extraction(self, index):
        # Read wave file and extract feature
        if isinstance(self.data[index], str) and '.json' in self.data[index]:
            dataset_name = self.datasets_of_datapoints[index]
            file_path = self._relative_path_to_absolute_path([self.data[index]], dataset_name)[0]
            datum = load_json(file_path)
        else:
            datum = self.data[index]

        if 'path' in datum and datum['path']:
            datum['path'] = self._relative_path_to_absolute_path([datum['path']], dataset_name)[0]

        if 'wav' in datum and datum['wav']:
            datum['wav'] = self._relative_path_to_absolute_path([datum['wav']], dataset_name)[0]

        
        random_start = None
        log_mel_spec, stft, waveform, frames = None, None, None, None
        audio_file = None

        if self.load_audio and not ('wav' in datum.keys() and os.path.exists(datum['wav'])):
            # assume that a .wav file exists in the same location as the .json file
            wav_path = self.replace_extension(file_path, 'wav')
            flac_path = self.replace_extension(file_path, 'flac')
            if os.path.exists(wav_path):
                datum['wav'] = wav_path
            elif os.path.exists(flac_path):
                datum['wav'] = flac_path
            elif 'wav' in datum:
                del datum['wav']

        # cache wav file: useful when there exists a local memory the is faster to do read operations on it
        if self.load_audio and 'wav' in datum and self.cache_dir is not None:
            target_audio_file_path = f"{self.cache_dir}{datum['wav']}"
            if not os.path.exists(target_audio_file_path):
                os.makedirs(os.path.dirname(target_audio_file_path), exist_ok=True)
                shutil.copy2(datum['wav'] , target_audio_file_path)
            
            # update
            datum['wav'] = target_audio_file_path
        
        save_random_start = False 
        random_start = None
        if self.consistent_start_time: # always sample from the same start time
            if 'random_start_t' in datum:
                random_start = datum.get('random_start_t', None)
                save_random_start = False
            else:
                save_random_start = True
        
        # load audio
        if self.load_audio:
            if 'wav' in datum: 
                (
                    log_mel_spec,
                    stft,
                    waveform,
                    random_start,
                ) = self.read_audio_file(datum["wav"], random_start=random_start)
                
                
                waveform = torch.FloatTensor(waveform)
                
                
                
            
            else:
                (
                    frames,
                    log_mel_spec,
                    stft,
                    waveform,
                    random_start,
                    audio_file
                ) = self.read_video_file(datum["path"], random_start=random_start, load_audio=True)
                waveform = torch.FloatTensor(waveform)

            # load video
            if self.load_video and 'path' in datum:
                (frames, _, _, _, _, _ ) = self.read_video_file(datum["path"], random_start=random_start, load_audio=self.load_audio and waveform is None)
        
        elif self.load_video and 'path' in datum:
            (   
                frames,
                log_mel_spec,
                stft,
                waveform,
                random_start,
                audio_file
            ) = self.read_video_file(datum["path"], random_start=random_start, load_audio=True)
            waveform = torch.FloatTensor(waveform)
        
        if audio_file is not None:
            # update json to include path to audio. Only effective if keep_audio_file is enabled
            updated_json = load_json(file_path)
            updated_json['wav'] = self._absolute_path_to_relative_path([audio_file], dataset_name)[0]
            datum["wav"] = updated_json['wav']
            updated_json['random_start_t'] = random_start
            # write_json(updated_json, file_path)

        elif save_random_start and random_start is not None:
            # update json to include the randomly sampled start time for future experiments
            updated_json = load_json(file_path)
            updated_json['random_start_t'] = random_start
            write_json(updated_json, file_path)

        mix_datum = None
        if self.load_video:
            assert frames.shape == (3, self.target_frame_cnt, self.frame_width, self.frame_height)
        

        # The filename of the wav file
        fname = datum["path"] if 'path' in datum and self.load_video else datum.get('wav', '')
        
        if not fname:
            fname = datum['fname']
            
        
        return (
            index,
            fname,
            frames,
            waveform,
            stft,
            log_mel_spec,
            None,
            (datum, mix_datum),
            random_start,
        )
        
    def combine_captions(self, caption1, caption2, remove_duplicates=False, background=False):
        """
        Useful function to combine two caption when doing mixup augmentation
        """
        words1 = caption1.split()
        words2 = caption2.split()

        seen = set(words1)
        combined_words = words1.copy()
        combined_words.append('and')

        for word in words2:
            if word not in seen or (not remove_duplicates):
                combined_words.append(word)
                seen.add(word)  # Add to set to keep track of seen words
    
        combined_caption = " ".join(combined_words)

        if background:
            combined_caption += " in the background"
        return combined_caption

    def pair_augmentation(self, batch):
        """
        Mixup augmentation function that combines two audio at different weight, such that one audio is considered to be a background sound.
        """
        # load a random audio
        idx = np.random.randint(0, self.__len__())
        second_data = self.__getitem__(idx, augment=False)
        
        if np.random.randint(0, 2):
            ratio = 0.2 + np.random.rand() * 0.2
        else:
            ratio = 0.5

        batch['waveform'] = ((1 - ratio) * batch['waveform'] + ratio * second_data['waveform'])
        batch['gt_audio_caption'] =  self.combine_captions(batch['gt_audio_caption'], second_data['gt_audio_caption'], background=(ratio!=0.5))
        batch['panda_caption'] = f"{batch['panda_caption']} and {second_data['panda_caption']}"
        batch['description'] = f"{batch['description']} and {second_data['description']}"
        return batch


    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

    def merge_paths(self, path1, path2):
        parts1 = path1.split('/')
        parts2 = path2.split('/')
        
        common_part = None
        for i, part in enumerate(parts1):
            if parts1[i:] == parts2[:len(parts1)-i]:
                common_part = i
                break
        
        if common_part is not None:
            merged_path = '/'.join(parts1[:common_part] + parts2)
        else:
            # no common part, simply concatenate
            merged_path = '/'.join([path1, path2])

        return merged_path
    
    def _relative_path_to_absolute_path(self, paths, dataset_name):
        root_path = self.get_dataset_root_path(dataset_name)
        for i, path in enumerate(paths):
            assert path[0] != "/", (
                "The dataset metadata should only contain relative path to the json file: "
                + str(path)
            )
            merged_path = self.merge_paths(
                root_path, path
            )
            if not os.path.exists(merged_path):
                merged_path = self.merge_paths(root_path, path.split('/')[-1])

            paths[i] = merged_path
        return paths

    def _absolute_path_to_relative_path(self, paths, dataset_name):
        root_path = self.get_dataset_root_path(dataset_name)
        for i, path in enumerate(paths):
            assert path[0] == "/", (
                "the json file should be absolute: "
                + str(path)
            )
            paths[i] = os.path.relpath(path, root_path)
        return paths

    def retrieve_paths(self):
        self.video_json_paths = []
        self.data = []
        self.datasets_of_datapoints = []
        print("[INFO] Build dataset split %s from %s" % (self.split, self.dataset_name))
        if type(self.dataset_name) is str:
            video_paths_list = load_file(
                self.get_dataset_metadata_path(self.dataset_name, key=self.split)
            )
            self.video_json_paths = video_paths_list
            self.datasets_of_datapoints = [self.dataset_name] * len(video_paths_list)
        
        elif type(self.dataset_name) is list:
            for dataset_name in self.dataset_name:
                video_paths_list = load_file(
                    self.get_dataset_metadata_path(dataset_name, key=self.split)
                )
                self.datasets_of_datapoints += [dataset_name] * len(video_paths_list)
                self.video_json_paths += video_paths_list
        else:
            raise Exception("[ERROR, videoaudio_dataset] Invalid data format:", type(self.dataset_name))

        self.data = self.video_json_paths
        print("[INFO] Data size: {}".format(len(self.data)))

        return self.data
    
    def build_dsp(self):
        self.mel_basis = {}
        self.hann_window = {}

        self.filter_length = self.config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = self.config["preprocessing"]["stft"]["hop_length"]
        self.win_length = self.config["preprocessing"]["stft"]["win_length"]
        self.n_mel = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]
        
        # video
        self.video_fps = self.config["preprocessing"]["video"]["fps"]
        self.frame_height = self.config["preprocessing"]["video"]["height"]
        self.frame_width = self.config["preprocessing"]["video"]["width"]

        if not self.waveform_only:
            self.STFT = Audio.stft.TacotronSTFT(
                self.config["preprocessing"]["stft"]["filter_length"],
                self.config["preprocessing"]["stft"]["hop_length"],
                self.config["preprocessing"]["stft"]["win_length"],
                self.config["preprocessing"]["mel"]["n_mel_channels"],
                self.config["preprocessing"]["audio"]["sampling_rate"],
                self.config["preprocessing"]["mel"]["mel_fmin"],
                self.config["preprocessing"]["mel"]["mel_fmax"],
            )

    def build_id_to_label(self):
        id2label = {}
        id2num = {}
        num2label = {}
        class_label_indices_path = self.get_dataset_metadata_path(
            dataset=self.config["data"]["class_label_indices"],
            key="class_label_indices",
        )
        if class_label_indices_path is not None:
            df = pd.read_csv(class_label_indices_path)
            for _, row in df.iterrows():
                index, mid, display_name = row["index"], row["mid"], row["display_name"]
                id2label[mid] = display_name
                id2num[mid] = index
                num2label[index] = display_name
            self.id2label, self.index_dict, self.num2label = id2label, id2num, num2label
        else:
            self.id2label, self.index_dict, self.num2label = {}, {}, {}

    def resample_wav(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length, random_start=None):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        if random_start is None:
            for i in range(10):
                random_start = int(random_uniform(0, waveform_length - target_length))
                if torch.max(
                    torch.abs(waveform[:, random_start : random_start + target_length])
                    > 1e-4
                ):
                    break

        return waveform[:, random_start : random_start + target_length], random_start

    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)
        return waveform[start:end]

    
    def process_wavform(self, waveform, sr):
        waveform = self.resample_wav(waveform, sr)
        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform


    def load_audio_with_timeout(self, file_path, timeout):
        """
        Load an audio file with a specified timeout using threading.

        :param file_path: Path to the audio file.
        :param timeout: Maximum time (in seconds) to allow for loading the file.
        :return: (waveform, sample_rate) if successful, None if timeout occurs.
        """
        class AudioLoader(threading.Thread):
            def __init__(self, file_path):
                super().__init__()
                self.file_path = file_path
                self.result = None

            def run(self):
                try:
                    waveform, sample_rate = torchaudio.load(self.file_path)
                    self.result = (waveform, sample_rate)
                except Exception as e:
                    print(f"Failed to load audio: {e}")
                    self.result = None

        # Start the thread
        audio_loader = AudioLoader(file_path)
        audio_loader.start()

        # Wait for the thread to complete or timeout
        audio_loader.join(timeout=timeout)
        if audio_loader.is_alive():
            print(f"Timeout while loading {file_path}")
            return None, None  # Timeout case

        return audio_loader.result


    def read_wav_file(self, filename, random_start=None):
        
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        # waveform = torch.from_numpy(waveform)
        # print("waveform shape", waveform.shape)
        waveform, sr = self.load_audio_with_timeout(filename, timeout=10)
        if waveform is None:
            print("[INFO] timeout when loading the audio")
            # # # TODO Important, dummy audio
            waveform = torch.zeros(1, int(self.sampling_rate * self.duration))
            sr = 16000
        
        # waveform = torch.zeros(1, int(self.sampling_rate * self.duration))
        # sr = 16000
        # waveform, sr = torchaudio.load(filename)
        # # # TODO Important, dummy audio
        # waveform = torch.zeros(1, int(self.sampling_rate * self.duration))

        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration), random_start=random_start
        )
        waveform = self.process_wavform(waveform, sr)

        return waveform, random_start

    def read_mp4_file(self, filename, random_start=None, load_audio=True):
        video = VideoFileClip(filename)
        video = video.resize(newsize=(self.frame_width, self.frame_height))
        audio = video.audio

        # audio part
        waveform = None
        tmp_audio_file = None
        
        if load_audio:
            tmp_audio_file = f"{filename[:-4]}.wav"
            audio.write_audiofile(tmp_audio_file, codec='pcm_s16le', fps=self.sampling_rate, nbytes=2, ffmpeg_params=["-ac", "1"])
            waveform, sr = torchaudio.load(tmp_audio_file, format='wav')

            if not self.keep_audio_files: # keep the audio file and save its path in the json file
                os.remove(tmp_audio_file)
                tmp_audio_file = None
            
            # random segment
            waveform, random_start = self.random_segment_wav(
                waveform, target_length=int(sr * self.duration), random_start=random_start
            )
            random_start_sec = random_start / sr
            waveform = self.process_wavform(waveform, sr)

        else:
            random_start_sec = random_start / self.sampling_rate
        
        # video part    
        frames = []
        if self.load_video:
            interval = 1 / self.video_fps
            current_time = random_start_sec

            # assuming fixed fps
            while current_time <= video.duration and len(frames) < self.target_frame_cnt:
                frame = video.get_frame(current_time)
                frames.append(frame) # T x H x W x C
                current_time += interval

            # transform frames
            frames = torch.from_numpy(np.stack(frames[:self.target_frame_cnt]))
            frames = frames.permute(3, 0, 1, 2).float()  # (C, T, H, W)
            frames = self.video_transform.transform(frames)

        return frames, waveform, random_start, tmp_audio_file

    def read_video_file(self, filename, load_audio=True, random_start=None):
        if os.path.exists(filename):
            frames, waveform, random_start, audio_file = self.read_mp4_file(filename, load_audio=load_audio, random_start=random_start)

            # frames C x T x H x W
            if frames and frames.shape[1] < self.target_frame_cnt:
                extra_frames = torch.zeros((frames.shape[0],  self.target_frame_cnt - frames.shape[1], frames.shape[2], frames.shape[3]))
                frames = torch.cat([frames, extra_frames], dim=1)
        else:
            print(
                '[WARNING, videoaudio_dataset] The path "',
                filename,
                '" is not find in the metadata. Use empty video instead. This is normal in the inference process.',
            )
            target_wavform_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_wavform_length))
            frames = torch.zeros((3, self.target_frame_cnt, self.frame_height, self.frame_width))

            random_start = 0
            audio_file = None

        if load_audio and not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return frames, log_mel_spec, stft, waveform, random_start, audio_file
    
    def read_audio_file(self, filename, random_start=None):
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename, random_start=random_start)
        else:
            print(
                'Non-fatal Warning [dataset.py]: The wav path "',
                filename,
                '" is not find in the metadata. Use empty waveform instead. This is normal in the inference process.',
            )
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0

        if not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return log_mel_spec, stft, waveform, random_start
    
    def get_sample_caption(self, datum, index):
        """
        Use groundtruth caption if exists, otherwise use a hand crafted caption based on the labels
        """
        caption = self.get_data_from_keys(datum, "gt_audio_caption", [])
        if caption:
            return caption
        
        # covert labels to caption
        labels = self.get_data_from_keys(datum, "labels", [])

        if not labels:
            dataset_name = self.datasets_of_datapoints[index]
            absolute_file_path = self._relative_path_to_absolute_path([self.data[index]], dataset_name)[0]
            print(f"Warning file {absolute_file_path} does not have gt caption")
            return ["Sound"]
        
        if not isinstance(labels, list):
            labels = [labels]

        # process each tag
        def clean_tag(tag):
            # Replace common delimiters with spaces
            for delimiter in [';', '_', '-', ',']:
                tag = tag.replace(delimiter, ' ')
            return tag.strip().lower()  

        unique_keywords = set()
        for tag in labels:
            words = clean_tag(tag).split(' ')
            unique_keywords.update(words)  # Add words to the set, which automatically removes duplicates

        cleaned_labels = list(unique_keywords)
        caption = 'The sound of ' + ', '.join(cleaned_labels[:-1]) + (', and ' + cleaned_labels[-1] + '.') if len(cleaned_labels) > 1 else cleaned_labels[0] + '.'
        return [caption]
    
    def get_sample_description(self, datum):
        """
        Use description from metadata if exists, otherwise use a hand crafted description based on the tags
        """
        if 'description' in datum.keys():
            return datum['description']
        
        # covert tags to description
        tags = self.get_data_from_keys(datum, "tags", [])
        if not tags:
            # print("[INFO] could not find tags for:", datum)
            return ""
        
        if not isinstance(tags, list):
            tags = [tags]

        # process each tag
        def clean_tag(tag):
            # Replace common delimiters with spaces
            for delimiter in [';', '_', '-', ',']:
                tag = tag.replace(delimiter, ' ')
            return tag.strip().lower()  

        unique_keywords = set()
        for tag in tags:
            words = clean_tag(tag).split(' ')
            unique_keywords.update(words)  # Add words to the set, which automatically removes duplicates

        cleaned_tags = list(unique_keywords)
        description = ', '.join(cleaned_tags[:-1]) + ', and ' + cleaned_tags[-1] + '.' if len(cleaned_tags) > 1 else cleaned_tags[0] + '.'
    
        return description


    def mel_spectrogram_train(self, y):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft

    # @profile
    # def wav_feature_extraction_torchaudio(self, waveform):
    #     waveform = waveform[0, ...]
    #     waveform = torch.FloatTensor(waveform)

    #     stft = self.stft_transform(waveform)
    #     mel_spec = self.melscale_transform(stft)
    #     log_mel_spec = torch.log(mel_spec + 1e-7)

    #     log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    #     stft = torch.FloatTensor(stft.T)

    #     log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
    #     return log_mel_spec, stft

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec


def custom_collate_fn(batch):
    
    # for test
    # for k in batch[0].keys():
    #     try:
    #         default_collate([{k:item[k]} for item in batch])
    #     except Exception as e:
    #         print("collect error in key", k)
    #         print("files", [b['fname'] for b in batch])
    #         print("shape", [item[k].shape for item in batch])
    #         print("error", e)
        
    collated_batch = default_collate(batch)

    # Handle the 'captions' manually as needed, here assuming you want to keep them as lists of lists
    collated_batch['gt_audio_caption'] = [item['gt_audio_caption'] for item in batch]

    return collated_batch

if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from src.tools.configuration import Configuration
    
    model_config = "settings/simple_runs/genau.yaml"
    config = Configuration(model_config)
    config = config.get_config()

    seed_everything(20)
    
    dataset = VideoAudioDataset(
        config=config, split="train", waveform_only=False,
        load_video=False, sample_single_caption=True, augment_p=1.0)
    
    print("[INFO] Dataset len:", len(dataset))
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True, collate_fn=custom_collate_fn)
    
    # # test augmentation on a single audio 
    # audio_1 = dataset.__getitem__(0, augment=False)
    # aug_audio_1 = dataset.__getitem__(1, augment=True)
    # aug_audio_2 = dataset.__getitem__(2, augment=True)

    # print("orginal_caption:", audio_1['gt_audio_caption'])
    # print("aug_caption_1:", aug_audio_1['gt_audio_caption'])
    # print("aug_caption_2:", aug_audio_2['gt_audio_caption'])

    # # save audio
    # torchaudio.save("original_audio.wav", audio_1['waveform'], 16000)
    # torchaudio.save("aug_audio_1.wav", aug_audio_1['waveform'], 16000)
    # torchaudio.save("aug_audio_2.wav", aug_audio_2['waveform'], 16000)

    for cnt, each in tqdm(enumerate(loader)):
        print("wav shape:", each['waveform'].shape, flush=True)
        print("log_mel_spec shape:", each['log_mel_spec'].shape, flush=True)
        print("names:", each['fname'], flush=True)
        break

    
    # Test from dataset_json
    dataset = VideoAudioDataset(
        config=config, 
        split='test',
        dataset_json= build_dataset_json_from_list("tests/captionlist/inference_submission.lst"),
        load_audio=False,
        load_video=False
    )
    # print("Item 0", dataset[0])