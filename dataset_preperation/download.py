import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool, get_context
import yt_dlp
import logging
from io import StringIO
import json
import argparse
from functools import partial
from download_manager import get_dataset_json_file, dataset_urls

def download_yt_video(entry,
                    save_dir,
                    yt_cookie_path=None,
                    audio_only=False,
                    proxy=None,
                    audio_sampling_rate=44100,
                    resume=True,
                    files_per_folder=5000):
    
    video_idx = entry[0]
    video_id, intervals = entry[1][0], entry[1][1]['intervals']
    
    for file_idx, video_info in enumerate(intervals):
        start = video_info['start']
        to = video_info['end']
        autocap_caption = video_info.get('text', None)
        subfolder_idx = f'{video_idx // files_per_folder:06}'
        st = f'{int(start//3600)}:{int(start//60)-60*int(start//3600)}:{start%60}'
        dur = f'{int(to//3600)}:{int(to//60)-60*int(to//3600)}:{to%60}'
        
        outpath = os.path.join(save_dir, subfolder_idx)
        os.makedirs(outpath, exist_ok=True)
        
        if resume and os.path.isfile(os.path.join(outpath, f'{video_id}_{file_idx:03d}.json')):
            continue
        else:
            ytdl_logger = logging.getLogger()
            log_stream = StringIO()    
            logging.basicConfig(stream=log_stream, level=logging.INFO)
            
            out_file_ext = 'wav' if audio_only else 'mp4'
            format = 'bestaudio/best' if audio_only else 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'
            ydl_opts = {
                "logger": ytdl_logger,
                'outtmpl': f"/temps/id_{video_id}_{file_idx:03d}/audio.%(ext)s",
                'format': format,
                'quiet': True,
                'ignoreerrors': False, 
                # 'write_thumbnail': True,
                'writeinfojson': True,  # This will write a separate .info.json with detailed info
                # 'writesubtitles': True,  # Attempt to download subtitles (transcripts)
                # 'writeautomaticsub': True,  # Attempt to download automatic subtitles (auto-generated transcripts)
                'force_generic_extractor': True,
                'postprocessor_args': ['-ar', str(audio_sampling_rate)],
                'external_downloader':'ffmpeg',
                'download_ranges': yt_dlp.utils.download_range_func([], [[start, to]]),
                'force-keyframe-at-cuts': True,
                'external_downloader_args':['-loglevel', 'quiet'],
            }
            if yt_cookie_path is not None:
                ydl_opts['cookiefile'] = f'/temps/id_{video_id}_{file_idx:03d}/cookies.txt'
            if audio_only:
                ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio',
                                               'preferredcodec': 'wav'}]
            else:
                ydl_opts['postprocessors'] = [{'key': 'FFmpegVideoConvertor',
                                                'preferedformat': 'mp4',  # Ensure the output is MP4
                                                }]
            if proxy is not None:
                ydl_opts['proxy'] = f'socks5://127.0.0.1:{proxy}/'
            
            url = f'https://www.youtube.com/watch?v={video_id}'
            os.makedirs(f'/temps/id_{video_id}_{file_idx:03d}', exist_ok=True)
            if yt_cookie_path is not None:
                shutil.copy(yt_cookie_path, f'/temps/id_{video_id}_{file_idx:03d}/cookies.txt')
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    file_exist = os.path.isfile(os.path.join(outpath, f'{video_id}_{file_idx:03d}.{out_file_ext}'))
                    info=ydl.extract_info(url, download=not file_exist)
                    filename = f'{video_id}_{file_idx:03d}.{out_file_ext}'
                    jsonname = f'{video_id}_{file_idx:03d}.json'
                    if not file_exist:
                        shutil.move(os.path.join(f'/temps/id_{video_id}_{file_idx:03d}',f'audio.{out_file_ext}'), os.path.join(outpath, filename))
                    else:
                        pass
                    file_meta = {'id':f'{video_id}','path': os.path.join(outpath, filename),'title': info['title'], 'url':url, 'start': start, 'end': to}

                    if autocap_caption is not None:
                        file_meta['autocap_caption'] = autocap_caption
                        
                    # meta data 
                    file_meta['resolution'] = info.get('resolution')
                    file_meta['fps'] = info.get('fps')
                    file_meta['aspect_ratio'] = info.get('aspect_ratio')
                    file_meta['audio_channels'] = info.get('audio_channels')

                    file_meta['description'] = info.get('description')
                    file_meta['uploader'] = info.get('uploader')
                    file_meta['upload_date'] = info.get('upload_date')
                    file_meta['duration'] = info.get('duration')
                    file_meta['view_count'] = info.get('view_count')
                    file_meta['like_count'] = info.get('like_count')
                    file_meta['channel_follower_count'] = info.get('channel_follower_count')
                    file_meta['dislike_count'] = info.get('dislike_count')
                    file_meta['channel_id'] = info.get('channel_id')
                    file_meta['channel_url'] = info.get('channel_url')
                    file_meta['channel_name'] = info.get('uploader')

                    print("[INFO] save meta data for", os.path.join(outpath, jsonname))
                    json.dump(file_meta, open(os.path.join(outpath, jsonname),'w'))
                os.system(f'rm -rf /temps/id_{video_id}_{file_idx:03d}')
            except Exception as e:
                os.system(f'rm -rf /temps/id_{video_id}_{file_idx:03d}')
                print(f"[ERROR] downloading {os.path.join(outpath, f'{video_id}_{file_idx:03d}.json')}:", e)
                return f'{url} - ytdl : {log_stream.getvalue()}, system : {str(e)}'
    return None

def update_interval_dict(dict_1, dict_2):
    """
    combine two dictionaries, and merge intervals list if it is replicated
    """
    for k, v in dict_2.items():
        if k in dict_1: 
            dict_2[k]['intervals'] += dict_1[k]['intervals']
    
    dict_1.update(dict_2)
        
def read_video_segments_info(local_input_video_segments, 
                             start_idx=0,
                             end_idx=int(1e9)):
    all_video_segments = {}
    with open(local_input_video_segments, 'r') as f:
        last_idx = 0
        for idx, json_str in enumerate(tqdm(f, desc="parsing json input")): 
            if idx > start_idx:
                try:
                    if json_str.endswith('\n'):
                        json_str = json_str[:-1]
                    if json_str.endswith(','):
                        json_str = json_str[:-1]
                    json_object = json.loads(json_str)
                    update_interval_dict(all_video_segments, json_object)
                except Exception as e:
                    print("[ERROR] Couldn't parse json string:", json_str)
                    continue
                last_idx += 1
            
            if last_idx >= end_idx:
                break
    
    return all_video_segments

def download_audioset_split(json_file,
                            save_dir,
                            yt_cookie_path,
                            audio_only=False,
                            proxy_port=None,
                            audio_sampling_rate=44100,
                            start_idx=0,
                            end_idx=int(1e9),
                            num_processes=os.cpu_count(),
                            resume=True,
                            files_per_folder=5000
                            ):
    
    os.makedirs(save_dir, exist_ok=True)
        
    all_video_segments = read_video_segments_info(json_file,
                                                  start_idx=start_idx,
                                                  end_idx=end_idx)
    
    download_audio_split = partial(download_yt_video,
                                   save_dir=save_dir,
                                   yt_cookie_path=yt_cookie_path,
                                   audio_only=audio_only,
                                   proxy=proxy_port,
                                   audio_sampling_rate=audio_sampling_rate,
                                   resume=resume,
                                   files_per_folder=files_per_folder) 
    
    logs = []
    p = get_context("spawn").Pool(num_processes*2)
    
    # download_audio_split = partial(save_metadata, split=split) # save_metadata
    with tqdm(total=len(all_video_segments),leave=False) as pbar:
        for log in p.imap_unordered(download_audio_split, enumerate(all_video_segments.items(), start=start_idx)):
            logs.append(log)
            pbar.update()
    p.close()
    p.join()
    logs = [l for l in logs if l is not None]
    open(f'download_logs.txt','w').write('\n'.join(logs))
    
if __name__ == "__main__":
    try:
        shutil.rmtree('/temps')
    except FileNotFoundError:
        pass
    os.makedirs('/temps', exist_ok=True)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", 
                        type=str,
                        required=True,
                        help=f"Provided the dataset names. Available datasets are {dataset_urls.keys()}")
    
    parser.add_argument("--input_file", 
                        type=str,
                        default=None,
                        required=False,
                        help="Provided the path to the json object that contains the dataset information. You may leave it empty to attempt to download the required files from the web")
    
    parser.add_argument("--save_dir", 
                        type=str,
                        required=False,
                        default='data/datasets/autocap/videos',
                        help="where to save the downloaded files")
    
    parser.add_argument("--audio_only", 
                        required=False,
                        action='store_true',
                        help="Enable to only save the wav files and discard the vidoes")
    
    parser.add_argument("--cookie_path", 
                        type=str,
                        required=False,
                        default=None,
                        help="Path to your Youtube cookies files")
    
    parser.add_argument("--sampling_rate", 
                        type=int,
                        default=44100,
                        help="Audio sampling rate, default is set to 44.1KHz")
    
    parser.add_argument("--proxy", 
                        type=str,
                        default=None,
                        help="provde a proxy port to bypass youtube blocking your IP")
    
    parser.add_argument("--files_per_folder", 
                        type=int,
                        default=50000,
                        help="How many files to store per folder")
    
    parser.add_argument('--start_idx', '-s', 
                        type=int, default=0,
                        help="start index of the json objects in the provided files")
    
    parser.add_argument('--end_idx', '-e', type=int, default=int(1e9),
                        help="start index of the json objects in the provided files")
    
    parser.add_argument('--redownload', action='store_true',
                        help="redownload already downloaded files")
    
    args = parser.parse_args()
    
    if args.input_file is None or not os.path.exists(args.input_file):
        args.input_file = get_dataset_json_file(args.dataset_name, args.input_file, download=True)

    download_audioset_split(json_file=args.input_file,
                            save_dir=args.save_dir,
                            audio_only=args.audio_only,
                            audio_sampling_rate=args.sampling_rate,
                            yt_cookie_path=args.cookie_path,
                            proxy_port=args.proxy,
                            start_idx=args.start_idx,
                            end_idx=args.end_idx,
                            resume=not args.redownload,
                            files_per_folder=args.files_per_folder)
