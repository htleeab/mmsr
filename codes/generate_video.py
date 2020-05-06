'''
This script is used for generate video or images sequence from video
please install required pretrained models under experiments/pretrained_models

reference: https://github.com/btahir/deoldify_and_edvr/blob/master/DeOldify_EDVR_Combined.ipynb
modification: Vivian LEE
Possible improvement: frame-by-frame without writing frame to disk, multi-GPU support
'''

import os
import os.path as osp
import glob
import shutil
import re
import gc
from pathlib import Path
import argparse
import subprocess  # for run ffmpeg cmd

from tqdm import tqdm
import cv2
from PIL import Image
import torch
import ffmpeg

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch

workfolder = Path('../video')
inframes_root = workfolder / "inframes"
outframes_s1_root = workfolder / "outframes_s1"
outframes_s2_root = workfolder / "outframes_s2"
result_folder = workfolder / "result"
pretrained_models = Path('../experiments/pretrained_models')

img_format = 'png'
device = torch.device('cuda')
img_vcodec = 'mjpeg' if img_format == 'jpg' else img_format


def clean_mem():
    # torch.cuda.empty_cache()
    gc.collect()


def get_fps(source_path: Path) -> str:
    print(source_path)
    probe = ffmpeg.probe(str(source_path))
    stream_data = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None,
    )
    return stream_data['avg_frame_rate']


def preProcess(imag_path_l, multiple):
    '''Need to resize images for blurred model (needs to be multiples of 4 or 16)'''
    for img_path in imag_path_l:
        im = Image.open(img_path)
        h, w = im.size
        if h % multiple == 0 and w % multiple == 0:
            # same video, all frame have same shape, no need to check remaining frame
            break
        # resize so they are multiples of 4 or 16 (for blurred)
        h = h - h % multiple
        w = w - w % multiple
        im = im.resize((h, w))
        im.save(img_path)


def purge_images(dir):
    for f in os.listdir(dir):
        if re.search(r'.*?\.' + img_format, f):
            os.remove(os.path.join(dir, f))


def get_frame_count(video_filepath):
    vid_in = cv2.VideoCapture(str(video_filepath))
    frame_count = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_in.release()
    return frame_count


def extract_raw_frames(source_path: Path, resolution: tuple):
    inframes_folder = inframes_root / (source_path.stem)
    inframes_folder.mkdir(parents=True, exist_ok=True)
    inframe_path_template = str(inframes_folder / ('%5d.' + img_format))
    if len(os.listdir(inframes_folder)) == get_frame_count(source_path):
        print(f'frame of {source_path} has been extracted already, skip')
    else:
        print(
            f'{inframes_folder}: {len(os.listdir(inframes_folder))} == {get_frame_count(source_path)}'
        )
        purge_images(inframes_folder)
        resolution_str = ':'.join([str(x) for x in resolution])
        subprocess.call(
            f'ffmpeg -y -i {str(source_path)} -s {resolution_str} -f image2'
            f' -pix_fmt rgb24 -c:v {img_vcodec} {inframe_path_template}', shell=True)
    return inframes_folder


def make_subfolders(img_path_l, chunk_size):
    subFolderList = []
    source_img_path = inframes_root / 'video_subfolders'
    source_img_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(img_path_l):
        if i % chunk_size == 0:
            img_path = source_img_path / str(i)
            img_path.mkdir(parents=True, exist_ok=True)
            subFolderList.append(str(img_path))
        img_name = osp.basename(img)
        img_path_name = img_path / img_name
        shutil.copyfile(img, img_path_name)
    return subFolderList


def remove_subfolders():
    shutil.rmtree(inframes_root / 'video_subfolders', ignore_errors=True, onerror=None)


def get_pretrained_model_path(data_mode, stage):
    if data_mode == 'Vid4':
        if stage == 1:
            model_path = pretrained_models / 'EDVR_Vimeo90K_SR_L.pth'
        else:
            raise ValueError('Vid4 does not support stage 2.')
    elif data_mode == 'sharp_bicubic':
        if stage == 1:
            model_path = pretrained_models / 'EDVR_REDS_SR_L.pth'
        else:
            model_path = pretrained_models / 'EDVR_REDS_SR_Stage2.pth'
    elif data_mode == 'blur_bicubic':
        if stage == 1:
            model_path = pretrained_models / 'EDVR_REDS_SRblur_L.pth'
        else:
            model_path = pretrained_models / 'EDVR_REDS_SRblur_Stage2.pth'
    elif data_mode == 'blur':
        if stage == 1:
            model_path = pretrained_models / 'EDVR_REDS_deblur_L.pth'
        else:
            model_path = pretrained_models / 'EDVR_REDS_deblur_Stage2.pth'
    elif data_mode == 'blur_comp':
        if stage == 1:
            model_path = pretrained_models / 'EDVR_REDS_deblurcomp_L.pth'
        else:
            model_path = pretrained_models / 'EDVR_REDS_deblurcomp_Stage2.pth'
    else:
        raise NotImplementedError
    return model_path


def edvrPredict(data_mode, stage, chunk_size, test_dataset_folder, save_folder):
    '''
    data_mode = Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
                Vid4: SR
                REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
                       blur (deblur-clean), blur_comp (deblur-compression).
    stage = 1 or 2, use two stage strategy for REDS dataset.
    chunk_size = number of images within sub-folder, handle when to clean memory
    '''
    # model config
    N_in = 7 if data_mode == 'Vid4' else 5  # use N_in images to restore one HR image
    HR_in = stage == 2 or data_mode in ['blur', 'blur_comp']
    back_RBs = 40 if stage == 1 else 20
    predeblur = 'blur' in data_mode  # True if blur_bicubic | blur | blur_comp
    model_path = get_pretrained_model_path(data_mode, stage)

    # set up the models
    model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    # generate output image
    preProcess_multiple_factor = 16 if predeblur else 4
    padding = 'new_info' if data_mode in ('Vid4',
                                          'sharp_bicubic') else 'replicate'  # temporal padding mode

    util.mkdirs(save_folder)
    remove_subfolders()  # remove old video_subfolder if exists

    img_path_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    util.mkdirs(save_folder)
    purge_images(save_folder)

    preProcess(img_path_l, preProcess_multiple_factor)
    # probably do not have enough memory to load all frames, parse video frames into subfolders for loading frame with read_img_seq function
    # TODO: pass list of image to data_util.read_img_seq instead of using subfolder
    subFolderList = make_subfolders(img_path_l, chunk_size)

    with tqdm(total=len(img_path_l)) as pbar:
        frame_num = 0
        #### read images in chunks, clean memory after each chunk
        for subSubFolder in subFolderList:
            clean_mem()
            imgs_LQ = data_util.read_img_seq(subSubFolder)
            subSubFolder_l = sorted(glob.glob(osp.join(subSubFolder, '*')))
            max_idx = len(subSubFolder_l)

            # process each image
            for img_idx, img_path in enumerate(subSubFolder_l):
                with open('offset_mean.log', 'a') as file:
                    file.write(f'[generate_video.py] forwarding frame {frame_num}\n')

                img_name = osp.splitext(osp.basename(img_path))[0]
                select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
                imgs_in = imgs_LQ.index_select(0,
                                               torch.LongTensor(select_idx)).unsqueeze(0).to(device)

                output = util.single_forward(model, imgs_in)
                output = util.tensor2img(output.squeeze(0))

                cv2.imwrite(osp.join(save_folder, '{}.{}'.format(img_name, img_format)), output)
                pbar.update()
                frame_num += 1

    return save_folder


def encode_video(source_path: Path, outframes_folder: Path, result_path: Path) -> Path:
    outframes_path_template = str(outframes_folder / ('%5d.' + img_format))
    result_folder.mkdir(parents=True, exist_ok=True)
    fps = get_fps(source_path)
    print('Original FPS is: ', fps)

    if result_path.exists():
        result_path.unlink()

    # try to extract audio
    audio_file = result_folder / (source_path.stem + '.aac')
    if audio_file.exists():
        audio_file.unlink()
    subprocess.call(f'ffmpeg -y -i "{str(source_path)}" -vn -c:a copy "{str(audio_file)}"',
                    shell=True)

    # encode output
    if audio_file.exists and os.path.exists(audio_file):
        # combine images and audio
        subprocess.call(
            f'ffmpeg -y -f image2 -r {fps} -c:v {img_vcodec} -i "{outframes_path_template}"'
            f' -i "{str(audio_file)}" -c:a copy '
            f' -c:v libx264 -b:v 3.2M "{str(result_path)}"', shell=True)
        audio_file.unlink()
    else:
        subprocess.call(
            f'ffmpeg -y -f image2 -r {fps} -c:v {img_vcodec} -i "{outframes_path_template}"'
            f' -c:v libx264 -b:v 3.2M "{str(result_path)}"', shell=True)

    return result_path


def encode_images(outframes_folder, output_filepath, fps):
    outframes_path_template = str(outframes_folder / ('%5d.' + img_format))
    result_folder = Path(os.path.dirname(output_filepath))
    result_folder.mkdir(parents=True, exist_ok=True)
    subprocess.call(
        f'ffmpeg -y -f image2 -r {fps} -c:v {img_vcodec} -i "{outframes_path_template}"'
        f' -c:v libx264 -vrf 17 "{str(output_filepath)}"', shell=True)
    return output_filepath


def edvr_img2video(img_src_dir: Path, data_mode: str, chunk_size: int, finetune_stage2: bool,
                   clean_frames: bool):
    sub_folder_name = f'{img_src_dir.stem}_{data_mode}'

    # process frames
    outframes = edvrPredict(data_mode, 1, chunk_size, img_src_dir,
                            outframes_s1_root / sub_folder_name)

    # fine-tune stage 2
    if finetune_stage2:
        print(f'fine-tune with stage 2 model')
        outframes = edvrPredict(data_mode, 2, chunk_size, outframes,
                                outframes_s2_root / sub_folder_name)

    # Encode video from predicted frames
    output_video_path = result_folder / f'{img_src_dir.name}_{data_mode}.mp4'
    encode_images(outframes, output_video_path, 25)

    print(f'Video output: {output_video_path}')

    if clean_frames:
        shutil.rmtree(inframes_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s1_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s2_root, ignore_errors=True, onerror=None)


def edvr_video(video_src_path: Path, data_mode: str, chunk_size: int, finetune_stage2: bool,
               clean_frames: bool, resolution: tuple):
    sub_folder_name = f'{video_src_path.stem}_{data_mode}'

    # extract frames
    inframes_folder = extract_raw_frames(video_src_path, resolution)

    # process frames
    outframes = edvrPredict(data_mode, 1, chunk_size, inframes_folder,
                            outframes_s1_root / sub_folder_name)

    # fine-tune stage 2
    if finetune_stage2:
        print(f'fine-tune with stage 2 model')
        outframes = edvrPredict(data_mode, 2, chunk_size, outframes,
                                outframes_s2_root / sub_folder_name)

    # Encode video from predicted frames
    output_video_path = result_folder / f'{video_src_path.stem}_{data_mode}.mp4'
    encode_video(video_src_path, outframes, output_video_path)

    print(f'Video output: {output_video_path}')

    if clean_frames:
        shutil.rmtree(inframes_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s1_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s2_root, ignore_errors=True, onerror=None)


def str2bool(v):
    # ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2tuple(v):
    if isinstance(v, tuple):
        return v
    try:
        return tuple(v.split(','))
    except:
        raise argparse.ArgumentTypeError(f'cannot turn {v} into tuple')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', default='../video/input.mp4',
                        help='input video file or folder of image sequences')
    parser.add_argument('-m', '--model', dest='model', type=str, default='blur_comp',
                        choices=['Vid4', 'sharp_bicubic', 'blur_bicubic', 'blur', 'blur_comp'])
    parser.add_argument('-s', '--two-stage', dest='two_stage_enabled', type=str2bool, default=True)
    parser.add_argument('-c', '--clean', dest='clean_frames', type=str2bool, default=True)
    parser.add_argument('-r', '--resolution', dest='resolution', type=str2tuple,
                        default=(1280, 720))
    args = parser.parse_args()
    if args.model == 'Vid4':
        assert not args.two_stage_enabled, f'Vid4 do not have stage 2 pretrained model'
    return args


if __name__ == '__main__':
    if os.path.exists('offset_mean.log'):
        os.remove('offset_mean.log')
    args = parse_args()
    if not os.path.isdir(args.input):
        edvr_video(Path(args.input), args.model, 100, args.two_stage_enabled, args.clean_frames,
                   args.resolution)
    else:
        edvr_img2vid(args.input, args.model, 100, args.two_stage_enabled, args.clean_frames)
