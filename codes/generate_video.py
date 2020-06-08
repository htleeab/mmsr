'''
Generate video or images sequence from video
Please download required pretrained models under experiments/pretrained_models

reference: https://github.com/btahir/deoldify_and_edvr/blob/master/DeOldify_EDVR_Combined.ipynb
modification: Vivian LEE
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
import numpy as np
import cv2
from PIL import Image
import torch
import ffmpeg

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
from scripts.diff_score import get_score_func

workfolder = Path('../video')
inframes_root = workfolder / "inframes"
outframes_s1_root = workfolder / "outframes_s1"
outframes_s2_root = workfolder / "outframes_s2"
fix_patch_root = workfolder / "fix_patch"
result_folder = workfolder / "result"
side_by_side_root = workfolder / "side_by_side"

PATCH_ARTIFACT_ENABLED = False  # If true, replace the patch with source when the ssim > threshold

device = torch.device('cuda')
img_format = 'png'
img_vcodec = 'mjpeg' if img_format == 'jpg' else img_format

offset_log_filepath = 'offset_mean.log'

torch.backends.cudnn.benchmark = True


def clean_mem():
    # torch.cuda.empty_cache()
    gc.collect()


def get_fps(source_path: Path) -> str:
    probe = ffmpeg.probe(str(source_path))
    stream_data = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None,
    )
    return stream_data['avg_frame_rate']


def preProcess(images_path, multiple):
    '''Need to resize images for blurred model (needs to be multiples of 4 or 16)'''
    ## Check shape of 1 image
    h, w, _ = cv2.imread(images_path[0]).shape
    assert cv2.imread(images_path[0]).shape == cv2.imread(
        images_path[-1]
    ).shape, f'first frame and last frame have different shape, please clean previous frame and re-extract it'
    if not (h % multiple == 0 and w % multiple == 0):
        h_padding = (multiple - h % multiple) % multiple
        w_padding = (multiple - w % multiple) % multiple
        print(f'resolution {h}x{w} is not multiple of {multiple}, pad {h_padding, w_padding}')
        for img_path in tqdm(images_path, desc='pre-Processing'):
            img = cv2.imread(img_path)
            padded_img = np.pad(img, [(0, h_padding), (0, w_padding), (0, 0)])
            cv2.imwrite(img_path, padded_img)


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
    # ffmpeg might generate an extra frame. might be de-interlaced?
    if abs(len(os.listdir(inframes_folder)) - get_frame_count(source_path)) <= 1:
        print(f'frame of {source_path} has been extracted already, skip')
    else:
        purge_images(inframes_folder)
        resolution_opt = f'-s {resolution[0]}:{resolution[1]}' if resolution[0] is not None else ''
        subprocess.check_call(
            f'ffmpeg -y -i "{str(source_path)}" {resolution_opt} -f image2'
            f' -pix_fmt rgb24 -c:v {img_vcodec} "{inframe_path_template}"', shell=True)
        print(f'extracted frames: {len(os.listdir(inframes_folder))}')
    return inframes_folder


def process_side_by_side_img_sequence(folder1, folder2, side_by_side_folder):
    side_by_side_folder.mkdir(parents=True, exist_ok=True)
    for img_name in sorted(os.listdir(folder1)):
        img1_path = os.path.join(folder1, img_name)
        img2_path = os.path.join(folder2, img_name)
        out_path = os.path.join(side_by_side_folder, img_name)
        assert os.path.exists(img1_path), f'{img1_path} not exists'
        assert os.path.exists(img2_path), f'{img2_path} not exists'
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        assert img1 is not None, f'cannot read {img1_path}'
        assert img2 is not None, f'cannot read {img2_path}'
        packed_img = np.concatenate([img1, img2], axis=1)
        cv2.imwrite(out_path, packed_img)
    return side_by_side_folder


def get_pretrained_model_path(data_mode, stage, pretrained_model_dir):
    pretrained_models = Path(pretrained_model_dir)
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
    elif data_mode == 'blur_comp_M':
        if stage == 1:
            model_path = pretrained_models / 'EDVR_REDS_deblurcomp_M.pth'
        else:
            raise NotImplementedError(f'{data_mode} stage {stage} is not implemented')
    elif data_mode in ['blur_comp', 'blur_comp_L']:
        if stage == 1:
            model_path = pretrained_models / 'EDVR_REDS_deblurcomp_L.pth'
        else:
            model_path = pretrained_models / 'EDVR_REDS_deblurcomp_Stage2.pth'
    else:
        raise NotImplementedError(f'{data_mode} is not implemented')
    return model_path


def edvrPredict(data_mode, stage, chunk_size, test_dataset_folder, save_folder, model_dir):
    '''
    data_mode = Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
                Vid4: SR
                REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
                       blur (deblur-clean), blur_comp (deblur-compression).
    stage = 1 or 2, use two stage strategy for REDS dataset.
    chunk_size = number of images within sub-folder, handle when to clean memory
    '''
    ## model config
    N_in = 7 if data_mode == 'Vid4' else 5  # use N_in images to restore one HR image
    HR_in = stage == 2 or data_mode in ['blur', 'blur_comp']
    back_RBs = 40 if stage == 1 else 20
    predeblur = 'blur' in data_mode  # True if blur_bicubic | blur | blur_comp
    model_path = get_pretrained_model_path(data_mode, stage, model_dir)

    ## set up the models
    model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    ## data pre-processing setup
    util.mkdirs(save_folder)
    purge_images(save_folder)

    img_path_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))

    preProcess_multiple_factor = 16 if predeblur else 4
    preProcess(img_path_l, preProcess_multiple_factor)
    padding = 'new_info' if data_mode in ('Vid4', 'sharp_bicubic') else 'replicate'

    ## Feed image sequence into model, predict output
    with tqdm(total=len(img_path_l)) as pbar:
        frame_num = 0
        # probably do not have enough memory to load all frames
        # parse video frames by chunk size for loading frames with read_img_seq function
        parsed_img_list = []
        for i in range(0, len(img_path_l), chunk_size):
            parsed_img_list.append(img_path_l[i:i + chunk_size])
        # fixme: the frames near parse point is not continues, might need to overlap those frame and adjust select idx for completely continuous inference
        for images_chunk in parsed_img_list:
            clean_mem()
            imgs_LQ = data_util.read_img_seq(images_chunk)
            max_idx = len(images_chunk)

            # process each image
            for img_idx, img_path in enumerate(images_chunk):
                img_name = osp.splitext(osp.basename(img_path))[0]
                select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
                # Take reference image with n neighbor frames
                imgs_in = imgs_LQ.index_select(0,
                                               torch.LongTensor(select_idx)).unsqueeze(0).to(device)
                output = util.single_forward(model, imgs_in)
                output = util.tensor2img(output.squeeze(0))

                cv2.imwrite(osp.join(save_folder, '{}.{}'.format(img_name, img_format)), output)
                pbar.update()
                frame_num += 1

    return save_folder


def contain_audio(video_filepath):
    result = subprocess.run(
        f'ffmpeg -i "{video_filepath}" -vn -f null - 2>&1 | grep does\ not\ contain\ any\ stream',
        shell=True, stdout=subprocess.PIPE)
    return len(result.stdout) == 0


def encode_video(source_path: Path, outframes_folder: Path, result_path: Path) -> Path:
    outframes_path_template = str(outframes_folder / ('%5d.' + img_format))
    result_folder.mkdir(parents=True, exist_ok=True)
    fps = get_fps(source_path)
    print('Original FPS is: ', fps)

    if result_path.exists():
        result_path.unlink()

    # encode output
    if contain_audio(source_path):
        subprocess.check_call(
            f'ffmpeg -y -f image2 -r {fps} -c:v {img_vcodec} -i "{outframes_path_template}"'
            f' -vn -i "{str(source_path)}" '
            f' -c:v libx264 -crf 10 "{str(result_path)}"', shell=True)
    else:
        subprocess.check_call(
            f'ffmpeg -y -f image2 -r {fps} -c:v {img_vcodec} -i "{outframes_path_template}"'
            f' -c:v libx264 -crf 10 "{str(result_path)}"', shell=True)

    return result_path


def encode_images(outframes_folder, output_filepath, fps):
    outframes_path_template = str(outframes_folder / ('%5d.' + img_format))
    result_folder = Path(os.path.dirname(output_filepath))
    result_folder.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        f'ffmpeg -y -f image2 -r {fps} -c:v {img_vcodec} -i "{outframes_path_template}"'
        f' -c:v libx264 "{str(output_filepath)}"', shell=True)
    return output_filepath


def edvr_img2video(img_src_dir: Path, data_mode: str, chunk_size: int, finetune_stage2: bool,
                   clean_frames: bool, model_dir_path: str):
    sub_folder_name = f'{img_src_dir.stem}_{data_mode}'

    # process frames
    outframes = edvrPredict(data_mode, 1, chunk_size, img_src_dir,
                            outframes_s1_root / sub_folder_name, model_dir_path)

    # fine-tune stage 2
    if finetune_stage2:
        print(f'fine-tune with stage 2 model')
        outframes = edvrPredict(data_mode, 2, chunk_size, outframes,
                                outframes_s2_root / sub_folder_name, model_dir_path)

    # Encode video from predicted frames
    output_video_path = result_folder / f'{img_src_dir.name}_{data_mode}.mp4'
    encode_images(outframes, output_video_path, 25)

    print(f'Video output: {output_video_path}')

    if clean_frames:
        shutil.rmtree(inframes_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s1_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s2_root, ignore_errors=True, onerror=None)


def patch_artifact(inframe_folder, edvr_frame_folder, output_folder, patch_parse=16, threshold=0.8):
    os.makedirs(output_folder, exist_ok=True)
    score_func = get_score_func('ssim')
    for frame_filename in tqdm(sorted(os.listdir(edvr_frame_folder)), desc='patch artifact'):
        edvr_frame = cv2.imread(os.path.join(edvr_frame_folder, frame_filename))
        src_frame = cv2.imread(os.path.join(inframe_folder, frame_filename))
        assert edvr_frame is not None, f'cannot read {os.path.join(edvr_frame_folder, frame_filename)}'
        assert src_frame is not None, f'cannot read {os.path.join(inframe_folder, frame_filename)}'
        resolution = edvr_frame.shape
        final_frame = np.zeros(edvr_frame.shape)
        patch_size = [int(x / patch_parse) for x in resolution[:2]]
        for row_idx in np.arange(0, resolution[0], patch_size[0], dtype=int):
            for col_idx in np.arange(0, resolution[1], patch_size[1], dtype=int):
                # [row_idx:row_idx+patch_size[0], col_idx:col_idx+patch_size[1]]
                patch_indices = tuple([
                    slice(row_idx, row_idx + patch_size[0]),
                    slice(col_idx, col_idx + patch_size[1])
                ])
                src_patch = src_frame[patch_indices]
                edvr_patch = edvr_frame[patch_indices]
                score = score_func(src_patch, edvr_patch)
                if score > threshold:
                    final_frame[patch_indices] = edvr_patch
                else:
                    final_frame[patch_indices] = src_patch
        cv2.imwrite(os.path.join(output_folder, frame_filename), final_frame)
    return output_folder


def edvr_video(video_src_path: Path, data_mode: str, chunk_size: int, finetune_stage2: bool,
               clean_frames: bool, resolution: tuple, model_dir_path: str):
    sub_folder_name = f'{video_src_path.stem}_{data_mode}'

    # extract frames
    inframes_folder = extract_raw_frames(video_src_path, resolution)

    # process frames
    outframes = outframes_s1_root / sub_folder_name
    outframes = edvrPredict(data_mode, 1, chunk_size, inframes_folder,
                            outframes_s1_root / sub_folder_name, model_dir_path)

    # fine-tune stage 2
    if finetune_stage2:
        print(f'fine-tune with stage 2 model')
        outframes = edvrPredict(data_mode, 2, chunk_size, outframes,
                                outframes_s2_root / sub_folder_name, model_dir_path)

    # patch artifact
    if PATCH_ARTIFACT_ENABLED:
        for patch_parse in [16, 32]:
            for threshold in [0.8, 0.9]:
                outframes = patch_artifact(inframes_folder, outframes,
                                           fix_patch_root / sub_folder_name, patch_parse, threshold)
                output_video_path = result_folder / f'{video_src_path.stem}_{data_mode}_{patch_parse}_{threshold}.mp4'
                encode_video(video_src_path, outframes, output_video_path)
    else:
        # Encode video from predicted frames
        output_video_path = result_folder / f'{video_src_path.stem}_{data_mode}.mp4'
        encode_video(video_src_path, outframes, output_video_path)

        # Generate side-by-side version
        side_by_side_folder = process_side_by_side_img_sequence(inframes_folder, outframes, side_by_side_root / sub_folder_name)
        output_video_path = result_folder / f'{video_src_path.stem}_{data_mode}_side_by_side.mp4'
        encode_video(video_src_path, side_by_side_folder, output_video_path)
    print(f'Video output: {output_video_path}')

    if clean_frames:
        shutil.rmtree(inframes_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s1_root, ignore_errors=True, onerror=None)
        shutil.rmtree(outframes_s2_root, ignore_errors=True, onerror=None)
        shutil.rmtree(side_by_side_root, ignore_errors=True, onerror=None)


def pack_img(sub_imgs, row=2):
    img_mid = int(len(sub_imgs) / row)
    packed_img = np.concatenate(
        [np.concatenate(sub_imgs[:img_mid], axis=0),
         np.concatenate(sub_imgs[img_mid:], axis=0)], axis=1)
    return packed_img


def encode_offset_layers_within_1_video(offset_folder, video_out_filepath, fps):
    offset_layers = sorted(os.listdir(offset_folder))
    print(offset_layers)
    imgs_shape = np.array([
        cv2.imread(x).shape[:-1] for x in glob.glob(os.path.join(offset_folder, '*', '00000.png'))
    ])
    largest_offset_resolution = np.max(imgs_shape, axis=0)[::-1]
    video_out = cv2.VideoWriter(video_out_filepath, cv2.VideoWriter_fourcc(*'MJPG'), fps,
                                tuple(largest_offset_resolution * 2))
    for frame_idx in tqdm(
            range(0, len(glob.glob(os.path.join(offset_folder, offset_layers[0], '*.png')))),
            desc='encode offset'):
        frame_filename = f'{frame_idx:05d}.png'
        offset_frames = [
            cv2.resize(cv2.imread(os.path.join(offset_folder, offset_layer, frame_filename)),
                       tuple(largest_offset_resolution)) for offset_layer in offset_layers
        ]
        cur_frame = pack_img(offset_frames)
        video_out.write(cur_frame)
    video_out.release()


def encoder_offset_layers_per_group(offset_folder, video_out_filepath, fps, deformable_groups=8):
    offset_layers = sorted(os.listdir(offset_folder))
    imgs_shape = np.array([
        cv2.imread(x).shape[:-1] for x in glob.glob(os.path.join(offset_folder, '*', '00000.png'))
    ])
    largest_offset_resolution = np.max(imgs_shape, axis=0)[::-1]
    final_output_resolution = tuple(
        [largest_offset_resolution[0] * 8, largest_offset_resolution[1] * 4])
    video_out = cv2.VideoWriter(video_out_filepath, cv2.VideoWriter_fourcc(*'MJPG'), fps,
                                final_output_resolution)
    for frame_idx in tqdm(
            range(0, len(glob.glob(os.path.join(offset_folder, offset_layers[0], '*.png'))))):
        frame_filename = f'{frame_idx:05d}.png'
        offset_layer_imgs = []
        for offset_layer in offset_layers:
            layer_offset_all_group = [
                cv2.resize(
                    cv2.imread(
                        os.path.join(offset_folder, offset_layer, str(group_idx), frame_filename)),
                    tuple(largest_offset_resolution)) for group_idx in range(0, deformable_groups)
            ]
            layer_offset_all_group_frame = np.concatenate(layer_offset_all_group, axis=1)
            offset_layer_imgs.append(layer_offset_all_group_frame)
        cur_frame = np.concatenate(offset_layer_imgs, axis=0)
        video_out.write(cur_frame)
    video_out.release()


def handle_offset(input_file, args):
    video_name = os.path.splitext(os.path.basename(input_file))[0]
    offset_img_folder = '../video/offset'
    if os.path.exists(offset_img_folder):
        offset_folder_moved = offset_img_folder + f'_{video_name}_{args.model}'
        while os.path.exists(offset_folder_moved):
            offset_folder_moved += '_'
        os.rename(offset_img_folder, offset_folder_moved)
    else:
        offset_folder_moved = offset_img_folder + f'_{video_name}_{args.model}'

    if os.path.exists(offset_folder_moved):
        fps = 25 * 5  # because PCD run 5 times per frame
        encode_offset_layers_within_1_video(
            offset_folder_moved, f'../video/offset_videos/offset_{video_name}_{args.model}_all.mp4',
            fps)
        encoder_offset_layers_per_group(
            offset_folder_moved,
            f'../video/offset_videos/offset_{video_name}_{args.model}_groups.mp4', fps)
    else:
        print(f'no offset generated')


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
    parser.add_argument('-i', '--input', dest='input', nargs='+', default=['../video/input.mp4'],
                        help='input video file or folder of image sequences')
    parser.add_argument('-m', '--model', dest='model', type=str, default='blur_comp',
                        choices=['Vid4', 'sharp_bicubic', 'blur_bicubic', 'blur', 'blur_comp'])
    parser.add_argument('-w', '--model_dir', dest='model_dir', type=str,
                        default='../experiments/pretrained_models',
                        help='directory of pretrained model')
    parser.add_argument('-s', '--two-stage', dest='two_stage_enabled', type=str2bool, default=True)
    parser.add_argument('-c', '--clean', dest='clean_frames', type=str2bool, default=True)
    parser.add_argument('-r', '--resolution', dest='resolution', type=str2tuple,
                        default=(None, None))
    args = parser.parse_args()
    if args.model == 'Vid4':
        assert not args.two_stage_enabled, f'Vid4 do not have stage 2 pretrained model'
    return args


if __name__ == '__main__':
    if os.path.exists(offset_log_filepath):
        os.remove(offset_log_filepath)
    args = parse_args()
    for input_file in args.input:
        assert os.path.exists(input_file), f'{input_file} not exists'
        if not os.path.isdir(input_file):
            edvr_video(Path(input_file), args.model, 100, args.two_stage_enabled, args.clean_frames,
                       args.resolution, args.model_dir)
        else:
            edvr_img2vid(input_file, args.model, 100, args.two_stage_enabled, args.clean_frames,
                         args.model_dir)
        if os.path.exists(offset_log_filepath):
            video_name = os.path.splitext(os.path.basename(input_file))[0]
            os.rename(offset_log_filepath, f'offset_log/offset_{video_name}_{args.model}.log')
        handle_offset(input_file, args)
