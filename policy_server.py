"""
Policy inference server for real robot deployment.

Run this in forcelens_dp repo:
    python policy_server.py --ckpt-path outputs/.../checkpoints/xxx.ckpt

On the robot-controller side, use RemotePolicy with:
    image_width=320, image_height=240
"""

import argparse
import atexit
import contextlib
import math
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from collections import deque
from pathlib import Path

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq
from PIL import Image

from diffusion_policy.common.pytorch_util import dict_apply

CONTROL_PERIOD = 0.1        # 10 Hz
LATENCY_BUDGET = 0.2        # 200 ms
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / CONTROL_PERIOD)  # 2

IMAGE_H = 240
IMAGE_W = 320

# Same gripper-colour prior used by ViewForce's offline mask bootstrapping.
GRIPPER_H_MIN = 130.0
GRIPPER_H_MAX = 185.0
GRIPPER_S_MIN = 0.30
GRIPPER_V_MIN = 0.20
GRIPPER_MIN_BLOB_AREA = 300


def _load_steering_classes(viewforce_root):
    if viewforce_root is not None:
        root = Path(viewforce_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f'ViewForce root does not exist: {root}')
        sys.path.insert(0, str(root))

    try:
        from src.steering import ForceSteeringConfig, ViewForceSteeringPipeline
    except ImportError as exc:
        raise ImportError(
            'Could not import ViewForce steering. Run from the ViewForce repo, '
            'or pass --tts-viewforce-root /path/to/ViewForce.'
        ) from exc
    return ForceSteeringConfig, ViewForceSteeringPipeline


class Sam2FrameMasker:
    """Run the same prompt+SAM2 mask path used by ViewForce offline rollouts."""

    def __init__(self, viewforce_root, model_key='small', sam2_repo=None, sam2_ckpt=None):
        root = Path(viewforce_root).expanduser().resolve()
        sys.path.insert(0, str(root))
        import segment_gripper

        sam2_repo_path = (
            Path(sam2_repo).expanduser().resolve()
            if sam2_repo is not None
            else root / 'third_party' / 'sam2'
        )
        if sam2_repo_path.exists():
            segment_gripper.SAM2_REPO = str(sam2_repo_path)
        if sam2_ckpt is not None:
            segment_gripper.LOCAL_CKPTS[model_key] = str(
                Path(sam2_ckpt).expanduser().resolve()
            )

        self.detect_orange_points = segment_gripper.detect_orange_points
        self.predictor, self.device = segment_gripper.build_predictor(model_key)
        self.lock = threading.Lock()

    def predict(self, frame):
        frame = np.asarray(frame).astype(np.uint8)
        point_coords, point_labels = self.detect_orange_points(frame)

        with tempfile.TemporaryDirectory(prefix='viewforce_tts_sam2_') as frame_dir:
            Image.fromarray(frame).save(
                str(Path(frame_dir) / '000000.jpg'),
                quality=95,
            )
            autocast = (
                torch.autocast(self.device, dtype=torch.bfloat16)
                if self.device == 'cuda'
                else contextlib.nullcontext()
            )
            with self.lock, torch.inference_mode(), autocast:
                state = self.predictor.init_state(video_path=frame_dir)
                self.predictor.reset_state(state)
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=point_coords,
                    labels=point_labels,
                )
                for _out_idx, _obj_ids, out_logits in self.predictor.propagate_in_video(state):
                    mask = (out_logits[0] > 0.0).cpu().numpy().squeeze()
                    return mask.astype(np.uint8) * 255

        return np.zeros(frame.shape[:2], dtype=np.uint8)


class DiffusionPolicy:
    def __init__(
        self,
        ckpt_path,
        device='cuda',
        steering_pipeline=None,
        tts_frame_key=None,
        tts_mask_key=None,
        tts_auto_mask=False,
        tts_masker=None,
        tts_mask_mode='sam2',
        tts_rollout_dir=None,
        tts_rollout_fps=10.0,
        tts_mask_h_min=GRIPPER_H_MIN,
        tts_mask_h_max=GRIPPER_H_MAX,
        tts_mask_s_min=GRIPPER_S_MIN,
        tts_mask_v_min=GRIPPER_V_MIN,
        tts_mask_min_area=GRIPPER_MIN_BLOB_AREA,
    ):
        print(f'Loading checkpoint: {ckpt_path}')
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.eval().to(device)

        self.policy = policy
        self.device = torch.device(device)
        self.obs_shape_meta = cfg.shape_meta['obs']
        self.warmed_up = False
        self.steering_pipeline = steering_pipeline
        self.tts_frame_key = tts_frame_key
        self.tts_mask_key = tts_mask_key
        self.tts_auto_mask = tts_auto_mask
        self.tts_masker = tts_masker
        self.tts_mask_mode = tts_mask_mode
        self.tts_rollout_dir = Path(tts_rollout_dir) if tts_rollout_dir else None
        self.tts_rollout_fps = tts_rollout_fps
        self.tts_rollout = None
        self.tts_rollout_frames = 0
        self.tts_rollout_lock = threading.Lock()
        self.tts_mask_h_min = tts_mask_h_min
        self.tts_mask_h_max = tts_mask_h_max
        self.tts_mask_s_min = tts_mask_s_min
        self.tts_mask_v_min = tts_mask_v_min
        self.tts_mask_min_area = tts_mask_min_area

    def reset(self):
        self.flush_tts_rollout()
        self.policy.reset()

    def step(self, obs_sequence):
        """obs_sequence: list of obs dicts, length == n_obs_steps"""
        obs_dict = self._build_obs(obs_sequence)
        with torch.no_grad():
            if not self.warmed_up:
                print('Warming up...')
                self.policy.predict_action(obs_dict)
                self.warmed_up = True
            result = self.policy.predict_action(obs_dict)
        # (1, horizon, 8) -> (horizon, 8)
        actions = result['action'][0].cpu().numpy()
        if self.steering_pipeline is not None:
            actions = self._steer_actions(obs_sequence[-1], actions)
        return self._split_action(actions)

    def _steer_actions(self, latest_obs, actions):
        frame_key = self.tts_frame_key
        if frame_key is None:
            image_keys = [
                key for key, meta in self.obs_shape_meta.items()
                if meta.get('type') == 'rgb'
            ]
            if len(image_keys) != 1:
                raise ValueError(
                    'TTS needs --tts-frame-key because the policy has '
                    f'{len(image_keys)} rgb observation keys: {image_keys}'
                )
            frame_key = image_keys[0]

        if frame_key not in latest_obs:
            raise KeyError(
                f'TTS frame key {frame_key!r} not found in obs. '
                f'Available obs keys: {sorted(latest_obs.keys())}'
            )
        frame = latest_obs[frame_key]
        if self.tts_mask_key is not None and self.tts_mask_key in latest_obs:
            mask = latest_obs[self.tts_mask_key]
        elif self.tts_auto_mask:
            mask = self._make_gripper_mask(frame)
        elif self.tts_mask_key is None:
            raise ValueError(
                'TTS is enabled but no mask source was provided. Pass '
                '--tts-mask-key KEY or --tts-auto-mask.'
            )
        else:
            raise KeyError(
                f'TTS mask key {self.tts_mask_key!r} not found in obs. '
                f'Available obs keys: {sorted(latest_obs.keys())}'
            )

        mask_area = int(np.asarray(mask).astype(bool).sum())
        mask_frac = mask_area / float(mask.shape[0] * mask.shape[1])
        frame_mean = float(np.asarray(frame).mean())
        hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV).astype(np.float32)
        sat_mean = float((hsv[:, :, 1] / 255.0).mean())
        val_mean = float((hsv[:, :, 2] / 255.0).mean())
        self._record_tts_rollout_frame(frame, mask)
        result = self.steering_pipeline.steer_action_chunk(
            frame,
            mask,
            actions,
        )
        print(
            'TTS force '
            f'{result.predicted_force.selected_key}='
            f'{result.predicted_force.selected_force:.3f}, '
            f'control={result.predicted_force.control_force:.3f}, '
            f'err={result.force_error:.3f}, '
            f'close_scale={result.close_scale:.3f}, '
            f'motion_scale={result.motion_scale:.3f}, '
            f'mask_px={mask_area}, '
            f'mask_frac={mask_frac:.4f}, '
            f'frame_mean={frame_mean:.1f}, '
            f'sat_mean={sat_mean:.3f}, '
            f'val_mean={val_mean:.3f}'
        )
        return result.action

    def _make_gripper_mask(self, frame):
        if self.tts_mask_mode == 'sam2':
            if self.tts_masker is None:
                raise ValueError('TTS SAM2 mask mode requested but masker was not initialized')
            return self.tts_masker.predict(frame)

        hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[:, :, 0] * 2.0
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        mask = (
            (hue >= self.tts_mask_h_min)
            & (hue <= self.tts_mask_h_max)
            & (sat >= self.tts_mask_s_min)
            & (val >= self.tts_mask_v_min)
        ).astype(np.uint8)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        n_labels, labels, stats, _centroids = cv.connectedComponentsWithStats(
            mask, connectivity=8
        )
        cleaned = np.zeros_like(mask)
        for label in range(1, n_labels):
            if stats[label, cv.CC_STAT_AREA] >= self.tts_mask_min_area:
                cleaned[labels == label] = 255
        return cleaned

    def _record_tts_rollout_frame(self, frame, mask):
        if self.tts_rollout_dir is None:
            return

        frame = np.asarray(frame)
        mask_bool = np.asarray(mask).astype(bool)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        if mask_bool.shape != frame.shape[:2]:
            mask_bool = cv.resize(
                mask_bool.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv.INTER_NEAREST,
            ).astype(bool)

        with self.tts_rollout_lock:
            if self.tts_rollout is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_dir = self.tts_rollout_dir / f'tts_rollout_{timestamp}'
                out_dir.mkdir(parents=True, exist_ok=True)
                frames_dir = out_dir / 'frames'
                frames_dir.mkdir(parents=True, exist_ok=True)
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                size = (frame.shape[1], frame.shape[0])
                self.tts_rollout = {
                    'dir': out_dir,
                    'frames_dir': frames_dir,
                    'overlay': cv.VideoWriter(
                        str(out_dir / 'overlay.mp4'), fourcc, self.tts_rollout_fps, size
                    ),
                    'mask': cv.VideoWriter(
                        str(out_dir / 'mask.mp4'), fourcc, self.tts_rollout_fps, size
                    ),
                }
                self.tts_rollout_frames = 0
                print(f'TTS rollout recording: {out_dir}')
                cv.imwrite(
                    str(out_dir / 'first_frame.png'),
                    cv.cvtColor(frame, cv.COLOR_RGB2BGR),
                )

            overlay = frame.copy()
            overlay[mask_bool] = (
                0.5 * overlay[mask_bool].astype(np.float32)
                + 0.5 * np.array([255, 50, 50], dtype=np.float32)
            ).astype(np.uint8)
            mask_rgb = np.repeat(mask_bool[:, :, None].astype(np.uint8) * 255, 3, axis=2)

            self.tts_rollout['overlay'].write(cv.cvtColor(overlay, cv.COLOR_RGB2BGR))
            self.tts_rollout['mask'].write(cv.cvtColor(mask_rgb, cv.COLOR_RGB2BGR))
            frame_idx = self.tts_rollout_frames
            cv.imwrite(
                str(self.tts_rollout['frames_dir'] / f'overlay_{frame_idx:06d}.png'),
                cv.cvtColor(overlay, cv.COLOR_RGB2BGR),
            )
            cv.imwrite(
                str(self.tts_rollout['frames_dir'] / f'mask_{frame_idx:06d}.png'),
                cv.cvtColor(mask_rgb, cv.COLOR_RGB2BGR),
            )
            self.tts_rollout_frames += 1

    def flush_tts_rollout(self):
        with self.tts_rollout_lock:
            if self.tts_rollout is None:
                return
            out_dir = self.tts_rollout['dir']
            self.tts_rollout['overlay'].release()
            self.tts_rollout['mask'].release()
            print(
                f'TTS rollout saved: {out_dir / "overlay.mp4"} '
                f'({self.tts_rollout_frames} frames)'
            )
            print(f'TTS mask video saved: {out_dir / "mask.mp4"}')
            print(f'TTS PNG frames saved: {out_dir / "frames"}')
            self._encode_h264_rollout(out_dir)
            self.tts_rollout = None
            self.tts_rollout_frames = 0

    def _encode_h264_rollout(self, out_dir):
        if shutil.which('ffmpeg') is None:
            print('ffmpeg not found; skipping H.264 rollout encode')
            return

        frames_dir = out_dir / 'frames'
        jobs = [
            ('overlay_%06d.png', 'overlay_h264.mp4'),
            ('mask_%06d.png', 'mask_h264.mp4'),
        ]
        for pattern, filename in jobs:
            cmd = [
                'ffmpeg',
                '-y',
                '-framerate',
                str(self.tts_rollout_fps),
                '-i',
                str(frames_dir / pattern),
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                str(out_dir / filename),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f'TTS H.264 video saved: {out_dir / filename}')
            except subprocess.CalledProcessError as exc:
                print(f'ffmpeg failed while writing {filename}:')
                print(exc.stderr.decode('utf-8', errors='replace'))

    def _build_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, meta in self.obs_shape_meta.items():
            if meta.get('type') == 'rgb':
                # Keep TTS/ViewForce at source resolution, but feed the DP policy
                # the original 320x240 shape it was deployed with.
                imgs = []
                for obs in obs_sequence:
                    img = obs[key]
                    if img.shape[:2] != (IMAGE_H, IMAGE_W):
                        img = cv.resize(img, (IMAGE_W, IMAGE_H))
                    imgs.append(img)
                # (T, H, W, 3) uint8 -> (T, 3, H, W) float32
                imgs = np.stack(imgs, axis=0)
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.transpose(imgs, (0, 3, 1, 2))  # THWC -> TCHW
                obs_dict_np[key] = imgs
            elif key == 'agent_pos':
                # concatenate arm_pos(3) + arm_quat(4) + gripper_pos(1) per step
                agent_pos = np.stack([
                    np.concatenate([
                        obs['arm_pos'], obs['arm_quat'], obs['gripper_pos']
                    ]) for obs in obs_sequence
                ], axis=0).astype(np.float32)
                obs_dict_np['agent_pos'] = agent_pos
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )
        return obs_dict

    def _split_action(self, actions):
        """(horizon, 8) -> list of action dicts"""
        result = []
        for act in actions:
            result.append({
                'arm_pos':     act[0:3],
                'arm_quat':    act[3:7],
                'gripper_pos': act[7:8],
            })
        return result


class PolicyWrapper:
    """Handles obs history buffering, action chunking, and latency compensation."""

    def __init__(self, policy, n_obs_steps=2, n_action_steps=8):
        self.policy = policy
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        threading.Thread(
            target=self._inference_loop, args=(policy,), daemon=True
        ).start()

    def reset(self):
        self.obs_queue.put('reset')

    def flush_tts_rollout(self):
        if hasattr(self.policy, 'flush_tts_rollout'):
            self.policy.flush_tts_rollout()

    def step(self, obs):
        self.obs_queue.put(obs)
        action = None if self.act_queue.empty() else self.act_queue.get()
        if action is None:
            print('Warning: action queue empty — latency budget may be too low')
        return action

    def _inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        while True:
            try:
                if not self.obs_queue.empty():
                    item = self.obs_queue.get()
                    if item == 'reset':
                        policy.reset()
                        obs_history.clear()
                        start_of_episode = True
                        while not self.act_queue.empty():
                            self.act_queue.get()
                        continue
                    obs_history.append(item)

                if (self.act_queue.qsize() < LATENCY_STEPS
                        and len(obs_history) == self.n_obs_steps):
                    act_sequence = policy.step(list(obs_history))
                    if not self.act_queue.empty():
                        print('Warning: action queue backlog')
                    if start_of_episode:
                        act_sequence = act_sequence[:self.n_action_steps - LATENCY_STEPS]
                        start_of_episode = False
                    else:
                        act_sequence = act_sequence[LATENCY_STEPS:self.n_action_steps]
                    for action in act_sequence:
                        self.act_queue.put(action)

                time.sleep(0.001)
            except Exception:
                print('Policy inference thread crashed:')
                traceback.print_exc()
                raise


class PolicyServer:
    def __init__(self, policy, port=5555):
        self.policy = policy
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f'tcp://*:{port}')
        print(f'Policy server listening on port {port}')

    def _decode_obs(self, obs):
        decoded = {}
        for k, v in obs.items():
            if k.endswith('image'):
                # Decode JPEG and preserve source resolution for TTS/ViewForce.
                # _build_obs handles policy-specific resizing later.
                img = cv.imdecode(v, cv.IMREAD_COLOR)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # fix channel order
                decoded[k] = img
            else:
                decoded[k] = v
        return decoded

    def run(self):
        try:
            while True:
                req = self.socket.recv_pyobj()
                rep = {}
                try:
                    if 'reset' in req:
                        self.policy.reset()
                        print('Policy reset')
                    elif 'obs' in req:
                        obs = self._decode_obs(req['obs'])
                        action = self.policy.step(obs)
                        rep['action'] = action
                except Exception as exc:
                    print('Policy server request failed:')
                    traceback.print_exc()
                    rep['error'] = repr(exc)
                self.socket.send_pyobj(rep)
        finally:
            if hasattr(self.policy, 'flush_tts_rollout'):
                self.policy.flush_tts_rollout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--n-obs-steps', type=int, default=2)
    parser.add_argument('--n-action-steps', type=int, default=8)
    parser.add_argument('--tts-viewforce-ckpt', default=None)
    parser.add_argument('--tts-viewforce-root', default='/home/wfy/repos/ViewForce')
    parser.add_argument('--tts-desired-force', type=float, default=None)
    parser.add_argument('--tts-force-key', default='Fz')
    parser.add_argument(
        '--tts-force-mode',
        choices=['magnitude', 'signed'],
        default='magnitude',
    )
    parser.add_argument('--tts-frame-key', default=None)
    parser.add_argument('--tts-mask-key', default=None)
    parser.add_argument(
        '--tts-auto-mask',
        action='store_true',
        help='Generate the gripper mask from --tts-frame-key',
    )
    parser.add_argument(
        '--tts-mask-mode',
        choices=['sam2', 'hsv'],
        default='sam2',
        help='Mask generator for --tts-auto-mask. sam2 matches the ViewForce offline chain.',
    )
    parser.add_argument(
        '--tts-sam2-model',
        choices=['tiny', 'small', 'base', 'large'],
        default='small',
    )
    parser.add_argument(
        '--tts-sam2-repo',
        default=None,
        help='Path to the local SAM2 repo containing the sam2 Python package.',
    )
    parser.add_argument(
        '--tts-sam2-ckpt',
        default=None,
        help='Path to the local SAM2 checkpoint for --tts-sam2-model.',
    )
    parser.add_argument(
        '--tts-rollout-dir',
        default='tts_rollouts',
        help='Directory for TTS overlay/mask rollout videos. Set empty string to disable.',
    )
    parser.add_argument('--tts-rollout-fps', type=float, default=10.0)
    parser.add_argument('--tts-mask-h-min', type=float, default=GRIPPER_H_MIN)
    parser.add_argument('--tts-mask-h-max', type=float, default=GRIPPER_H_MAX)
    parser.add_argument('--tts-mask-s-min', type=float, default=GRIPPER_S_MIN)
    parser.add_argument('--tts-mask-v-min', type=float, default=GRIPPER_V_MIN)
    parser.add_argument('--tts-mask-min-area', type=int, default=GRIPPER_MIN_BLOB_AREA)
    parser.add_argument('--tts-gripper-index', type=int, default=7)
    parser.add_argument('--tts-close-positive', action='store_true')
    parser.add_argument('--tts-close-negative', action='store_true')
    parser.add_argument('--tts-motion-indices', nargs='*', type=int, default=[])
    parser.add_argument('--tts-deadband', type=float, default=0.10)
    parser.add_argument('--tts-slowdown-band', type=float, default=1.0)
    parser.add_argument('--tts-stop-margin', type=float, default=0.75)
    parser.add_argument('--tts-open-command', type=float, default=0.0)
    parser.add_argument('--tts-close-gain', type=float, default=0.0)
    parser.add_argument('--tts-max-close-command', type=float, default=None)
    args = parser.parse_args()

    steering_pipeline = None
    tts_masker = None
    if args.tts_viewforce_ckpt is not None:
        if args.tts_desired_force is None:
            raise ValueError('--tts-desired-force is required when TTS is enabled')
        if args.tts_close_positive and args.tts_close_negative:
            raise ValueError('Choose only one of --tts-close-positive/--tts-close-negative')

        ForceSteeringConfig, ViewForceSteeringPipeline = _load_steering_classes(
            args.tts_viewforce_root
        )
        close_positive = not args.tts_close_negative
        if args.tts_close_positive:
            close_positive = True
        steering_config = ForceSteeringConfig(
            desired_force=args.tts_desired_force,
            force_key=args.tts_force_key,
            force_mode=args.tts_force_mode,
            deadband=args.tts_deadband,
            slowdown_band=args.tts_slowdown_band,
            stop_margin=args.tts_stop_margin,
            gripper_index=args.tts_gripper_index,
            close_positive=close_positive,
            open_command=args.tts_open_command,
            close_gain=args.tts_close_gain,
            max_close_command=args.tts_max_close_command,
            motion_indices=tuple(args.tts_motion_indices),
        )
        steering_pipeline = ViewForceSteeringPipeline(
            args.tts_viewforce_ckpt,
            steering_config,
            device=args.device,
        )
        if args.tts_auto_mask and args.tts_mask_mode == 'sam2':
            tts_masker = Sam2FrameMasker(
                args.tts_viewforce_root,
                args.tts_sam2_model,
                sam2_repo=args.tts_sam2_repo,
                sam2_ckpt=args.tts_sam2_ckpt,
            )
        print('Test-time force steering enabled')

    policy = DiffusionPolicy(
        args.ckpt_path,
        device=args.device,
        steering_pipeline=steering_pipeline,
        tts_frame_key=args.tts_frame_key,
        tts_mask_key=args.tts_mask_key,
        tts_auto_mask=args.tts_auto_mask,
        tts_masker=tts_masker,
        tts_mask_mode=args.tts_mask_mode,
        tts_rollout_dir=args.tts_rollout_dir if steering_pipeline is not None else None,
        tts_rollout_fps=args.tts_rollout_fps,
        tts_mask_h_min=args.tts_mask_h_min,
        tts_mask_h_max=args.tts_mask_h_max,
        tts_mask_s_min=args.tts_mask_s_min,
        tts_mask_v_min=args.tts_mask_v_min,
        tts_mask_min_area=args.tts_mask_min_area,
    )
    wrapped = PolicyWrapper(policy, args.n_obs_steps, args.n_action_steps)
    atexit.register(wrapped.flush_tts_rollout)

    def flush_and_exit(signum, _frame):
        print(f'Received signal {signum}, flushing TTS rollout before exit')
        wrapped.flush_tts_rollout()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, flush_and_exit)
    signal.signal(signal.SIGTERM, flush_and_exit)
    server = PolicyServer(wrapped, port=args.port)
    server.run()


if __name__ == '__main__':
    main()
