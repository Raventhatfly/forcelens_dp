from typing import Dict, List, Optional
import torch
import numpy as np
import os
import copy
import joblib
import cv2
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def _read_video_frames(video_path: str, out_h: int, out_w: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv2.resize(frame, (out_w, out_h))
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def _load_episode(episode_dir: str, image_keys: List[str], out_h: int, out_w: int):
    data = joblib.load(os.path.join(episode_dir, 'data.pkl'))
    T = len(data['timestamps'])

    # state: arm_pos(3) + arm_quat(4) + gripper_pos(1) = 8
    obs_list = data['observations']
    agent_pos = np.stack([
        np.concatenate([o['arm_pos'], o['arm_quat'], o['gripper_pos']])
        for o in obs_list
    ], axis=0).astype(np.float32)  # (T, 8)

    # action: arm_pos(3) + arm_quat(4) + gripper_pos(1) = 8
    action = np.stack([
        np.concatenate([a['arm_pos'], a['arm_quat'], a['gripper_pos']])
        for a in data['actions']
    ], axis=0).astype(np.float32)  # (T, 8)

    episode = {'agent_pos': agent_pos, 'action': action}

    for key in image_keys:
        video_path = os.path.join(episode_dir, f'{key}.mp4')
        episode[key] = _read_video_frames(video_path, out_h, out_w)  # (T, H, W, 3)

    return episode, T


class PickImageDataset(BaseImageDataset):
    def __init__(self,
            dataset_path,  # str or list of str
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            image_keys: List[str] = None,
            image_size: List[int] = None,  # [H, W]
            ):
        if image_keys is None:
            image_keys = ['base_image', 'wrist_image']
        if image_size is None:
            image_size = [240, 320]

        out_h, out_w = image_size

        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]

        episode_dirs = []
        for path in dataset_path:
            episodes_dir = os.path.join(path, 'episodes')
            episode_dirs += sorted([
                os.path.join(episodes_dir, d)
                for d in os.listdir(episodes_dir)
                if os.path.isdir(os.path.join(episodes_dir, d))
            ])

        replay_buffer = ReplayBuffer.create_empty_numpy()
        for ep_dir in episode_dirs:
            episode, T = _load_episode(ep_dir, image_keys, out_h, out_w)
            replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.image_keys = image_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.train_mask = train_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['agent_pos'])
        for key in self.image_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        obs_dict = {}
        for key in self.image_keys:
            # (T, H, W, 3) -> (T, 3, H, W), float32 [0, 1]
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0
        obs_dict['agent_pos'] = sample['agent_pos'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        return dict_apply({
            'obs': obs_dict,
            'action': action,
        }, torch.from_numpy)
