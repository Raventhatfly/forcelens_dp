"""
Policy inference server for real robot deployment.

Run this in forcelens_dp repo:
    python policy_server.py --ckpt-path outputs/.../checkpoints/xxx.ckpt

On the robot-controller side, use RemotePolicy with:
    image_width=320, image_height=240
"""

import argparse
import math
import queue
import threading
import time
from collections import deque

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq

from diffusion_policy.common.pytorch_util import dict_apply

CONTROL_PERIOD = 0.1        # 10 Hz
LATENCY_BUDGET = 0.2        # 200 ms
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / CONTROL_PERIOD)  # 2

IMAGE_H = 240
IMAGE_W = 320


class DiffusionPolicy:
    def __init__(self, ckpt_path, device='cuda'):
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

    def reset(self):
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
        return self._split_action(actions)

    def _build_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, meta in self.obs_shape_meta.items():
            if meta.get('type') == 'rgb':
                # (T, H, W, 3) uint8 -> (T, 3, H, W) float32
                imgs = np.stack([obs[key] for obs in obs_sequence], axis=0)
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
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        threading.Thread(
            target=self._inference_loop, args=(policy,), daemon=True
        ).start()

    def reset(self):
        self.obs_queue.put('reset')

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
                # Decode JPEG; robot-controller sends RGB encoded as BGR by OpenCV
                img = cv.imdecode(v, cv.IMREAD_COLOR)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # fix channel order
                if img.shape[:2] != (IMAGE_H, IMAGE_W):
                    img = cv.resize(img, (IMAGE_W, IMAGE_H))
                decoded[k] = img
            else:
                decoded[k] = v
        return decoded

    def run(self):
        while True:
            req = self.socket.recv_pyobj()
            rep = {}
            if 'reset' in req:
                self.policy.reset()
                print('Policy reset')
            elif 'obs' in req:
                obs = self._decode_obs(req['obs'])
                action = self.policy.step(obs)
                rep['action'] = action
            self.socket.send_pyobj(rep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--n-obs-steps', type=int, default=2)
    parser.add_argument('--n-action-steps', type=int, default=8)
    args = parser.parse_args()

    policy = DiffusionPolicy(args.ckpt_path, device=args.device)
    wrapped = PolicyWrapper(policy, args.n_obs_steps, args.n_action_steps)
    server = PolicyServer(wrapped, port=args.port)
    server.run()


if __name__ == '__main__':
    main()
