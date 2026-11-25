"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    # TODO: Initialize a random agent position state inside a square 2D room...
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randint(0, room_size, (2,), dtype=torch.float32)

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    # TODO: Apply a discrete action to the agent state with wall clamping.
    new_state = state.clone()
    match action:
        case 0:
            new_state[1] -= 1
        case 1:
            new_state[1] += 1
        case 2:
            new_state[0] -= 1
        case 3:
            new_state[0] += 1

    return torch.clamp(new_state, 0, room_size-1)

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    # TODO: Render a (1, room_size, room_size) float32 obs with agent pixel 1.0...
    obs = torch.zeros((room_size, room_size), dtype=torch.float32)
    x, y = state.long().tolist()
    obs[y, x] = 1.0

    return obs.unsqueeze(0)

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Reset the environment by sampling a new state and its observation.
    state = init_env_state(room_size, seed)
    obs = render_observation(state, room_size)

    return state, obs

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Advance the env by one action; return (next_state, next_observation).
    next_state = apply_action(state, action, room_size)
    next_obs = render_observation(next_state, room_size)

    return next_state, next_obs

# Step 6 - collect_random_transitions
def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: collect a dataset of (obs, action, next_obs, state, next_state) transitions...
    torch.manual_seed(seed)
    observations = []
    next_observations = []
    actions = torch.randint(0, 4, (num_transitions,))
    states = []
    next_states = []

    state, obs = env_reset(room_size, seed)
    for action in actions.tolist():
        next_state, next_obs = env_step(state, action, room_size)

        observations.append(obs)
        next_observations.append(next_obs)
        states.append(state)
        next_states.append(next_state)

        state, obs = next_state, next_obs

    if num_transitions == 0:
        observations = torch.empty((0, 1, room_size, room_size))
        next_observations = torch.empty((0, 1, room_size, room_size))
        states = torch.empty((0, 2))
        next_states = torch.empty((0, 2))
    else:
        observations = torch.stack(observations)
        next_observations = torch.stack(next_observations)
        actions = actions
        states = torch.stack(states)
        next_states = torch.stack(next_states)

    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        states=states,
        next_states=next_states
    )

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: Build a JEPA training-ready transition dataset by collecting random transitions...
    return collect_random_transitions(num_transitions, room_size, seed)

# Step 8 - init_encoder_params
import torch.nn as nn


def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    # TODO: Initialize parameters of a small CNN encoder that maps pixel observations to latent embeddings.
    torch.manual_seed(seed)
    conv1_w = nn.Conv2d(obs_channels, 16, (3, 3), stride=1, padding=1, bias=False)
    nn.init.normal_(conv1_w.weight, mean=0.0, std=0.1)
    conv1_b = torch.zeros((16, ), requires_grad=True)

    conv2_w = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1, bias=False)
    nn.init.normal_(conv1_w.weight, mean=0.0, std=0.1)
    conv2_b = torch.zeros((32, ), requires_grad=True)

    # h1 = room_size
    hout = (room_size - 1)//2 + 1
    flat_dim = 32 * hout * hout

    fc_w = nn.Linear(flat_dim, latent_dim, bias=False)
    nn.init.normal_(fc_w.weight, mean=0.0, std=0.1)
    fc_b = torch.zeros((latent_dim, ), requires_grad=True)

    return dict(
        conv1_w=conv1_w.weight,
        conv1_b=conv1_b,
        conv2_w=conv2_w.weight,
        conv2_b=conv2_b,
        fc_w=fc_w.weight,
        fc_b=fc_b
    )

# Step 9 - encoder_forward
import torch.nn.functional as F


def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Run a small two-layer CNN that maps pixel observations to latent embeddings.
    b = obs.shape[0]
    x = F.conv2d(obs, encoder_params['conv1_w'], bias=encoder_params['conv1_b'], stride=1, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, encoder_params['conv2_w'], bias=encoder_params['conv2_b'], stride=2, padding=1)
    x = F.relu(x)
    x = x.reshape(b, -1)

    x = F.linear(x, encoder_params['fc_w'], bias=encoder_params['fc_b'])
    return x

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    # TODO: Create the EMA target encoder by deep-copying the online encoder params.
    return {key: val.detach().clone() for key, val in encoder_params.items()}

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    # TODO: Refresh target encoder params via EMA of the online encoder.
    return {key: target_params[key] * tau + encoder_params[key] * (1 - tau) for key in target_params}

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Batch-encode observations into latent embeddings using encoder params.
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    # TODO: Initialize and return a dict of MLP predictor parameters.
    torch.manual_seed(seed)
    action_embed_w = torch.randn((action_dim, latent_dim)) * 0.02
    action_embed_w.requires_grad = True

    fc1_w = torch.randn((hidden_dim, 2*latent_dim)) * 0.02
    fc1_w.requires_grad = True
    fc1_b = torch.zeros((hidden_dim,), requires_grad=True)

    fc2_w = torch.randn((latent_dim, hidden_dim)) * 0.02
    fc2_w.requires_grad = True
    fc2_b = torch.zeros((latent_dim,), requires_grad=True)

    return dict(
        action_embed_w=action_embed_w,
        fc1_w=fc1_w,
        fc1_b=fc1_b,
        fc2_w=fc2_w,
        fc2_b=fc2_b
    )

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Embed discrete actions into continuous vectors via a learned action embedding table.
    return predictor_params['action_embed_w'][actions]

# Step 15 - predictor_forward (not yet solved)
# TODO: implement

# Step 16 - predict_next_embedding (not yet solved)
# TODO: implement

# Step 17 - prediction_loss (not yet solved)
# TODO: implement

# Step 18 - variance_loss (not yet solved)
# TODO: implement

# Step 19 - covariance_loss (not yet solved)
# TODO: implement

# Step 20 - vicreg_regularizer (not yet solved)
# TODO: implement

# Step 21 - jepa_loss (not yet solved)
# TODO: implement

# Step 22 - collapse_metric (not yet solved)
# TODO: implement

# Step 23 - jepa_training_step (not yet solved)
# TODO: implement

# Step 24 - train_jepa (not yet solved)
# TODO: implement

# Step 25 - rollout_latent_dynamics (not yet solved)
# TODO: implement

# Step 26 - multi_step_prediction_error (not yet solved)
# TODO: implement

# Step 27 - init_linear_probe (not yet solved)
# TODO: implement

# Step 28 - train_linear_probe (not yet solved)
# TODO: implement

# Step 29 - probe_state_recovery (not yet solved)
# TODO: implement

# Step 30 - encode_goal (not yet solved)
# TODO: implement

# Step 31 - latent_cost (not yet solved)
# TODO: implement

# Step 32 - sample_action_sequences (not yet solved)
# TODO: implement

# Step 33 - score_action_sequences (not yet solved)
# TODO: implement

# Step 34 - select_best_plan (not yet solved)
# TODO: implement

# Step 35 - mpc_step (not yet solved)
# TODO: implement

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

