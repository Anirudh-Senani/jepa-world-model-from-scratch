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

# Step 15 - predictor_forward
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Implement predictor_forward, the forward pass of the action-conditioned dynamics predictor.
    action_embeddings = embed_action(actions, predictor_params)

    x = torch.cat([embeddings, action_embeddings], dim=-1)
    x = F.linear(x, predictor_params['fc1_w'], bias=predictor_params['fc1_b'])
    x = F.relu(x)
    x = F.linear(x, predictor_params['fc2_w'], bias=predictor_params['fc2_b'])

    return x

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Predict the next latent embedding given current embeddings and actions...
    return predictor_forward(embeddings, actions, predictor_params)

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: Compute the JEPA prediction loss as mean squared error...
    return ((predicted - target)**2).mean()

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # TODO: Compute VICReg variance hinge loss encouraging each dim std >= gamma.
    dim_std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.relu(gamma - dim_std).mean()

# Step 19 - covariance_loss
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Implement `covariance_loss` to compute the VICReg covariance regularization from embeddings.
    B, D = embeddings.shape
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered)/(B-1)

    loss = (covariance**2).sum() - (covariance.diagonal()**2).sum()
    return loss/D

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    # TODO: Combine variance and covariance terms into the VICReg regularizer...
    var_loss = variance_loss(embeddings, gamma)
    cov_loss = covariance_loss(embeddings)

    return var_weight * var_loss + cov_weight * cov_loss

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    # TODO: Compose the full JEPA objective from prediction and VICReg terms...
    pred_loss = prediction_loss(predicted, target)
    vicreg_loss = vicreg_regularizer(online_embeddings, var_weight, cov_weight)

    return pred_weight * pred_loss + vicreg_loss

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Measure collapse as the mean of per-dimension batch stds
    return embeddings.std(dim=0).mean()

# Step 23 - jepa_training_step
def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    # TODO: Perform one full JEPA gradient update on a batch of transitions
    for key in encoder_params:
        encoder_params[key].grad = None

    for key in target_params:
        target_params[key].grad = None

    for key in predictor_params:
        predictor_params[key].grad = None

    online_embeddings = encode_batch(batch['observations'], encoder_params)

    with torch.no_grad():
        target_embeddings = encode_batch(batch['next_observations'], target_params)

    predicted_embeddings = predict_next_embedding(online_embeddings, batch['actions'], predictor_params)
    loss = jepa_loss(predicted_embeddings, target_embeddings, online_embeddings)

    loss.backward()

    updated_encoder_params = {}
    for key, param in encoder_params.items():
        if param.grad is not None:
            updated_encoder_params[key] = (param - lr * param.grad).detach().requires_grad_(True)
        else:
            updated_encoder_params[key] = encoder_params[key]

    updated_predictor_params = {}
    for key, param in predictor_params.items():
        if param.grad is not None:
            updated_predictor_params[key] = (param - lr * param.grad).detach().requires_grad_(True)
        else:
            updated_predictor_params[key] = predictor_params[key]

    updated_target_params = ema_update(target_params, updated_encoder_params, tau)
    collapse = collapse_metric(online_embeddings.detach())

    return updated_encoder_params, updated_target_params, updated_predictor_params, loss.item(), collapse.item()

# Step 24 - train_jepa
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    # TODO: Train JEPA for num_steps with collapse monitoring, return params + history
    torch.manual_seed(seed)
    history = []
    N = dataset['observations'].shape[0]

    for _ in range(num_steps):
        inds = torch.randint(0, N, (batch_size,))
        batch = {key: dataset[key][inds] for key in dataset}
        encoder_params, target_params, predictor_params, loss, collapse = jepa_training_step(batch, encoder_params, target_params, predictor_params, lr, tau)

        history.append({'loss':loss, 'collapse': collapse})

    return encoder_params, target_params, predictor_params, history

# Step 25 - rollout_latent_dynamics
def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Roll out multi-step latent dynamics via the action-conditioned predictor...
    batched = True
    embedding = initial_embedding
    if initial_embedding.dim() == 1:
        batched = False
        embedding = initial_embedding.unsqueeze(0)

    B = embedding.shape[0]

    if actions.dim() == 1:    
        actions = actions.expand(B, -1)

    traj = []
    traj.append(embedding)
    for t in range(actions.shape[1]):
        embedding = predict_next_embedding(embedding, actions[:, t], predictor_params)
        traj.append(embedding)

    traj = torch.stack(traj, dim=0)
    if not batched:
        traj = traj.squeeze(1)

    return traj

# Step 26 - multi_step_prediction_error
def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    # TODO: Evaluate multi-step latent prediction accuracy via mean MSE over samples
    N, C, H, _ = dataset['observations'].shape
    samples = min(num_samples, N - horizon)

    with torch.no_grad():
        obs = dataset['observations'][:samples]
        actions = torch.stack([dataset['actions'][i:i+horizon] for i in range(samples)], dim=0)

        initial_embeddings = encode_batch(obs, encoder_params)
        traj = rollout_latent_dynamics(initial_embeddings, actions, predictor_params)

        next_obs = torch.stack([dataset['next_observations'][i:i+horizon] for i in range(samples)], dim=0)

        next_obs = next_obs.transpose(0, 1).reshape(samples*horizon, C, H, H)
        target_embeddings = encode_batch(next_obs, target_params)
        target_embeddings = target_embeddings.reshape(horizon, samples, -1)

    return ((traj[1:] - target_embeddings)**2).mean().item()

# Step 27 - init_linear_probe
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    # TODO: Initialize a linear probe that maps latent embeddings to true agent state (x, y).
    torch.manual_seed(seed)

    w = torch.randn(state_dim, latent_dim) * 0.01
    w.requires_grad = True

    b = torch.zeros(state_dim, requires_grad=True)

    return dict(w=w, b=b)

# Step 28 - train_linear_probe
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    # TODO: Train the linear probe via MSE regression from frozen embeddings to true agent states.
    # emb = embeddings.detach().clone()
    # target = states.detach().clone()

    # emb.requires_grad = True
    # target.requires_grad = True

    for _ in range(num_steps):
        params = {}
        params['w'] = probe_params['w'].detach().clone()
        params['b'] = probe_params['b'].detach().clone()

        params['w'].requires_grad = True
        params['b'].requires_grad = True

        params['w'].grad = None
        params['b'].grad = None

        preds = embeddings.detach() @ params['w'].T + params['b']
        loss = ((preds - states.detach())**2).mean()

        loss.backward()
        with torch.no_grad():
            params['w'] -= lr*params['w'].grad
            params['b'] -= lr*params['b'].grad

        probe_params = params

    return probe_params

# Step 29 - probe_state_recovery
def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    # TODO: Encode observations, train linear probe, report state recovery metrics
    embeddings = encode_batch(dataset['observations'], encoder_params)
    latent_dim = embeddings.shape[-1]
    state_dim = dataset['states'].shape[-1]

    if probe_params is None:
        probe_params = init_linear_probe(latent_dim=latent_dim, state_dim=state_dim)

    # if probe_params['w'].shape[0] == latw_dim:
    #     probe_params['w'] = probe_params['w'].T

    probe_params = train_linear_probe(embeddings, dataset['states'], probe_params, num_probe_steps)

    with torch.no_grad():
        preds = embeddings @ probe_params['w'].T + probe_params['b']
        mse = ((preds - dataset['states'])**2).mean()
        mean_abs_error = ((preds - dataset['states']).abs()).mean()

    return dict(
        mse=mse.item(),
        mean_abs_error=mean_abs_error.item(),
        probe_params=probe_params
    )

# Step 30 - encode_goal
def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    # TODO: Convert a desired agent position into a latent goal embedding.
    obs = render_observation(goal_state, room_size)
    embedding = encoder_forward(obs[None, :], encoder_params)

    return embedding.squeeze(0)

# Step 31 - latent_cost
def latent_cost(latents, goal_embedding):
    # TODO: Compute squared L2 distance from each latent embedding to a goal embedding...
    return ((latents - goal_embedding[None, :])**2).sum(dim=-1)

# Step 32 - sample_action_sequences
def sample_action_sequences(n_sequences, horizon, n_actions):
    # TODO: Sample random discrete action sequences for random-shooting MPC...
    return torch.randint(0, n_actions, (n_sequences, horizon))

# Step 33 - score_action_sequences
def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    # TODO: Score each action sequence by cumulative latent distance to the goal...
    N, H = action_sequences.shape
    initial_embedding = start_embedding.unsqueeze(0).expand(N, -1)
    traj = rollout_latent_dynamics(initial_embedding, action_sequences, predictor_params)
    return latent_cost(traj[1:], goal_embedding).sum(dim=0)

# Step 34 - select_best_plan
def select_best_plan(action_sequences, costs):
    # TODO: Return the action sequence with the lowest cost from the candidates.
    ind = torch.argmin(costs)
    return action_sequences[ind]

# Step 35 - mpc_step
def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    # TODO: Perform one random-shooting MPC step; return first action of best plan
    action_seqs = sample_action_sequences(n_sequences, horizon, n_actions)
    costs = score_action_sequences(start_embedding, action_seqs, goal_embedding, predictor_params)
    return select_best_plan(action_seqs, costs).tolist()[0]

# Step 36 - run_mpc_episode
def run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions):
    # TODO: Run a closed-loop random-shooting MPC episode toward a goal position.
    state, obs = env_reset(room_size)
    goal_pos = torch.tensor(goal_pos)
    goal_embedding = encode_goal(goal_pos, encoder_params, room_size)

    trajectory = [tuple(state.tolist())]
    success = False
    steps = 0
    while steps < max_steps:
        if all((state == goal_pos).tolist()):
            success = True
            break

        start_embedding = encode_batch(obs, encoder_params)
        action = mpc_step(start_embedding.squeeze(0), goal_embedding, predictor_params, n_sequences, horizon, n_actions)
        state, obs = env_step(state, action, room_size)
        trajectory.append(tuple(state.tolist()))

        steps += 1

    final_distance = (((state - goal_pos)**2).sum().item())**0.5

    return dict(
        success=success,
        steps=steps,
        trajectory=trajectory,
        final_distance=final_distance
    )

# Step 37 - evaluate_planner
def evaluate_planner(encoder_params, predictor_params, room_size, agent_size, n_episodes, max_steps, n_sequences, horizon, n_actions):
    # TODO: Evaluate the random-shooting MPC planner over multiple random start-goal episodes
    success_rate = 0
    mean_steps = 0
    mean_final_distance = 0.0

    for _ in range(n_episodes):
        goal_pos = torch.randint(0, room_size, (2,)).float()
        stats = run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions)
        success_rate += int(stats['success'])
        mean_steps += stats['steps']
        mean_final_distance += stats['final_distance']

    return dict(
        success_rate=success_rate/n_episodes,
        mean_steps=mean_steps/n_episodes,
        mean_final_distance=mean_final_distance/n_episodes
    )

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

