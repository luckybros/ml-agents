from typing import Dict, cast
import attr
import logging

from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)

@attr.s(auto_attribs=True)
class PPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR


class TorchPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )

        params = list(self.policy.actor.parameters())
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())
            params += list(self._critic.parameters())

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters

        # 1. Ottiene i valori "decaduti" degli iperparametri (learning rate, epsilon, beta).
        #    Questi valori possono diminuire nel corso dell'addestramento.
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())

        # 2. Recupera i dati dal mini-batch e li converte in tensori PyTorch.
        #    Questi sono i "target" e i dati "vecchi" che abbiamo calcolato e salvato in precedenza.
        returns = {}    # Ritorni ottenuti prima (somma dei vantaggi e dei valori)
        old_values = {}    # Valori ottenuti prima  
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        # In questa fase recuperiamo informazioni dal batch
        n_obs = len(self.policy.behavior_spec.observation_specs)    # Numero di osservazioni, che sarebbe numero di agenti
        current_obs = ObsUtil.from_buffer(batch, n_obs)     # Ottengo le osservazioni, sono per ogni agente pari al batch size
        #logger.debug("---- UPDATE IN PPOOPTIMIZER ----")
        #logger.debug(f"n_obs: {n_obs}")
        #logger.debug(f"current_obs: {current_obs}")
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]   # Le trasformo in tensori
        #logger.debug(f"current_obs_tensor: {current_obs}")


        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK]) # Recupera le maschere delle azioni
        actions = AgentAction.from_buffer(batch)    # Recupera le azioni, sono di numero pari al batch size
        # logger.debug(f"actions: {actions}")

        # Recupera le memorie LSTM (se presenti)
        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        # Chiedo all'attore le probabilità con i pesi attuali (CON LE MEDESIME AZIONI DELLA OLD), sempre pari al batch size
        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]        
        entropy = run_out["entropy"]

        #logger.info(f"log_probs: {log_probs}")
        #logger.info(f"sum: {log_probs[0].exp().sum()}")
        
        #logger.info(f"log_probs_2: {log_probs.all_discrete_tensor.shape[1]}")
        # Value del critic con i pesi attuali della rete 
        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )

        # Rimodellamento, serve per ottenere solo le probabilità dell'azione scelta 
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        #logger.debug(f"old_log_probs: {old_log_probs}")
        

        # "filtro di sicurezza" fondamentale per garantire che l'agente impari solo dall'esperienza valida e non da dati "spazzatura" o irrilevanti.
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        # È un solo valore
        # Passo: valori attuali, vecchi valori, returns (valori + ricompense vecchi)
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )

        # Anch'esso è un solo valore
        # Passo log probs attuali, vecchie log probs, vantaggi
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )

        loss = (
            policy_loss
            + 0.5 * value_loss
            # Minimizzare una loss con
            #  -entropia è equivalente a massimizzare l'entropia. 'decay_bet' ne regola l'intensità.
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)

        # Azzera i gradienti calcolati nel mini-batch precedente.
        self.optimizer.zero_grad()

        # Calcola il gradiente
        loss.backward()

        # Aggiorna i pesi della rete
        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        return update_stats

    # TODO move module update into TorchOptimizer for reward_provider
    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
