from typing import Any, Dict, List
import numpy as np
from mlagents.torch_utils import torch, default_device
import copy
import logging

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import DecisionSteps, BehaviorSpec
from mlagents_envs.timers import timed
from mlagents_envs.logging_util import get_logger

from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import GlobalSteps

from mlagents.trainers.torch_entities.utils import ModelUtils

EPSILON = 1e-7  # Small value to avoid divide by zero

logger = get_logger(__name__)

class TorchPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor_cls: type,
        actor_kwargs: Dict[str, Any],
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous actions, as well as recurrent networks.
        :param seed: Random seed.
        :param behavior_spec: Assigned BehaviorSpec object.
        :param network_settings: Defined network parameters.
        :param actor_cls: The type of Actor
        :param actor_kwargs: Keyword args for the Actor class
        """
        super().__init__(seed, behavior_spec, network_settings)
        self.global_step = (
            GlobalSteps()
        )  # could be much simpler if TorchPolicy is nn.Module

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.actor = actor_cls(
            observation_specs=self.behavior_spec.observation_specs,
            network_settings=network_settings,
            action_spec=behavior_spec.action_spec,
            **actor_kwargs,
        )

        # Save the m_size needed for export
        self._export_m_size = self.m_size
        # m_size needed for training is determined by network, not trainer settings
        self.m_size = self.actor.memory_size

        # self.mock_buffer : List[np.ndarray] = []
        self.actor.to(default_device())

    @property
    def export_memory_size(self) -> int:
        """
        Returns the memory size of the exported ONNX policy. This only includes the memory
        of the Actor and not any auxillary networks.
        """
        return self._export_m_size

    def _extract_masks(self, decision_requests: DecisionSteps) -> np.ndarray:
        mask = None
        if self.behavior_spec.action_spec.discrete_size > 0:
            num_discrete_flat = np.sum(self.behavior_spec.action_spec.discrete_branches)
            mask = torch.ones([len(decision_requests), num_discrete_flat])
            if decision_requests.action_mask is not None:
                mask = torch.as_tensor(
                    1 - np.concatenate(decision_requests.action_mask, axis=1)
                )
        return mask

    @timed
    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param global_agent_ids:
        :param decision_requests: DecisionStep object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        # 1. Estrazione dei Dati di Input
        obs = decision_requests.obs # Prende le osservazioni (vettori, immagini, ecc.)
        #ogger.info(f"obs: {obs}")
        masks = self._extract_masks(decision_requests)  # Prende la maschera delle azioni proibite (che abbiamo appena analizzato)

        # 2. Conversione in Tensori PyTorch
        # La rete neurale lavora con tensori, non con array numpy, quindi li convertiamo.
        tensor_obs = [torch.as_tensor(np_ob) for np_ob in obs]

        # 3. Gestione della Memoria (per Reti Ricorrenti come LSTM)
        # Se la rete ha una memoria (per ricordare il passato), qui la recupera.
        # Se non è una rete ricorrente, questo sarà un tensore vuoto.
        memories = torch.as_tensor(self.retrieve_memories(global_agent_ids)).unsqueeze(
            0
        )

        # 4. ESECUZIONE DELLA RETE NEURALE (IL PASSAGGIO CHIAVE)
        with torch.no_grad():
            # torch.no_grad() dice a PyTorch: "Stiamo solo decidendo, non calcolare
            # gradienti o prepararti per l'apprendimento". Questo rende l'esecuzione molto più veloce.

            # Chiama l'attore (la parte della rete che decide l'azione)
            action, run_out, memories = self.actor.get_action_and_stats(
                tensor_obs, masks=masks, memories=memories
            ) # AgentAction, Dict[str, Any]

        # 5. Formattazione dell'Output
        # Il resto della funzione prende i risultati grezzi dalla rete (che sono tensori)
        # e li impacchetta in un formato più pulito (numpy array, tuple, ecc.)
        # che il resto del framework può usare.
        run_out["action"] = action.to_action_tuple()     # L'azione da eseguire
        if "log_probs" in run_out:
            run_out["log_probs"] = run_out["log_probs"].to_log_probs_tuple()    # La probabilità logaritmica dell'azione scelta
        if "entropy" in run_out:
            run_out["entropy"] = ModelUtils.to_numpy(run_out["entropy"])    # Una misura di quanto "incerta" era la rete sulla sua scelta
        if self.use_recurrent:
            run_out["memory_out"] = ModelUtils.to_numpy(memories).squeeze(0)    # La nuova memoria da salvare per il prossimo passo
        return run_out  # Restituisce un dizionario con tutte queste informazioni

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param worker_id:
        :param decision_requests: A dictionary of behavior names and DecisionSteps from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(decision_requests) == 0:
            return ActionInfo.empty()

        obs = decision_requests.obs

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.

        #logger.info(f"agent: {global_agent_ids}")
        run_out = self.evaluate(decision_requests, global_agent_ids)
        self.save_memories(global_agent_ids, run_out.get("memory_out"))
        self.check_nan_action(run_out.get("action"))
        
        """
        for agent_id in global_agent_ids:
            a = run_out.get("action")[agent_id]
            logger.debug(f"Agent id: {agent_id}")
            logger.debug(f"Action: {a}")
        """
        #logger.debug(decision_requests.agent_id)
        #logger.debug(run_out.get("action").discrete)
        #logger.debug(run_out.get("env_action").discrete)
        

        return ActionInfo(
            action=run_out.get("action"),
            env_action=run_out.get("env_action"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        return self.global_step.current_step

    def set_step(self, step: int) -> int:
        """
        Sets current model step to step without creating additional ops.
        :param step: Step to set the current model step to.
        :return: The step the model was set to.
        """
        self.global_step.current_step = step
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        self.global_step.increment(n_steps)
        return self.get_current_step()

    def load_weights(self, values: List[np.ndarray]) -> None:
        self.actor.load_state_dict(values)

    def init_load_weights(self) -> None:
        pass

    def get_weights(self) -> List[np.ndarray]:
        return copy.deepcopy(self.actor.state_dict())

    def get_modules(self):
        return {"Policy": self.actor, "global_step": self.global_step}
