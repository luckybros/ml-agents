from typing import Dict, List

from mlagents_envs.base_env import BaseEnv, BehaviorName, BehaviorSpec
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep, AllStepResult
from mlagents_envs.timers import timed
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.settings import ParameterRandomizationSettings
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)


class SimpleEnvManager(EnvManager):
    """
    Simple implementation of the EnvManager interface that only handles one BaseEnv at a time.
    This is generally only useful for testing; see SubprocessEnvManager for a production-quality implementation.
    """

    def __init__(self, env: BaseEnv, env_params: EnvironmentParametersChannel):
        super().__init__()
        self.env_params = env_params
        self.env = env
        self.previous_step: EnvironmentStep = EnvironmentStep.empty(0)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}

    def _step(self) -> List[EnvironmentStep]:
        all_action_info = self._take_step(self.previous_step)
        self.previous_all_action_info = all_action_info

        for brain_name, action_info in all_action_info.items():
            self.env.set_actions(brain_name, action_info.env_action)
        self.env.step()
        all_step_result = self._generate_all_results()

        step_info = EnvironmentStep(
            all_step_result, 0, self.previous_all_action_info, {}
        )
        self.previous_step = step_info
        return [step_info]

    def _reset_env(
        self, config: Dict[BehaviorName, float] = None
    ) -> List[EnvironmentStep]:  # type: ignore
        self.set_env_parameters(config)
        self.env.reset()
        all_step_result = self._generate_all_results()
        self.previous_step = EnvironmentStep(all_step_result, 0, {}, {})
        return [self.previous_step]

    def set_env_parameters(self, config: Dict = None) -> None:
        """
        Sends environment parameter settings to C# via the
        EnvironmentParametersSidehannel.
        :param config: Dict of environment parameter keys and values
        """
        if config is not None:
            for k, v in config.items():
                if isinstance(v, float):
                    self.env_params.set_float_parameter(k, v)
                elif isinstance(v, ParameterRandomizationSettings):
                    v.apply(k, self.env_params)

    @property
    def training_behaviors(self) -> Dict[BehaviorName, BehaviorSpec]:
        return self.env.behavior_specs

    def close(self):
        self.env.close()

    @timed
    def _take_step(self, last_step: EnvironmentStep) -> Dict[BehaviorName, ActionInfo]:
        # Crea un contenitore vuoto per le azioni che verranno decise.
        all_action_info: Dict[str, ActionInfo] = {}

        # Itera su ogni "tipo" di agente presente nella scena. 
        # 'brain_name' è il nome del tipo di agente (es. "Player", "Enemy").
        # 'step_tuple' contiene i dati di tutti gli agenti di quel tipo.
        for brain_name, step_tuple in last_step.current_all_step_result.items():

            # Questa è la riga più importante.
            # 1. self.policies[brain_name]: Seleziona il "cervello" (la policy) corretto per questo tipo di agente.
            # 2. .get_action(...): Chiama la funzione di decisione di quel cervello.
            # 3. step_tuple[0]: Passa a .get_action le OSSERVAZIONI dell'agente, cioè quello che vede.
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                step_tuple[0],
                0,  # As there is only one worker, we assign the worker_id to 0.
            )

        # Restituisce il dizionario che contiene l'azione decisa per ogni tipo di agente.
        return all_action_info

    def _generate_all_results(self) -> AllStepResult:
        all_step_result: AllStepResult = {}
        for brain_name in self.env.behavior_specs:
            all_step_result[brain_name] = self.env.get_steps(brain_name)
        return all_step_result
