"""
Alert Escalation MDP (Markov Decision Process) for SepsisNet.

States   : GREEN → YELLOW → ORANGE → RED
Actions  : No-Action, Notify, Escalate, ICU-Code
Rewards  : TP (+100), FN (-500), FP (-50), TN (+10)

Hysteresis prevents premature escalation by requiring a configurable
number of consecutive non-GREEN states before escalating.
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AlertState(IntEnum):
    GREEN = 0
    YELLOW = 1
    ORANGE = 2
    RED = 3


class AlertAction(IntEnum):
    NO_ACTION = 0
    NOTIFY = 1
    ESCALATE = 2
    ICU_CODE = 3


_STATE_NAMES: Dict[int, str] = {
    0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"
}
_ACTION_NAMES: Dict[int, str] = {
    0: "No-Action", 1: "Notify", 2: "Escalate", 3: "ICU-Code"
}

# Reward table: reward[action][true_label]
_REWARDS: Dict[AlertAction, Dict[int, float]] = {
    AlertAction.NO_ACTION: {1: -500.0, 0: 10.0},
    AlertAction.NOTIFY:    {1:  100.0, 0: -50.0},
    AlertAction.ESCALATE:  {1:  100.0, 0: -50.0},
    AlertAction.ICU_CODE:  {1:  100.0, 0: -50.0},
}


class AlertEscalationMDP:
    """
    MDP-based alert escalation that maps (P_sepsis, uncertainty) → action.

    Hysteresis: requires `consecutive_threshold` consecutive elevated states
    before escalating to the next action level.
    """

    def __init__(self, consecutive_threshold: int = 2) -> None:
        self.consecutive_threshold = consecutive_threshold

    # ------------------------------------------------------------------

    def get_state(self, sepsis_prob: float, uncertainty: float) -> AlertState:
        """Map prediction + uncertainty to an alert state."""
        adjusted = float(sepsis_prob) + 0.10 * float(uncertainty)
        if adjusted < 0.10:
            return AlertState.GREEN
        elif adjusted < 0.30:
            return AlertState.YELLOW
        elif adjusted < 0.60:
            return AlertState.ORANGE
        else:
            return AlertState.RED

    def get_action(
        self,
        state: AlertState,
        prev_state: AlertState,
        consecutive_alerts: int,
    ) -> AlertAction:
        """
        Select action based on current state and alert history.

        Hysteresis prevents false-positive escalation cascades:
        YELLOW requires `consecutive_threshold` counts before Notify,
        ORANGE requires threshold before Escalate,
        RED always triggers ICU-Code.
        """
        if state == AlertState.GREEN:
            return AlertAction.NO_ACTION

        if state == AlertState.YELLOW:
            return (AlertAction.NOTIFY
                    if consecutive_alerts >= self.consecutive_threshold
                    else AlertAction.NO_ACTION)

        if state == AlertState.ORANGE:
            return (AlertAction.ESCALATE
                    if consecutive_alerts >= self.consecutive_threshold
                    else AlertAction.NOTIFY)

        # RED — immediate action regardless of history
        return AlertAction.ICU_CODE

    def compute_reward(self, action: AlertAction, true_label: int) -> float:
        """Return the reward for (action, label) pair."""
        return _REWARDS[action][int(true_label)]

    # ------------------------------------------------------------------

    def process_sequence(
        self,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[List[Dict], List[AlertState], List[AlertAction], np.ndarray]:
        """
        Simulate the MDP over a patient cohort.

        Parameters
        ----------
        y_pred_mean : mean P(sepsis),  shape (n,)
        y_pred_std  : prediction std,  shape (n,)
        y_true      : true labels,     shape (n,)

        Returns
        -------
        alerts  : list of per-patient alert dicts
        states  : list of AlertState per patient
        actions : list of AlertAction per patient
        rewards : ndarray, shape (n,)
        """
        alerts: List[Dict] = []
        states: List[AlertState] = []
        actions: List[AlertAction] = []
        rewards: List[float] = []

        prev_state = AlertState.GREEN
        consecutive = 0

        for i in range(len(y_pred_mean)):
            state = self.get_state(y_pred_mean[i], y_pred_std[i])
            consecutive = (consecutive + 1) if state > AlertState.GREEN else 0
            action = self.get_action(state, prev_state, consecutive)
            reward = self.compute_reward(action, int(y_true[i]))

            alerts.append({
                "patient": i,
                "state": _STATE_NAMES[int(state)],
                "action": _ACTION_NAMES[int(action)],
                "sepsis_prob": float(y_pred_mean[i]),
                "uncertainty": float(y_pred_std[i]),
                "true_label": int(y_true[i]),
                "reward": reward,
            })
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            prev_state = state

        total_reward = float(np.sum(rewards))
        logger.info("MDP simulation complete — total reward: %.1f", total_reward)
        return alerts, states, actions, np.array(rewards)
