#!/Users/shukaihu/Code_Project_Local/AlphaBig2-codex/.venv/bin/python
"""Bridge Big2VisionAgent observations to the AlphaBig2 v196 ML_AB model.

Usage:
    BIG2_AGENT_COMMAND=/abs/path/to/alpha_big2_wrapper.py \
        uv run big2-agent autoplay-agent --executor packet

The wrapper reads one AgentObservation JSON object from stdin and writes one
AgentDecision JSON object to stdout. Diagnostics go to stderr only.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[alpha_big2] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ALPHA_BIG2_DIR = ROOT_DIR.parent / "AlphaBig2-codex"
DEFAULT_CKPT = Path("ML_AB/models/big2_transformer_best.pt")
DEFAULT_AGENT_TYPE = "mcts"

SEAT_TO_PLAYER = {
    "self": 1,
    "right": 2,
    "top": 3,
    "left": 4,
}
PLAYER_TO_SEAT = {player: seat for seat, player in SEAT_TO_PLAYER.items()}
PASS_DECISION = {"action": "pass", "card_codes": [], "combo_type": None}
REEXEC_ENV_KEY = "ALPHA_BIG2_WRAPPER_REEXEC"
DEBUG_ENV_KEY = "BIG2_AGENT_DEBUG_DIR"

BIG2_TO_ALPHA_SUIT = {"1": 4, "2": 3, "3": 2, "4": 1}
BIG2_TO_ALPHA_RANK = {
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "1": 12,
    "2": 13,
}
STRAIGHT_SEQUENCES = [
    (12, 13, 1, 2, 3),
    (1, 2, 3, 4, 5),
    (2, 3, 4, 5, 6),
    (3, 4, 5, 6, 7),
    (4, 5, 6, 7, 8),
    (5, 6, 7, 8, 9),
    (6, 7, 8, 9, 10),
    (7, 8, 9, 10, 11),
    (8, 9, 10, 11, 12),
    (13, 1, 2, 3, 4),
]

_ALPHA_CTX: dict[str, Any] | None = None


def card_code_to_alpha_id(code: str) -> int:
    if len(code) != 2:
        raise ValueError(f"Invalid card code: {code!r}")
    suit_code = code[0]
    rank_code = code[1]
    if suit_code not in BIG2_TO_ALPHA_SUIT or rank_code not in BIG2_TO_ALPHA_RANK:
        raise ValueError(f"Invalid card code: {code!r}")
    return (BIG2_TO_ALPHA_RANK[rank_code] - 1) * 4 + BIG2_TO_ALPHA_SUIT[suit_code]


def card_codes_to_alpha_ids(codes: list[str]) -> list[int]:
    return sorted(card_code_to_alpha_id(code) for code in codes)


def _card_rank_value(card_id: int) -> int:
    return ((int(card_id) - 1) // 4) + 1


def _straight_rank(card_ids: list[int]) -> tuple[int, int] | None:
    if len(card_ids) != 5:
        return None
    values = [_card_rank_value(card_id) for card_id in card_ids]
    if len(set(values)) != 5:
        return None
    value_set = set(values)
    for seq_idx, seq in enumerate(STRAIGHT_SEQUENCES):
        if value_set != set(seq):
            continue
        high_value = 13 if seq == (13, 1, 2, 3, 4) else seq[-1]
        high_card = max(card_id for card_id in card_ids if _card_rank_value(card_id) == high_value)
        return seq_idx, high_card
    return None


def _is_straight_codes(codes: list[str]) -> bool:
    try:
        return _straight_rank(card_codes_to_alpha_ids(codes)) is not None
    except ValueError:
        return False


def _all_straights(hand_codes: list[str]) -> list[list[str]]:
    import itertools

    straights = []
    for combo in itertools.combinations(hand_codes, 5):
        combo_list = list(combo)
        if _is_straight_codes(combo_list):
            straights.append(combo_list)
    return straights


def _action_card_codes(action: dict[str, Any]) -> list[str]:
    return [
        card.get("code")
        for card in action.get("cards") or []
        if isinstance(card, dict) and isinstance(card.get("code"), str)
    ]


def _straight_accessory_cost(card_ids: list[int], rank_key: tuple[int, int]) -> tuple[int, int, tuple[int, ...]]:
    high_card = int(rank_key[1])
    remaining = list(card_ids)
    try:
        remaining.remove(high_card)
    except ValueError:
        pass
    return sum(remaining), sum(card_ids), tuple(sorted(card_ids))


def next_player(player: int) -> int:
    return 1 if player == 4 else player + 1


def current_pass_map(last_player: int | None, passes_since_last_play: int) -> dict[str, bool]:
    passed = {str(player): False for player in range(1, 5)}
    if last_player is None:
        return passed
    for offset in range(1, int(passes_since_last_play) + 1):
        passed[str(((last_player - 1 + offset) % 4) + 1)] = True
    return passed


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PublicStateTracker:
    game_index: int | None = None
    my_hand: list[int] = field(default_factory=list)
    opponent_counts: dict[str, int] = field(
        default_factory=lambda: {"2": 13, "3": 13, "4": 13}
    )
    played_cards: list[int] = field(default_factory=list)
    played_cards_by_player: dict[str, list[int]] = field(default_factory=dict)
    last_hand: list[int] | None = None
    last_player: int | None = None
    control: bool = True
    passed: dict[str, bool] = field(
        default_factory=lambda: {"1": False, "2": False, "3": False, "4": False}
    )
    action_history: list[dict[str, Any]] = field(default_factory=list)
    current_player: int = 1
    _prev_last_hand: list[int] | None = None
    _prev_last_player: int | None = None
    _prev_passes_since: int = 0
    _prev_trick_index: int | None = None

    def reset(self) -> None:
        self.my_hand = []
        self.opponent_counts = {"2": 13, "3": 13, "4": 13}
        self.played_cards = []
        self.played_cards_by_player = {}
        self.last_hand = None
        self.last_player = None
        self.control = True
        self.passed = {"1": False, "2": False, "3": False, "4": False}
        self.action_history = []
        self.current_player = 1
        self._prev_last_hand = None
        self._prev_last_player = None
        self._prev_passes_since = 0
        self._prev_trick_index = None

    def update(self, obs: dict[str, Any]) -> None:
        new_game_index = int(obs.get("game_index", 0))
        observed_hand = sorted(
            card_code_to_alpha_id(card["code"])
            for card in obs.get("self_hand", [])
            if isinstance(card, dict) and isinstance(card.get("code"), str)
        )
        trick_index = _int_or_none(obs.get("trick_index"))
        if self.game_index != new_game_index:
            self.reset()
            self.game_index = new_game_index
            log.info("New game detected (game_index=%s)", new_game_index)
        elif self._looks_like_reused_index_new_round(observed_hand, trick_index):
            self.reset()
            self.game_index = new_game_index
            log.info(
                "New game inferred despite reused game_index=%s trick_index=%s",
                new_game_index,
                trick_index,
            )

        self.my_hand = observed_hand
        self._update_opponent_counts(obs.get("opponents", []))

        turn = obs.get("turn")
        if turn in SEAT_TO_PLAYER:
            self.current_player = SEAT_TO_PLAYER[str(turn)]

        constraint = obs.get("constraint") or {}
        current_last_hand = [
            card_code_to_alpha_id(card["code"])
            for card in constraint.get("last_played_cards", [])
            if isinstance(card, dict) and isinstance(card.get("code"), str)
        ]
        current_last_player = SEAT_TO_PLAYER.get(constraint.get("last_played_by"))
        passes_since = int(constraint.get("passes_since_last_play", 0) or 0)

        play_changed = False
        if current_last_hand and current_last_player is not None:
            if (
                current_last_hand != self._prev_last_hand
                or current_last_player != self._prev_last_player
            ):
                if (
                    self._prev_last_hand
                    and self._prev_last_player == current_last_player
                    and self._prev_passes_since < 3
                ):
                    for offset in range(self._prev_passes_since + 1, 4):
                        player = ((self._prev_last_player - 1 + offset) % 4) + 1
                        self._record_pass(player, control_break=(offset == 3))
                self._record_play(current_last_player, current_last_hand)
                play_changed = True

        if (
            current_last_player is not None
            and passes_since > (0 if play_changed else self._prev_passes_since)
        ):
            start_offset = 1 if play_changed else self._prev_passes_since + 1
            for offset in range(start_offset, passes_since + 1):
                player = ((current_last_player - 1 + offset) % 4) + 1
                self._record_pass(player, control_break=(offset == 3))

        self.last_hand = current_last_hand or None
        self.last_player = current_last_player
        self.control = self.last_hand is None
        self.passed = current_pass_map(self.last_player, passes_since)

        if self.control:
            self.last_player = None

        self._prev_last_hand = list(current_last_hand) if current_last_hand else None
        self._prev_last_player = current_last_player
        self._prev_passes_since = passes_since
        self._prev_trick_index = trick_index

    def _looks_like_reused_index_new_round(
        self,
        observed_hand: list[int],
        trick_index: int | None,
    ) -> bool:
        if self.game_index is None:
            return False
        if len(observed_hand) != 13:
            return False

        hand_refilled = 0 < len(self.my_hand) < 13
        trick_rewound = (
            trick_index is not None
            and trick_index <= 1
            and self._prev_trick_index is not None
            and self._prev_trick_index > trick_index
        )
        hand_changed = set(observed_hand) != set(self.my_hand)
        return hand_refilled or (trick_rewound and hand_changed)

    def _update_opponent_counts(self, opponents: list[dict[str, Any]]) -> None:
        for opponent in opponents:
            if not isinstance(opponent, dict):
                continue
            seat = opponent.get("seat")
            if seat not in SEAT_TO_PLAYER or seat == "self":
                continue
            player = SEAT_TO_PLAYER[str(seat)]
            count = opponent.get("remaining_count")
            if isinstance(count, int):
                self.opponent_counts[str(player)] = count

    def _record_play(self, player: int, hand: list[int]) -> None:
        self.action_history.append(
            {
                "player": player,
                "hand": list(hand),
                "pass": False,
                "forced_skip": False,
                "control_break": False,
                "passed_snapshot": [self.passed[str(idx)] for idx in range(1, 5)],
            }
        )
        self.played_cards.extend(hand)
        self.played_cards_by_player.setdefault(str(player), []).extend(hand)
        self.last_hand = list(hand)
        self.last_player = player
        self.control = False
        self.passed = {"1": False, "2": False, "3": False, "4": False}
        self.current_player = next_player(player)

    def _record_pass(self, player: int, control_break: bool) -> None:
        passed_snapshot = [self.passed[str(idx)] for idx in range(1, 5)]
        self.passed[str(player)] = True
        self.action_history.append(
            {
                "player": player,
                "hand": None,
                "pass": True,
                "forced_skip": False,
                "control_break": control_break,
                "passed_snapshot": passed_snapshot,
            }
        )
        if control_break:
            self.control = True
            self.last_hand = None
            self.last_player = None
            self.passed = {"1": False, "2": False, "3": False, "4": False}
            self.current_player = 1
        else:
            self.current_player = next_player(player)

    def to_ml_observation(self) -> dict[str, Any]:
        return {
            "my_hand": list(self.my_hand),
            "perspective_player": 1,
            "current_player": int(self.current_player),
            "opponent_counts": dict(self.opponent_counts),
            "played_cards": list(self.played_cards),
            "played_cards_by_player": {
                key: list(cards) for key, cards in self.played_cards_by_player.items()
            },
            "last_hand": None if self.last_hand is None else list(self.last_hand),
            "last_player": self.last_player,
            "control": bool(self.control),
            "passed": dict(self.passed),
            "action_history": list(self.action_history),
        }


def choose_legal_decision(
    legal_actions: list[dict[str, Any]],
    logits: Any,
    pass_index: int,
    action_index_from_cards: Callable[[list[int]], int],
    opponents: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    note: str = "model_choice",
) -> dict[str, Any]:
    legal_actions = prepare_legal_actions(legal_actions, opponents or [], observation)
    forced = choose_forced_max_single(legal_actions, opponents or [])
    if forced is not None:
        return forced

    candidates: list[tuple[float, dict[str, Any]]] = []
    for action in legal_actions:
        if action.get("action") == "pass":
            candidates.append((float(logits[pass_index]), action))
            continue

        cards = action.get("cards") or []
        codes = [card.get("code") for card in cards if isinstance(card, dict)]
        if len(codes) not in {1, 2, 5}:
            continue
        try:
            card_ids = sorted(card_code_to_alpha_id(code) for code in codes if isinstance(code, str))
            action_index = action_index_from_cards(card_ids)
        except Exception:
            continue
        candidates.append((float(logits[action_index]), action))

    if not candidates:
        return dict(PASS_DECISION)

    _score, best = max(
        candidates,
        key=lambda item: (
            item[0],
            1 if item[1].get("action") == "play" else 0,
            len(item[1].get("cards") or []),
        ),
    )
    if best.get("action") == "pass":
        return dict(PASS_DECISION)
    return {
        "action": "play",
        "card_codes": [card["code"] for card in best.get("cards", []) if isinstance(card, dict)],
        "combo_type": best.get("combo_type"),
        "note": note,
    }


def legal_action_indices(
    legal_actions: list[dict[str, Any]],
    pass_index: int,
    action_index_from_cards: Callable[[list[int]], int],
    opponents: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
) -> list[int]:
    indices: list[int] = []
    for action in prepare_legal_actions(legal_actions, opponents or [], observation):
        if action.get("action") == "pass":
            indices.append(int(pass_index))
            continue
        cards = action.get("cards") or []
        codes = [card.get("code") for card in cards if isinstance(card, dict)]
        if len(codes) not in {1, 2, 5}:
            continue
        try:
            card_ids = sorted(card_code_to_alpha_id(code) for code in codes if isinstance(code, str))
            indices.append(int(action_index_from_cards(card_ids)))
        except Exception:
            continue
    return indices


def prepare_legal_actions(
    legal_actions: list[dict[str, Any]],
    opponents: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    actions = filter_right_one_single_rule(list(legal_actions), opponents or [])
    actions = canonicalize_equivalent_straights(actions)
    actions = filter_straight_breaking_singles(actions, observation)
    return actions


def canonicalize_equivalent_straights(legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_rank: dict[tuple[int, int], tuple[tuple[int, int, tuple[int, ...]], int]] = {}
    straight_ranks_by_index: dict[int, tuple[int, int]] = {}
    for index, action in enumerate(legal_actions):
        if action.get("action") != "play" or action.get("combo_type") != "straight":
            continue
        codes = _action_card_codes(action)
        try:
            card_ids = card_codes_to_alpha_ids(codes)
        except ValueError:
            continue
        rank_key = _straight_rank(card_ids)
        if rank_key is None:
            continue
        straight_ranks_by_index[index] = rank_key
        cost = _straight_accessory_cost(card_ids, rank_key)
        current = best_by_rank.get(rank_key)
        if current is None or cost < current[0]:
            best_by_rank[rank_key] = (cost, index)

    keep_indices = {
        best_index
        for _cost, best_index in best_by_rank.values()
    }
    filtered = []
    for index, action in enumerate(legal_actions):
        if index not in straight_ranks_by_index or index in keep_indices:
            filtered.append(action)
    return filtered


def filter_straight_breaking_singles(
    legal_actions: list[dict[str, Any]],
    observation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(observation, dict):
        return legal_actions
    constraint = observation.get("constraint") or {}
    if constraint.get("required_combo_type") != "single":
        return legal_actions

    hand_codes = [
        card.get("code")
        for card in observation.get("self_hand", [])
        if isinstance(card, dict) and isinstance(card.get("code"), str)
    ]
    if not _all_straights(hand_codes):
        return legal_actions

    def preserves_straight(action: dict[str, Any]) -> bool:
        codes = _action_card_codes(action)
        if action.get("action") != "play" or action.get("combo_type") != "single" or len(codes) != 1:
            return True
        remaining = list(hand_codes)
        try:
            remaining.remove(codes[0])
        except ValueError:
            return True
        return bool(_all_straights(remaining))

    single_actions = [
        action
        for action in legal_actions
        if action.get("action") == "play"
        and action.get("combo_type") == "single"
        and len(_action_card_codes(action)) == 1
    ]
    if not single_actions or not any(preserves_straight(action) for action in single_actions):
        return legal_actions

    return [
        action
        for action in legal_actions
        if preserves_straight(action)
    ]


def filter_right_one_single_rule(
    legal_actions: list[dict[str, Any]],
    opponents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    right_remaining = None
    for opponent in opponents:
        if not isinstance(opponent, dict):
            continue
        if opponent.get("seat") == "right":
            count = opponent.get("remaining_count")
            if isinstance(count, int):
                right_remaining = count
            break

    if right_remaining != 1:
        return legal_actions

    single_actions: list[tuple[int, dict[str, Any]]] = []
    for action in legal_actions:
        if action.get("action") != "play" or action.get("combo_type") != "single":
            continue
        cards = action.get("cards") or []
        if len(cards) != 1 or not isinstance(cards[0], dict):
            continue
        code = cards[0].get("code")
        if isinstance(code, str):
            single_actions.append((card_code_to_alpha_id(code), action))
    if len(single_actions) <= 1:
        return legal_actions

    max_card_id, _chosen = max(single_actions, key=lambda item: item[0])
    filtered: list[dict[str, Any]] = []
    for action in legal_actions:
        if action.get("action") != "play" or action.get("combo_type") != "single":
            filtered.append(action)
            continue
        cards = action.get("cards") or []
        code = cards[0].get("code") if len(cards) == 1 and isinstance(cards[0], dict) else None
        if isinstance(code, str) and card_code_to_alpha_id(code) == max_card_id:
            filtered.append(action)
    return filtered


def choose_forced_max_single(
    legal_actions: list[dict[str, Any]],
    opponents: list[dict[str, Any]],
) -> dict[str, Any] | None:
    right_remaining = None
    for opponent in opponents:
        if not isinstance(opponent, dict):
            continue
        if opponent.get("seat") == "right":
            count = opponent.get("remaining_count")
            if isinstance(count, int):
                right_remaining = count
            break

    if right_remaining != 1:
        return None

    play_actions = [action for action in legal_actions if action.get("action") == "play"]
    if not play_actions:
        return None
    if any(action.get("combo_type") != "single" for action in play_actions):
        return None

    single_actions = []
    for action in play_actions:
        cards = action.get("cards") or []
        if len(cards) != 1 or not isinstance(cards[0], dict):
            continue
        code = cards[0].get("code")
        if not isinstance(code, str):
            continue
        single_actions.append((card_code_to_alpha_id(code), action))
    if not single_actions:
        return None

    _max_card, chosen = max(single_actions, key=lambda item: item[0])
    return {
        "action": "play",
        "card_codes": [chosen["cards"][0]["code"]],
        "combo_type": "single",
        "note": "override:forced_max_single_right_one_left",
    }


def build_candidate_scores(
    legal_actions: list[dict[str, Any]],
    logits: Any,
    pass_index: int,
    action_index_from_cards: Callable[[list[int]], int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for action in legal_actions:
        record: dict[str, Any] = {
            "action": action.get("action"),
            "combo_type": action.get("combo_type"),
        }
        cards = action.get("cards") or []
        record["card_codes"] = [
            card.get("code") for card in cards if isinstance(card, dict) and isinstance(card.get("code"), str)
        ]
        if action.get("action") == "pass":
            record["action_index"] = pass_index
            record["score"] = float(logits[pass_index])
            candidates.append(record)
            continue
        try:
            card_ids = sorted(card_code_to_alpha_id(code) for code in record["card_codes"])
            action_index = action_index_from_cards(card_ids)
        except Exception as exc:
            record["error"] = str(exc)
            candidates.append(record)
            continue
        record["action_index"] = int(action_index)
        record["score"] = float(logits[action_index])
        candidates.append(record)
    candidates.sort(key=lambda item: item.get("score", float("-inf")), reverse=True)
    return candidates


def build_observation_key(observation: dict[str, Any]) -> dict[str, Any]:
    constraint = observation.get("constraint", {}) or {}
    return {
        "game_index": observation.get("game_index"),
        "trick_index": observation.get("trick_index"),
        "source_seq": observation.get("source_seq"),
        "turn": observation.get("turn"),
        "self_hand_codes": [
            card.get("code")
            for card in observation.get("self_hand", [])
            if isinstance(card, dict) and isinstance(card.get("code"), str)
        ],
        "required_combo_type": constraint.get("required_combo_type"),
        "last_played_codes": [
            card.get("code")
            for card in constraint.get("last_played_cards", [])
            if isinstance(card, dict) and isinstance(card.get("code"), str)
        ],
        "last_played_by": constraint.get("last_played_by"),
        "passes_since_last_play": constraint.get("passes_since_last_play"),
        "opponents": [
            {
                "seat": opp.get("seat"),
                "remaining_count": opp.get("remaining_count"),
            }
            for opp in observation.get("opponents", [])
            if isinstance(opp, dict)
        ],
    }


def align_ml_observation_to_runtime_turn(
    ml_observation: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    aligned = dict(ml_observation)
    turn = observation.get("turn")
    if turn in SEAT_TO_PLAYER:
        aligned["current_player"] = SEAT_TO_PLAYER[str(turn)]
    return aligned


def append_debug_record(record: dict[str, Any]) -> None:
    debug_dir = os.environ.get(DEBUG_ENV_KEY)
    if not debug_dir:
        return
    path = Path(debug_dir) / "model_debug.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _alpha_big2_dir() -> Path:
    return Path(os.environ.get("ALPHA_BIG2_DIR", str(DEFAULT_ALPHA_BIG2_DIR))).resolve()


def _alpha_big2_ckpt(alpha_big2_dir: Path) -> Path:
    return Path(os.environ.get("ALPHA_BIG2_CKPT", str(alpha_big2_dir / DEFAULT_CKPT))).resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid %s=%r; using default %.3f", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid %s=%r; using default %d", name, raw, default)
        return int(default)


def maybe_reexec_with_alpha_python() -> None:
    if os.environ.get(REEXEC_ENV_KEY) == "1":
        return

    alpha_big2_dir = _alpha_big2_dir()
    target_python = Path(
        os.environ.get("ALPHA_BIG2_PYTHON", str(alpha_big2_dir / ".venv/bin/python"))
    ).resolve()
    if not target_python.exists():
        return
    if Path(sys.executable).resolve() == target_python:
        return

    env = os.environ.copy()
    env[REEXEC_ENV_KEY] = "1"
    os.execve(
        str(target_python),
        [str(target_python), str(Path(__file__).resolve())],
        env,
    )


def bootstrap_alpha_big2() -> dict[str, Any]:
    global _ALPHA_CTX
    if _ALPHA_CTX is not None:
        return _ALPHA_CTX

    alpha_big2_dir = _alpha_big2_dir()
    if not alpha_big2_dir.exists():
        raise FileNotFoundError(f"AlphaBig2 repo not found: {alpha_big2_dir}")

    sys.path.insert(0, str(alpha_big2_dir))
    os.chdir(alpha_big2_dir)

    import numpy as np  # noqa: WPS433
    import torch  # noqa: WPS433

    from ML_AB.agents import ModelAgent, RerankAgent  # noqa: WPS433
    from ML_AB.eval import load_model  # noqa: WPS433
    from ML_AB.live_mcts import LiveBeliefMCTSAgent  # noqa: WPS433
    from ML_AB.online import build_public_game  # noqa: WPS433
    import enumerateOptions  # noqa: WPS433

    device_name = os.environ.get("ALPHA_BIG2_DEVICE", "auto")
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

    ckpt_path = _alpha_big2_ckpt(alpha_big2_dir)
    model = load_model(str(ckpt_path), device_name)
    agent_type = os.environ.get("ALPHA_BIG2_AGENT", DEFAULT_AGENT_TYPE).strip().lower()
    if agent_type == "mcts":
        agent = LiveBeliefMCTSAgent(
            model,
            device=device_name,
            time_limit_sec=_env_float("ALPHA_BIG2_MCTS_SECONDS", 1.0),
            determinizations=_env_int("ALPHA_BIG2_MCTS_DETERMINIZATIONS", 4),
            simulations=_env_int("ALPHA_BIG2_MCTS_SIMULATIONS", 10000),
            max_children=_env_int("ALPHA_BIG2_MCTS_MAX_CHILDREN", 64),
            c_puct=_env_float("ALPHA_BIG2_MCTS_C_PUCT", 1.5),
            belief_blend=_env_float("ALPHA_BIG2_MCTS_BELIEF_BLEND", 0.0),
            selection_metric=os.environ.get("ALPHA_BIG2_MCTS_SELECTION", "q"),
            posterior_particles=_env_int("ALPHA_BIG2_MCTS_POSTERIOR_PARTICLES", 24),
            history_likelihood_weight=_env_float("ALPHA_BIG2_MCTS_HISTORY_WEIGHT", 1.0),
            root_warmup_children=_env_int("ALPHA_BIG2_MCTS_ROOT_WARMUP", 12),
            root_q_min_actions=_env_int("ALPHA_BIG2_MCTS_ROOT_Q_MIN_ACTIONS", 2),
            root_q_min_coverage=_env_float("ALPHA_BIG2_MCTS_ROOT_Q_MIN_COVERAGE", 0.7),
            root_q_max_required=_env_int("ALPHA_BIG2_MCTS_ROOT_Q_MAX_REQUIRED", 8),
            root_q_min_margin=_env_float("ALPHA_BIG2_MCTS_ROOT_Q_MIN_MARGIN", 0.01),
            action_value_fallback_weight=_env_float("ALPHA_BIG2_MCTS_ACTION_VALUE_FALLBACK_WEIGHT", 0.0),
        )
    elif agent_type == "rerank":
        agent = RerankAgent(
            model,
            device=device_name,
            temperature=0.0,
            control_five_bonus=_env_float("ALPHA_BIG2_CONTROL_FIVE_BONUS", 1.2),
            card_count_bonus=_env_float("ALPHA_BIG2_CARD_COUNT_BONUS", 0.12),
            finish_bonus=_env_float("ALPHA_BIG2_FINISH_BONUS", 3.0),
            urgent_opponent_count=_env_int("ALPHA_BIG2_URGENT_OPPONENT_COUNT", 3),
            urgent_five_bonus=_env_float("ALPHA_BIG2_URGENT_FIVE_BONUS", 1.0),
            preserve_five_card_penalty=_env_float("ALPHA_BIG2_PRESERVE_FIVE_CARD_PENALTY", 0.35),
        )
    elif agent_type == "model":
        agent = ModelAgent(model, device=device_name, temperature=0.0)
    else:
        log.warning("Unknown ALPHA_BIG2_AGENT=%r; falling back to %s", agent_type, DEFAULT_AGENT_TYPE)
        agent_type = DEFAULT_AGENT_TYPE
        agent = LiveBeliefMCTSAgent(
            model,
            device=device_name,
            time_limit_sec=_env_float("ALPHA_BIG2_MCTS_SECONDS", 1.0),
            selection_metric=os.environ.get("ALPHA_BIG2_MCTS_SELECTION", "q"),
            posterior_particles=_env_int("ALPHA_BIG2_MCTS_POSTERIOR_PARTICLES", 24),
            history_likelihood_weight=_env_float("ALPHA_BIG2_MCTS_HISTORY_WEIGHT", 1.0),
            root_warmup_children=_env_int("ALPHA_BIG2_MCTS_ROOT_WARMUP", 12),
            root_q_min_actions=_env_int("ALPHA_BIG2_MCTS_ROOT_Q_MIN_ACTIONS", 2),
            root_q_min_coverage=_env_float("ALPHA_BIG2_MCTS_ROOT_Q_MIN_COVERAGE", 0.7),
            root_q_max_required=_env_int("ALPHA_BIG2_MCTS_ROOT_Q_MAX_REQUIRED", 8),
            root_q_min_margin=_env_float("ALPHA_BIG2_MCTS_ROOT_Q_MIN_MARGIN", 0.01),
            action_value_fallback_weight=_env_float("ALPHA_BIG2_MCTS_ACTION_VALUE_FALLBACK_WEIGHT", 0.0),
        )
    _ALPHA_CTX = {
        "np": np,
        "torch": torch,
        "agent": agent,
        "agent_type": agent_type,
        "build_public_game": build_public_game,
        "enumerateOptions": enumerateOptions,
        "ckpt_path": ckpt_path,
        "device": device_name,
    }
    log.info("Loaded ML_AB model ckpt=%s device=%s agent=%s", ckpt_path, device_name, agent_type)
    return _ALPHA_CTX


def decide_with_v196(tracker: PublicStateTracker, obs: dict[str, Any]) -> dict[str, Any]:
    if obs.get("turn") != "self":
        return dict(PASS_DECISION)

    ctx = bootstrap_alpha_big2()
    ml_observation = align_ml_observation_to_runtime_turn(tracker.to_ml_observation(), obs)
    game = ctx["build_public_game"](**ml_observation)
    raw_legal_actions = list(obs.get("legal_actions", []))
    prepared_legal_actions = prepare_legal_actions(
        raw_legal_actions,
        opponents=list(obs.get("opponents", [])),
        observation=obs,
    )
    mcts_debug = None
    if ctx.get("agent_type") == "mcts":
        allowed_indices = legal_action_indices(
            legal_actions=prepared_legal_actions,
            pass_index=ctx["enumerateOptions"].passInd,
            action_index_from_cards=ctx["enumerateOptions"].action_index_from_cards,
            opponents=list(obs.get("opponents", [])),
        )
        logits, raw_policy_logits, value, _belief_probs, mcts_debug = ctx["agent"].search(
            game,
            legal_action_indices=allowed_indices,
        )
        decision_note = "mcts_choice"
    else:
        logits, value = ctx["agent"].action_logits(game, 1)
        raw_policy_logits = logits
        decision_note = "model_choice" if ctx.get("agent_type") == "model" else "rerank_choice"
    decision = choose_legal_decision(
        legal_actions=prepared_legal_actions,
        logits=logits,
        pass_index=ctx["enumerateOptions"].passInd,
        action_index_from_cards=ctx["enumerateOptions"].action_index_from_cards,
        opponents=list(obs.get("opponents", [])),
        note=decision_note,
    )
    append_debug_record(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ckpt_path": str(ctx["ckpt_path"]),
            "device": ctx["device"],
            "agent_type": ctx.get("agent_type"),
            "model_value": float(value),
            "mcts": mcts_debug,
            "legal_filter": {
                "raw_actions": len(raw_legal_actions),
                "prepared_actions": len(prepared_legal_actions),
                "removed_actions": max(len(raw_legal_actions) - len(prepared_legal_actions), 0),
            },
            "observation_key": build_observation_key(obs),
            "raw_observation": obs,
            "ml_public_state": ml_observation,
            "candidate_scores": build_candidate_scores(
                legal_actions=prepared_legal_actions,
                logits=logits,
                pass_index=ctx["enumerateOptions"].passInd,
                action_index_from_cards=ctx["enumerateOptions"].action_index_from_cards,
            )[:10],
            "policy_candidate_scores": build_candidate_scores(
                legal_actions=prepared_legal_actions,
                logits=raw_policy_logits,
                pass_index=ctx["enumerateOptions"].passInd,
                action_index_from_cards=ctx["enumerateOptions"].action_index_from_cards,
            )[:10],
            "decision": decision,
        }
    )
    return decision


def main() -> None:
    maybe_reexec_with_alpha_python()
    tracker = PublicStateTracker()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            obs = json.loads(raw)
            tracker.update(obs)
            decision = decide_with_v196(tracker, obs)
            log.info("-> %s %s", decision["action"], decision.get("card_codes", []))
            print(json.dumps(decision), flush=True)
        except Exception as exc:
            log.exception("Inference failed; falling back to pass")
            fallback = dict(PASS_DECISION)
            fallback["note"] = "fallback:inference_error"
            append_debug_record(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "decision": fallback,
                    "raw_observation": locals().get("obs"),
                    "observation_key": build_observation_key(locals().get("obs", {}))
                    if isinstance(locals().get("obs"), dict)
                    else None,
                }
            )
            print(json.dumps(fallback), flush=True)


if __name__ == "__main__":
    main()
