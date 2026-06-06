from __future__ import annotations

import json
import hashlib
import itertools
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from big2_vision_agent.decision_review import decision_key, observation_key, summarize_step
from big2_vision_agent.session_review import build_rounds, load_json


RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "1", "2"]
STRAIGHT_RANKS = RANK_ORDER[:-1]
SUIT_STRENGTH = {"4": 0, "3": 1, "2": 2, "1": 3}
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


def card_rank(code: str) -> str:
    return str(code)[1]


def card_rank_index(code: str) -> int:
    return RANK_ORDER.index(card_rank(code))


def card_strength(code: str) -> tuple[int, int]:
    text = str(code)
    return card_rank_index(text), SUIT_STRENGTH.get(text[0], -1)


def card_alpha_id(code: str) -> int:
    text = str(code)
    return (BIG2_TO_ALPHA_RANK[text[1]] - 1) * 4 + BIG2_TO_ALPHA_SUIT[text[0]]


def card_rank_value(code: str) -> int:
    return BIG2_TO_ALPHA_RANK[str(code)[1]]


def straight_rank(card_codes: list[str]) -> tuple[int, int] | None:
    if len(card_codes) != 5:
        return None
    values = [card_rank_value(code) for code in card_codes]
    if len(set(values)) != 5:
        return None
    value_set = set(values)
    for seq_index, seq in enumerate(STRAIGHT_SEQUENCES):
        if value_set != set(seq):
            continue
        high_value = 13 if seq == (13, 1, 2, 3, 4) else seq[-1]
        high_card = max(card_alpha_id(code) for code in card_codes if card_rank_value(code) == high_value)
        return seq_index, high_card
    return None


def straight_accessory_cost(card_codes: list[str], rank_key: tuple[int, int]) -> tuple[int, int, tuple[int, ...]]:
    card_ids = [card_alpha_id(code) for code in card_codes]
    remaining = list(card_ids)
    try:
        remaining.remove(int(rank_key[1]))
    except ValueError:
        pass
    return sum(remaining), sum(card_ids), tuple(sorted(card_ids))


def all_straights(card_codes: list[str]) -> list[list[str]]:
    return [
        list(combo)
        for combo in itertools.combinations(card_codes, 5)
        if straight_rank(list(combo)) is not None
    ]


def is_pair_twos(card_codes: list[str]) -> bool:
    return len(card_codes) == 2 and all(card_rank(code) == "2" for code in card_codes)


def contains_straight(card_codes: list[str]) -> bool:
    ranks = {card_rank(code) for code in card_codes if card_rank(code) in STRAIGHT_RANKS}
    for start in range(0, len(STRAIGHT_RANKS) - 4):
        if set(STRAIGHT_RANKS[start : start + 5]).issubset(ranks):
            return True
    return False


def played_single_breaks_straight(hand_codes: list[str], decision_codes: list[str]) -> bool:
    if len(decision_codes) != 1 or not all_straights(hand_codes):
        return False
    remaining = list(hand_codes)
    try:
        remaining.remove(decision_codes[0])
    except ValueError:
        return False
    return not all_straights(remaining)


def legal_single_preserves_straight(hand_codes: list[str], candidate_codes: list[str]) -> bool:
    if len(candidate_codes) != 1:
        return False
    remaining = list(hand_codes)
    try:
        remaining.remove(candidate_codes[0])
    except ValueError:
        return False
    return bool(all_straights(remaining))


def has_five_card_action(candidate_scores: list[dict[str, Any]]) -> bool:
    return any(len(row.get("card_codes") or []) == 5 for row in candidate_scores)


def five_card_candidates(model_row: dict[str, Any]) -> list[dict[str, Any]]:
    policy_candidates = [
        row
        for row in model_row.get("policy_candidate_scores", []) or []
        if row.get("action") == "play" and len(row.get("card_codes") or []) == 5
    ]
    if policy_candidates:
        return policy_candidates
    return [
        row
        for row in model_row.get("candidate_scores", []) or []
        if row.get("action") == "play" and len(row.get("card_codes") or []) == 5
    ]


def lowest_legal_single(candidate_scores: list[dict[str, Any]]) -> str | None:
    singles = [
        row["card_codes"][0]
        for row in candidate_scores
        if row.get("action") == "play"
        and row.get("combo_type") == "single"
        and len(row.get("card_codes") or []) == 1
    ]
    if not singles:
        return None
    return min(singles, key=card_rank_index)


def highest_legal_single(candidate_scores: list[dict[str, Any]]) -> str | None:
    singles = [
        row["card_codes"][0]
        for row in candidate_scores
        if row.get("action") == "play"
        and row.get("combo_type") == "single"
        and len(row.get("card_codes") or []) == 1
    ]
    if not singles:
        return None
    return max(singles, key=card_strength)


def right_opponent_has_one_card(observation_key: dict[str, Any]) -> bool:
    for opponent in observation_key.get("opponents") or []:
        if not isinstance(opponent, dict):
            continue
        if opponent.get("seat") == "right" and opponent.get("remaining_count") == 1:
            return True
    return False


def _action_ref(candidate: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "action": candidate.get("action"),
        "combo_type": candidate.get("combo_type"),
        "card_codes": list(candidate.get("card_codes") or []),
    }
    if isinstance(candidate.get("action_index"), int):
        ref["action_index"] = int(candidate["action_index"])
    if isinstance(candidate.get("score"), (int, float)):
        ref["score"] = float(candidate["score"])
    return ref


def single_breaks_straight_preference(model_row: dict[str, Any]) -> dict[str, Any] | None:
    observation_key = model_row.get("observation_key", {}) or {}
    decision = model_row.get("decision", {}) or {}
    decision_codes = list(decision.get("card_codes") or [])
    hand_codes = list(observation_key.get("self_hand_codes") or [])
    if not played_single_breaks_straight(hand_codes, decision_codes):
        return None

    preferred = [
        _action_ref(candidate)
        for candidate in model_row.get("candidate_scores", []) or []
        if candidate.get("action") == "play"
        and candidate.get("combo_type") == "single"
        and legal_single_preserves_straight(hand_codes, list(candidate.get("card_codes") or []))
    ]
    if not preferred:
        return None
    return {
        "reason": "single_breaks_straight",
        "bad_actions": [
            {
                "action": decision.get("action"),
                "combo_type": decision.get("combo_type"),
                "card_codes": decision_codes,
            }
        ],
        "preferred_actions": preferred,
    }


def straight_accessory_preference(model_row: dict[str, Any]) -> dict[str, Any] | None:
    decision = model_row.get("decision", {}) or {}
    decision_codes = list(decision.get("card_codes") or [])
    if decision.get("combo_type") != "straight" or len(decision_codes) != 5:
        return None
    rank_key = straight_rank(decision_codes)
    if rank_key is None:
        return None

    same_rank_candidates = []
    for candidate in model_row.get("candidate_scores", []) or []:
        if candidate.get("action") != "play" or candidate.get("combo_type") != "straight":
            continue
        candidate_codes = list(candidate.get("card_codes") or [])
        if straight_rank(candidate_codes) == rank_key:
            same_rank_candidates.append(candidate)
    if not same_rank_candidates:
        return None

    best_cost = min(
        straight_accessory_cost(list(candidate.get("card_codes") or []), rank_key)
        for candidate in same_rank_candidates
    )
    chosen_cost = straight_accessory_cost(decision_codes, rank_key)
    if best_cost >= chosen_cost:
        return None

    preferred = [
        _action_ref(candidate)
        for candidate in same_rank_candidates
        if straight_accessory_cost(list(candidate.get("card_codes") or []), rank_key) == best_cost
    ]
    if not preferred:
        return None
    return {
        "reason": "straight_nonminimal_accessory",
        "bad_actions": [
            {
                "action": decision.get("action"),
                "combo_type": decision.get("combo_type"),
                "card_codes": decision_codes,
            }
        ],
        "preferred_actions": preferred,
    }


def five_card_available_preference(model_row: dict[str, Any]) -> dict[str, Any] | None:
    observation_key = model_row.get("observation_key", {}) or {}
    if observation_key.get("required_combo_type") is not None:
        return None
    decision = model_row.get("decision", {}) or {}
    decision_codes = list(decision.get("card_codes") or [])
    if not (0 < len(decision_codes) < 5):
        return None

    preferred = [_action_ref(candidate) for candidate in five_card_candidates(model_row)]
    if not preferred:
        return None
    return {
        "reason": "five_card_available_but_non_five_played",
        "bad_actions": [
            {
                "action": decision.get("action"),
                "combo_type": decision.get("combo_type"),
                "card_codes": decision_codes,
            }
        ],
        "preferred_actions": preferred,
    }


def endgame_lowest_single_preference(model_row: dict[str, Any]) -> dict[str, Any] | None:
    observation_key = model_row.get("observation_key", {}) or {}
    if observation_key.get("required_combo_type") is not None:
        return None
    if right_opponent_has_one_card(observation_key):
        return None

    hand_codes = list(observation_key.get("self_hand_codes") or [])
    decision = model_row.get("decision", {}) or {}
    decision_codes = list(decision.get("card_codes") or [])
    if len(hand_codes) > 3 or len(decision_codes) != 1:
        return None

    candidate_scores = model_row.get("candidate_scores", []) or []
    lowest = lowest_legal_single(candidate_scores)
    if lowest is None or decision_codes[0] == lowest:
        return None

    preferred = [
        _action_ref(candidate)
        for candidate in candidate_scores
        if candidate.get("action") == "play"
        and candidate.get("combo_type") == "single"
        and list(candidate.get("card_codes") or []) == [lowest]
    ]
    if not preferred:
        return None
    return {
        "reason": "endgame_single_not_lowest",
        "bad_actions": [
            {
                "action": decision.get("action"),
                "combo_type": decision.get("combo_type"),
                "card_codes": decision_codes,
            }
        ],
        "preferred_actions": preferred,
    }


def preference_label(model_row: dict[str, Any]) -> dict[str, Any] | None:
    for builder in (
        straight_accessory_preference,
        five_card_available_preference,
        endgame_lowest_single_preference,
    ):
        label = builder(model_row)
        if label is not None:
            return label
    return None


def suspicious_tags(model_row: dict[str, Any], step: dict[str, Any] | None) -> list[str]:
    tags: list[str] = []
    observation_key = model_row.get("observation_key", {}) or {}
    decision = model_row.get("decision", {}) or {}
    candidate_scores = model_row.get("candidate_scores", []) or []
    decision_codes = list(decision.get("card_codes") or [])
    hand_codes = list(observation_key.get("self_hand_codes") or [])
    required_combo = observation_key.get("required_combo_type")

    if step and not step.get("decision_matches_model"):
        tags.append("decision_mismatch")
    if step and step.get("result_reason"):
        tags.append(f"executor_{step.get('result_reason')}")
    note = decision.get("note")
    if note == "fallback:inference_error":
        tags.append("fallback_inference_error")
    if has_five_card_action(candidate_scores) and 0 < len(decision_codes) < 5 and required_combo is None:
        tags.append("five_card_available_but_non_five_played")
    if played_single_breaks_straight(hand_codes, decision_codes):
        tags.append("single_breaks_straight")
    if straight_accessory_preference(model_row):
        tags.append("straight_nonminimal_accessory")
    if required_combo is None and is_pair_twos(decision_codes):
        tags.append("control_pair_twos")
    if right_opponent_has_one_card(observation_key) and len(decision_codes) == 1:
        highest = highest_legal_single(candidate_scores)
        if highest is not None and decision_codes[0] != highest:
            tags.append("right_one_single_not_max")
    forced_max_single = isinstance(note, str) and note.startswith("override:forced_max_single")
    if (
        not forced_max_single
        and required_combo is None
        and len(hand_codes) <= 3
        and len(decision_codes) == 1
    ):
        lowest = lowest_legal_single(candidate_scores)
        if lowest is not None and decision_codes[0] != lowest:
            tags.append("endgame_single_not_lowest")
    return tags


def _rounds_by_game_index(artifact_dir: Path, model_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    timeline_path = artifact_dir / "game_timeline.json"
    if not timeline_path.exists():
        return {}
    rounds = build_rounds(load_json(timeline_path))
    game_indices: list[int] = []
    for row in model_rows:
        game_index = (row.get("observation_key", {}) or {}).get("game_index")
        if isinstance(game_index, int) and game_index not in game_indices:
            game_indices.append(game_index)
    return {
        game_index: rounds[idx]
        for idx, game_index in enumerate(game_indices)
        if idx < len(rounds)
    }


def _round_remaining_cards_by_actor(round_info: dict[str, Any]) -> dict[str, list[str]]:
    remaining: dict[str, list[str]] = {}
    for entry in round_info.get("entries", []) or []:
        actor = entry.get("actor")
        if isinstance(actor, str):
            remaining[actor] = list(entry.get("remaining_cards") or [])
    return remaining


def _load_model_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    path = artifact_dir / "model_debug.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _matched_steps_for_model_rows(
    artifact_dir: Path,
    model_rows: list[dict[str, Any]],
) -> list[dict[str, Any] | None]:
    action_log_path = artifact_dir / "action_log.json"
    if not action_log_path.exists():
        return [None for _ in model_rows]

    action_rows = load_json(action_log_path)
    action_steps = [
        row
        for row in action_rows
        if isinstance(row, dict) and row.get("step") == "agent_decision"
    ]
    action_rows_by_observation: dict[tuple, deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    for index, action_row in enumerate(action_steps):
        action_rows_by_observation[observation_key(action_row.get("observation"))].append((index, action_row))

    matched_steps: list[dict[str, Any] | None] = []
    for model_row in model_rows:
        candidates = action_rows_by_observation.get(observation_key(model_row.get("raw_observation")))
        if not candidates:
            matched_steps.append(None)
            continue

        selected_index = None
        model_decision_key = decision_key(model_row.get("decision"))
        for candidate_index, (_step_index, action_row) in enumerate(candidates):
            if decision_key(action_row.get("decision")) == model_decision_key:
                selected_index = candidate_index
                break
        if selected_index is None:
            step_index, action_row = candidates.popleft()
        else:
            step_index, action_row = candidates[selected_index]
            del candidates[selected_index]
        matched_steps.append(summarize_step(step_index, action_row, model_row))

    return matched_steps


def _artifact_id(artifact_dir: Path) -> str:
    if artifact_dir.name == "autoplay_agent" and artifact_dir.parent.name:
        return artifact_dir.parent.name
    return artifact_dir.name


def _repo_root_for_artifact(artifact_dir: Path) -> Path:
    artifact_dir = artifact_dir.resolve()
    for parent in [artifact_dir, *artifact_dir.parents]:
        if parent.name == "artifacts" and parent.parent.exists():
            return parent.parent
    return artifact_dir.parent


def _file_sha256(path_text: object) -> str | None:
    if not isinstance(path_text, str):
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_exclusion_reason(row: dict[str, Any]) -> str | None:
    if row.get("executed_ok") is not True:
        return "not_executed"
    if row.get("decision_matches_model") is not True:
        return "decision_mismatch"
    if not isinstance(row.get("round_self_score"), int):
        return "missing_round_score"
    tags = set(row.get("tags") or [])
    if "fallback_inference_error" in tags:
        return "fallback_inference_error"
    executor_tags = sorted(tag for tag in tags if isinstance(tag, str) and tag.startswith("executor_"))
    if executor_tags:
        return executor_tags[0]
    return None


def build_training_rows(artifact_dir: str | Path) -> list[dict[str, Any]]:
    artifact_dir = Path(artifact_dir)
    model_rows = _load_model_rows(artifact_dir)
    steps = _matched_steps_for_model_rows(artifact_dir, model_rows)
    rounds_by_game = _rounds_by_game_index(artifact_dir, model_rows)

    rows = []
    artifact_id = _artifact_id(artifact_dir)
    for index, model_row in enumerate(model_rows):
        observation_key = model_row.get("observation_key", {}) or {}
        game_index = observation_key.get("game_index")
        round_info = rounds_by_game.get(game_index, {}) if isinstance(game_index, int) else {}
        step = steps[index] if index < len(steps) else None
        row = {
            "artifact_dir": str(artifact_dir),
            "artifact_id": artifact_id,
            "decision_index": index,
            "game_index": game_index,
            "source_seq": observation_key.get("source_seq"),
            "ckpt_path": model_row.get("ckpt_path"),
            "agent_type": model_row.get("agent_type"),
            "device": model_row.get("device"),
            "observation_key": observation_key,
            "ml_public_state": model_row.get("ml_public_state"),
            "candidate_scores": model_row.get("candidate_scores", []),
            "decision": model_row.get("decision"),
            "executed_ok": None if step is None else step.get("result_ok"),
            "executor_sent": None if step is None else step.get("result_sent"),
            "executor_note": None if step is None else step.get("result_note"),
            "decision_matches_model": None if step is None else step.get("decision_matches_model"),
            "round_self_score": round_info.get("self_score"),
            "round_self_won": round_info.get("self_won"),
            "round_self_remaining_count": round_info.get("self_remaining_count"),
            "round_winner_actor": round_info.get("winner_actor"),
            "round_entries": round_info.get("entries", []),
            "round_remaining_cards_by_actor": _round_remaining_cards_by_actor(round_info),
        }
        row["decision_id"] = f"{artifact_id}:{index}"
        row["ckpt_sha256"] = _file_sha256(row.get("ckpt_path"))
        row["tags"] = suspicious_tags(model_row, step)
        row["preference_label"] = preference_label(model_row)
        row["training_exclusion_reason"] = training_exclusion_reason(row)
        row["training_usable"] = row["training_exclusion_reason"] is None
        rows.append(row)
    return rows


def render_summary(rows: list[dict[str, Any]]) -> str:
    tag_counts = Counter(tag for row in rows for tag in row.get("tags", []))
    exclusion_counts = Counter(
        row.get("training_exclusion_reason")
        for row in rows
        if row.get("training_exclusion_reason")
    )
    score_values = [row.get("round_self_score") for row in rows if isinstance(row.get("round_self_score"), int)]
    usable_score_values = [
        row.get("round_self_score")
        for row in rows
        if row.get("training_usable") and isinstance(row.get("round_self_score"), int)
    ]
    lines = [
        "# Training Dataset Summary",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Training usable rows: `{sum(1 for row in rows if row.get('training_usable'))}`",
        f"- Training excluded rows: `{sum(1 for row in rows if not row.get('training_usable'))}`",
        f"- Decisions with tags: `{sum(1 for row in rows if row.get('tags'))}`",
        f"- Distinct rounds: `{len({row.get('game_index') for row in rows})}`",
        f"- Mean attached self score: `{(sum(score_values) / len(score_values)):.3f}`" if score_values else "- Mean attached self score: `None`",
        f"- Mean usable self score: `{(sum(usable_score_values) / len(usable_score_values)):.3f}`" if usable_score_values else "- Mean usable self score: `None`",
        "",
        "## Tags",
        "",
    ]
    if tag_counts:
        for tag, count in tag_counts.most_common():
            lines.append(f"- `{tag}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Training Exclusions", ""])
    if exclusion_counts:
        for reason, count in exclusion_counts.most_common():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def write_training_export(artifact_dir: str | Path) -> tuple[Path, Path]:
    artifact_dir = Path(artifact_dir)
    rows = build_training_rows(artifact_dir)
    dataset_path = artifact_dir / "training_dataset.jsonl"
    summary_path = artifact_dir / "training_summary.md"
    dataset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path.write_text(render_summary(rows), encoding="utf-8")
    append_live_training_corpus(artifact_dir, rows)
    return dataset_path, summary_path


def append_live_training_corpus(
    artifact_dir: str | Path,
    rows: list[dict[str, Any]],
    corpus_path: str | Path | None = None,
) -> Path:
    artifact_dir = Path(artifact_dir)
    if corpus_path is None:
        corpus_path = _repo_root_for_artifact(artifact_dir) / "data" / "live_training_corpus.jsonl"
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    incoming_ids = {
        row["decision_id"]
        for row in rows
        if isinstance(row.get("decision_id"), str)
    }
    usable_rows = [
        row
        for row in rows
        if isinstance(row.get("decision_id"), str) and row.get("training_usable")
    ]
    existing_rows: list[dict[str, Any]] = []
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision_id = row.get("decision_id")
                if isinstance(decision_id, str) and decision_id in incoming_ids:
                    continue
                existing_rows.append(row)

    with corpus_path.open("w", encoding="utf-8") as fh:
        for row in existing_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in usable_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return corpus_path
