from typing import Any, Set


def _print_event(event: dict, _printed: Set[str], max_length: int = 1500) -> None:
    """
    Pretty-print a single LangGraph stream event with safety guards.

    - Shows the latest dialog state if present.
    - Prints the last message in the event (if a list is provided).
    - Avoids duplicate printing via the `_printed` id set.
    - Truncates overly long message representations for readability.
    - Falls back gracefully when a message lacks `id` or `pretty_repr`.

    Args:
        event (dict): A single event dict emitted by graph.stream(..., stream_mode="values").
                      Expected keys include "dialog_state" and/or "messages".
        _printed (set[str]): A mutable set of printed message ids to avoid duplicates.
        max_length (int): Maximum number of characters to print for the message body.

    Returns:
        None. This function prints to stdout and updates `_printed` in place.
    """
    # 1) Dialog state line
    current_state = event.get("dialog_state")
    if current_state:
        try:
            # `current_state` is usually a list of node names; show the last.
            print("State:", current_state[-1])
        except Exception:
            # Be tolerant if the structure is not as expected
            print("State:", current_state)

    # 2) Extract the message (often a list with the newest at the end)
    message: Any = event.get("messages")
    if message is None:
        return

    if isinstance(message, list) and message:
        message = message[-1]  # take the last message

    # 3) Determine a stable "id" to deduplicate prints
    #    Use message.id if available; otherwise synthesize a fingerprint.
    msg_id = getattr(message, "id", None)
    if not msg_id:
        content = getattr(message, "content", None)
        role = getattr(message, "type", type(message).__name__)
        msg_id = f"{role}:{hash(str(content))}"

    if msg_id in _printed:
        return

    # 4) Build a readable representation
    #    Prefer `pretty_repr(html=True)` if available; otherwise fallback to str(message).
    try:
        if hasattr(message, "pretty_repr"):
            msg_repr = message.pretty_repr(html=True)
        else:
            msg_repr = str(message)
    except Exception:
        # Absolute fallback
        msg_repr = str(getattr(message, "content", message))

    # 5) Truncate if necessary
    if isinstance(msg_repr, str) and len(msg_repr) > max_length:
        msg_repr = msg_repr[:max_length] + " ... (truncated)"

    # 6) Print and record as printed
    print(msg_repr)
    _printed.add(msg_id)
