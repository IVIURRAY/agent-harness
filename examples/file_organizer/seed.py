"""Create 10 messy sample text files in the demo directory."""

from __future__ import annotations

import shutil
from pathlib import Path

SAMPLES = {
    "invoice_march_acme.txt": "INVOICE #4471\nAcme Ltd\nAmount due: £1,240.00\nDue: 31 March",
    "trip_ideas_portugal.txt": "Surf trip ideas: Ericeira, Peniche, Sagres.\nCampervan from Lisbon, 10 days in June.",
    "meeting_notes_q3_planning.txt": "Q3 planning\n- Ship RAG eval harness\n- Hire 2 engineers\n- Reduce p95 latency",
    "receipt_climbing_shoes.txt": "Receipt\nClimbing shoes x1  £95.00\nPaid by card 12/05",
    "recipe_feijoada.txt": "Feijoada\nBlack beans, pork shoulder, chorizo, bay leaves. Slow cook 3h.",
    "notes_langgraph_checkpoints.txt": "LangGraph: checkpointers persist graph state per thread_id; use for HITL interrupts.",
    "invoice_april_hosting.txt": "INVOICE\nHosting provider - April\nAmount: $48.00\nStatus: paid",
    "packing_list_alps_hike.txt": "Alps hut-to-hut packing: shell, down jacket, poles, headtorch, blister kit.",
    "standup_2026_11_10.txt": "Standup 10 Nov\nYesterday: tracing for agent loop\nToday: approval gates\nBlockers: none",
    "recipe_pao_de_queijo.txt": "Pão de queijo\nTapioca flour, milk, oil, eggs, parmesan. Bake 20 min at 200C.",
}


def seed(directory: Path, reset: bool = True) -> Path:
    if reset and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in SAMPLES.items():
        (directory / name).write_text(body + "\n", encoding="utf-8")
    return directory


if __name__ == "__main__":
    path = seed(Path("demo_files"))
    print(f"Seeded {len(SAMPLES)} files in {path.resolve()}")
