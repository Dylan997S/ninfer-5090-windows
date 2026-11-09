"""Check sampled DSpine lookup-table rows against float64 matrix equations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.reader import Artifact
from .dspine_tables import BoundRows, HEAD_TABLE, PREFIX, SENDER_TABLE


def compare(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict:
    actual = actual.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"passed": False, "finite": False}
    error = actual - expected
    reference_norm = torch.linalg.vector_norm(expected).item()
    error_norm = torch.linalg.vector_norm(error).item()
    relative = error_norm / reference_norm if reference_norm else error_norm
    return {
        "passed": relative < tolerance,
        "finite": True,
        "relative_l2": relative,
        "max_absolute_error": torch.max(torch.abs(error)).item(),
        "reference_l2": reference_norm,
    }


def verify(base_path: Path, table_path: Path, samples: int, tolerance: float) -> dict:
    if not 2 <= samples <= 64:
        raise ValueError("samples must be in [2,64]")
    if not 0 < tolerance <= 0.005:
        raise ValueError("relative L2 tolerance must be in (0,0.005]")
    with Artifact(base_path) as base, Artifact(table_path) as tables, torch.inference_mode():
        config = base.directory.components["text"]["config"]
        vocab, hidden = int(config["vocab_size"]), int(config["hidden_size"])
        table_config = tables.directory.components["text"]["config"]
        if table_config != config:
            raise ValueError("lookup-table target config differs from the source")
        if hidden != 5120 or samples > vocab:
            raise ValueError("unsupported target geometry")
        ids = [i * (vocab - 1) // (samples - 1) for i in range(samples)]
        head = BoundRows(base, "text/output_head", vocab, hidden)
        norm = BoundRows(base, "dspine/final_norm", 1, hidden).read(0, 1).double()
        write = BoundRows(base, PREFIX + "layers/4/write", hidden, 512).read(0, hidden).double()
        codebook = BoundRows(base, PREFIX + "codebook", vocab, 1024)
        projection = BoundRows(base, PREFIX + "projection", 1536, 1024).read(0, 1536).double()
        stored_head = BoundRows(tables, HEAD_TABLE, vocab, 512)
        stored_sender = BoundRows(tables, SENDER_TABLE, vocab, 1024)
        head_expected, head_actual, sender_expected, sender_actual = [], [], [], []
        head_rows, sender_rows = [], []
        for row in ids:
            # These BF16 casts are observable inputs to the matrix contractions.
            # Each contraction and its reduction are independently evaluated in FP64.
            normalized_head = (head.read(row, row + 1).double() * norm).to(torch.bfloat16)
            expected_head = normalized_head.double() @ write
            exact_code = (32 * codebook.read(row, row + 1).double()).to(torch.bfloat16)
            expected_sender = exact_code.double() @ projection[512:].t()
            actual_head = stored_head.read(row, row + 1)
            actual_sender = stored_sender.read(row, row + 1)
            head_rows.append({"token_id": row, **compare(actual_head, expected_head, tolerance)})
            sender_rows.append({"token_id": row, **compare(actual_sender, expected_sender, tolerance)})
            head_expected.append(expected_head)
            head_actual.append(actual_head)
            sender_expected.append(expected_sender)
            sender_actual.append(actual_sender)
        head_summary = compare(torch.cat(head_actual), torch.cat(head_expected), tolerance)
        sender_summary = compare(torch.cat(sender_actual), torch.cat(sender_expected), tolerance)
        return {
            "base": str(base_path), "tables": str(table_path),
            "sampled_token_ids": ids, "relative_l2_tolerance": tolerance,
            "reference": "FP64 contractions on represented BF16 operands",
            HEAD_TABLE: {"rows": head_rows, "aggregate": head_summary},
            SENDER_TABLE: {"rows": sender_rows, "aggregate": sender_summary},
            "passed": (head_summary["passed"] and sender_summary["passed"]
                       and all(row["passed"] for row in (*head_rows, *sender_rows))),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=8, type=int)
    parser.add_argument("--tolerance", default=0.005, type=float)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = verify(args.base, args.tables, args.samples, args.tolerance)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in (HEAD_TABLE, SENDER_TABLE):
        print(json.dumps({"table": name, **result[name]["aggregate"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
