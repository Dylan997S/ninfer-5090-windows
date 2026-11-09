"""Append optional BF16 DSpine lookup tables without changing existing objects."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.row_split import RowPlanes, dequantize_row_split
from tools.artifact.layouts import row_split_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec, binding_parts
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter


PREFIX = "dspine/bridge/"
HEAD_TABLE = PREFIX + "head_write_table"
SENDER_TABLE = PREFIX + "sender_table"


class BoundRows:
    """Read rows of one logical parameter from its existing physical parent."""

    def __init__(self, artifact: Artifact, name: str, rows: int, columns: int):
        self.artifact = artifact
        self.name = name
        self.rows = rows
        self.columns = columns
        parts = binding_parts(artifact.directory.bindings[name], artifact.by_id, name)
        if len(parts) != 1:
            raise ValueError(f"{name}: expected one row-aligned physical parent")
        object_id, begin, end = parts[0]
        obj = artifact.object(object_id)
        if not isinstance(obj, TensorObject):
            raise ValueError(f"{name}: expected a tensor")
        if end - begin != rows * columns or begin % columns:
            raise ValueError(f"{name}: binding is not the requested complete rows")
        if len(obj.shape) > 1 and obj.shape[-1] != columns:
            raise ValueError(f"{name}: physical row width differs from logical width")
        if obj.format == "bf16" and obj.layout == "contiguous_le_v1":
            self.geometry = None
        elif obj.format == "q8_g32_fp16" and obj.layout == "row_split_k128_v1":
            self.geometry = row_split_geometry(obj.format, obj.shape)
        else:
            raise ValueError(f"{name}: unsupported {obj.format}/{obj.layout}")
        self.object = obj
        self.first_row = begin // columns

    def read(self, begin: int, end: int) -> torch.Tensor:
        if not 0 <= begin < end <= self.rows:
            raise ValueError(f"{self.name}: row range exceeds binding")
        count = end - begin
        first = self.first_row + begin
        obj = self.object
        if self.geometry is None:
            raw = self.artifact.read_range(
                obj.offset + first * self.columns * 2, count * self.columns * 2
            )
            return decode_direct(raw, "bf16", (count, self.columns))
        geometry = self.geometry
        base = self.artifact.read_range(
            obj.offset + geometry.base_offset + first * geometry.base_row_bytes,
            count * geometry.base_row_bytes,
        )
        scales = self.artifact.read_range(
            obj.offset + geometry.scale_offset + first * geometry.scale_row_bytes,
            count * geometry.scale_row_bytes,
        )
        return dequantize_row_split(
            RowPlanes(base, b"", scales, count), obj.format, (count, self.columns)
        )


def add_tables(base_path: Path, output: Path, chunk_rows: int) -> None:
    if not 1 <= chunk_rows <= 4096:
        raise ValueError("chunk_rows must be in [1,4096] to bound temporary memory")
    if base_path.resolve() == output.resolve():
        raise ValueError("output must differ from the source artifact")
    if output.exists():
        raise FileExistsError(output)
    with Artifact(base_path) as base, torch.inference_mode():
        if "dspine" not in base.directory.components:
            raise ValueError("source artifact has no DSpine component")
        if any(name in base.directory.bindings or name in base.by_id
               for name in (HEAD_TABLE, SENDER_TABLE)):
            raise ValueError("source already contains DSpine lookup tables")
        target = base.directory.components["text"]["config"]
        vocab, hidden = int(target["vocab_size"]), int(target["hidden_size"])
        if hidden != 5120:
            raise ValueError("DSpine tables require hidden size 5120")
        head = BoundRows(base, "text/output_head", vocab, hidden)
        norm = BoundRows(base, "dspine/final_norm", 1, hidden).read(0, 1).float()
        write = BoundRows(base, PREFIX + "layers/4/write", hidden, 512).read(0, hidden)
        projection = BoundRows(base, PREFIX + "projection", 1536, 1024).read(0, 1536)
        sender_projection = projection[512:].t().contiguous()
        codebook = BoundRows(base, PREFIX + "codebook", vocab, 1024)
        specs = [
            TensorSpec(obj.id, obj.shape, obj.format, obj.layout)
            if isinstance(obj, TensorObject)
            else ResourceSpec(obj.id, obj.bytes, obj.encoding)
            for obj in base.objects
        ]
        specs.extend((TensorSpec(HEAD_TABLE, (vocab, 512), "bf16", "contiguous_le_v1"),
                      TensorSpec(SENDER_TABLE, (vocab, 1024), "bf16", "contiguous_le_v1")))
        bindings = deepcopy(base.directory.bindings)
        bindings.update({name: {"object": name} for name in (HEAD_TABLE, SENDER_TABLE)})
        base_hashes: dict[str, str] = {}
        table_records: dict[str, dict[str, int | str]] = {}
        report = {
            "base": str(base_path), "output": str(output), "status": "writing",
            "chunk_rows": chunk_rows, "threads": torch.get_num_threads(),
            "base_objects": base_hashes, "tables": table_records,
            "equations": {
                HEAD_TABLE: "bf16(bf16(dequant_bf16(head).float()*norm.float()) @ last_write)",
                SENDER_TABLE: "bf16(bf16(32*codebook.float()) @ cat(predecessor_gate,prev).T)",
            },
            # Includes generous allocator/library headroom, all row tensors, and I/O chunks.
            "estimated_working_bytes": 512 * 1024**2 + chunk_rows * hidden * 64,
        }
        with ArtifactWriter(
            output, specs, components=deepcopy(base.directory.components), bindings=bindings,
            uses=deepcopy(list(base.directory.uses)), metadata=deepcopy(base.directory.metadata),
            provenance={**base.directory.provenance, "dspine_tables_base": str(base_path)},
        ) as writer:
            for obj in base.objects:
                digest = hashlib.sha256()

                def chunks():
                    for chunk in base.iter_object(obj.id):
                        digest.update(chunk)
                        yield chunk

                writer.write_object(obj.id, chunks())
                base_hashes[obj.id] = digest.hexdigest()
            head_output = TensorOutput(writer, HEAD_TABLE)
            sender_output = TensorOutput(writer, SENDER_TABLE)
            for first in range(0, vocab, chunk_rows):
                last = min(first + chunk_rows, vocab)
                normalized_head = (head.read(first, last).float() * norm).to(torch.bfloat16)
                head_output.write_values(first * 512, normalized_head @ write)
                sender_codes = (codebook.read(first, last).float() * 32).to(torch.bfloat16)
                sender_output.write_values(first * 1024, sender_codes @ sender_projection)
                if first == 0 or last == vocab or first // chunk_rows % 16 == 0:
                    print(f"DSpine tables {last}/{vocab}", flush=True)
        with Artifact(output) as result:
            for obj in base.objects:
                digest = hashlib.sha256()
                for chunk in result.iter_object(obj.id):
                    digest.update(chunk)
                if digest.hexdigest() != base_hashes[obj.id]:
                    raise RuntimeError(f"original object changed: {obj.id}")
            for name in (HEAD_TABLE, SENDER_TABLE):
                digest = hashlib.sha256()
                for chunk in result.iter_object(name):
                    digest.update(chunk)
                obj = result.object(name)
                table_records[name] = {"bytes": obj.bytes, "sha256": digest.hexdigest()}
        report["status"] = "verified"
        Path(str(output) + ".conversion.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunk-rows", default=1024, type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    add_tables(args.base, args.out, args.chunk_rows)


if __name__ == "__main__":
    main()
