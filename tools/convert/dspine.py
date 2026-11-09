"""Add an explicitly selected DSpine companion without re-encoding the target artifact."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from .model import Model, Parameter
from .qwen3_5 import _Builder, draft_config
from .recipe import Recipe
from .methods import grouped_absmax
from .quantization.nvfp4 import nvfp4_maxabs
from .sources.logical import array_source
from .sources.safetensors import SafetensorsSource


def add_dspine(base_path: Path, draft_path: Path, output: Path,
               projection_format: str = "nvfp4") -> None:
    if projection_format not in ("q8_g32_fp16", "nvfp4"):
        raise ValueError("unsupported DSpine projection format")
    with Artifact(base_path) as base, SafetensorsSource(draft_path) as source:
        if "dspine" in base.directory.components:
            raise ValueError("base already contains DSpine")
        target = base.directory.components["text"]["config"]
        config = draft_config(source.config, target, "dspine")
        for version in ("independent_write_version", "terminal_disabled_version"):
            if source.read_flat("selected_prefix_bridge." + version).item() != 1:
                raise ValueError(f"unsupported DSpine {version}")
        if torch.count_nonzero(source.read_flat("selected_prefix_bridge.out.weight")).item():
            raise ValueError("DSpine terminal-disabled projection must be zero")
        model = Model({"text": {"config": target}, "dspine": {"config": config, "target": "text"}})
        builder = _Builder(model)
        builder.draft(source, config, target, "dspine")
        p = "dspine/bridge/"
        sp = "selected_prefix_bridge."
        builder.add(p + "codebook", source, sp + "codebook", (248320, 1024))
        builder.add(p + "gate_bias", source, sp + "gate.bias", (512,))
        packed = torch.cat(
            [
                source.read_flat(sp + key).reshape(512, 1024)
                for key in ("gate.weight", "predecessor_gate.weight", "prev.weight")
            ]
        )
        model.add(
            Parameter(
                p + "projection",
                (1536, 1024),
                array_source(packed, "packed DSpine bridge"),
                inputs=(p + "codes",),
                residency="dspine",
            )
        )
        for layer in range(5):
            lp = p + f"layers/{layer}/"
            if layer == 4:
                builder.add(
                    lp + "read",
                    source,
                    sp + "read_weight",
                    (1024, 5120),
                    inputs=(lp + "normalized",),
                )
            else:
                builder.add(
                    lp + "read",
                    source,
                    sp + "early_read_weights",
                    (1024, 5120),
                    source_shape=(4, 1024, 5120),
                    offset=layer * 1024 * 5120,
                    inputs=(lp + "normalized",),
                )
            builder.add(
                lp + "write",
                source,
                sp + "feature_write_weights",
                (5120, 512),
                source_shape=(5, 5120, 512),
                offset=layer * 5120 * 512,
                inputs=(lp + "message",),
            )
        recipe = Recipe(model)
        for name, parameter in model.parameters.items():
            if parameter.projection and not name.startswith(p):
                if projection_format == "nvfp4":
                    recipe.assign(name, format="nvfp4", method=nvfp4_maxabs,
                                  activation_policy="A16Only")
                else:
                    recipe.assign(name, format="q8_g32_fp16", method=grouped_absmax)
        for layer in range(5):
            prefix = f"dspine/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)
        prepared = recipe.prepare(device="cpu", rows_per_chunk=256, source_cache_bytes=64 * 1024**2)
        rename = {job.spec.id: "dspine/" + job.spec.id for job in prepared.weights}
        rename.update({spec.id: "dspine/" + spec.id for spec, _ in prepared.auxiliaries})

        def renamed(value):
            if isinstance(value, str):
                return rename.get(value, value)
            if isinstance(value, dict):
                return {k: renamed(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [renamed(v) for v in value]
            return value

        specs = [
            TensorSpec(o.id, o.shape, o.format, o.layout)
            if isinstance(o, TensorObject)
            else ResourceSpec(o.id, o.bytes, o.encoding)
            for o in base.objects
        ]
        specs.extend(replace(spec, id=rename[spec.id]) for spec, _ in prepared.auxiliaries)
        specs.extend(replace(job.spec, id=rename[job.spec.id]) for job in prepared.weights)
        components = deepcopy(base.directory.components)
        components["dspine"] = model.components["dspine"]
        bindings = deepcopy(base.directory.bindings)
        bindings.update(renamed(prepared.bindings))
        uses = deepcopy(list(base.directory.uses)) + renamed(prepared.uses)
        head_use = next(
            deepcopy(u)
            for u in uses
            if u["parameter"] == "text/output_head" and u["input"] == "text/final_hidden"
        )
        head_use["input"] = "dspine/final_hidden"
        uses.append(head_use)
        base_hashes: dict[str, str] = {}
        report = {
            "base": str(base_path),
            "draft": str(draft_path),
            "dspine_projection_format": projection_format,
            "base_objects": base_hashes,
            "status": "writing",
        }
        with ArtifactWriter(
            output,
            specs,
            components=components,
            bindings=bindings,
            uses=uses,
            metadata={**base.directory.metadata, "name": "qwen27-dspine-lab"},
            provenance={**base.directory.provenance, "dspine_source": str(draft_path)},
        ) as writer:
            for obj in base.objects:
                digest = hashlib.sha256()

                def chunks():
                    for chunk in base.iter_object(obj.id):
                        digest.update(chunk)
                        yield chunk

                writer.write_object(obj.id, chunks())
                base_hashes[obj.id] = digest.hexdigest()
            for spec, data in prepared.auxiliaries:
                writer.write_object(rename[spec.id], data)
            for index, job in enumerate(prepared.weights):
                print(f"DSpine {index + 1}/{len(prepared.weights)} {job.parameters[0]}", flush=True)
                job.prepared.produce(TensorOutput(writer, rename[job.spec.id]))
        with Artifact(output) as result:
            for obj in base.objects:
                digest = hashlib.sha256()
                for chunk in result.iter_object(obj.id):
                    digest.update(chunk)
                if digest.hexdigest() != base_hashes[obj.id]:
                    raise RuntimeError(f"base weight changed: {obj.id}")
        report["status"] = "verified"
        Path(str(output) + ".conversion.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--projection-format", choices=("q8_g32_fp16", "nvfp4"),
                        default="nvfp4")
    args = parser.parse_args()
    torch.set_num_threads(8)
    add_dspine(args.base, args.draft, args.out, args.projection_format)
