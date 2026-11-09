#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

DraftWeights bind_draft(Bindings& b, const DraftConfig& config, const TextConfig& target,
                        const TextWeights& weights, const std::string& component) {
    const auto h = target.hidden_size;
    const auto q = config.attention.query_width();
    const auto k = config.attention.key_width();
    const auto d = config.attention.head_dim;
    DraftWeights out;
    out.feature_projection = b.parameter(
        component + "/feature_projection",
        {h, artifact::checked_mul(config.target_layer_ids.size(), h, "target features")},
        {component + "/target_features"});
    out.context_norm    = b.direct(component + "/context_norm", {h});
    out.final_norm      = b.direct(component + "/final_norm", {h});
    out.token_embedding = weights.token_embedding;
    out.output_head     = weights.output_head;
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = component + "/layers/" + std::to_string(i) + "/";
        DraftBlockWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.attention.query =
            b.parameter(p + "attention/query", {q, h}, {p + "query_projection_input"});
        layer.attention.key =
            b.parameter(p + "attention/key", {k, h}, {p + "query_projection_input"});
        layer.attention.value =
            b.parameter(p + "attention/value", {k, h}, {p + "query_projection_input"});
        layer.attention.context_key =
            b.parameter(p + "attention/context_key", {k, h}, {component + "/context_input"});
        layer.attention.context_value =
            b.parameter(p + "attention/context_value", {k, h}, {component + "/context_input"});
        layer.attention.query_norm = b.direct(p + "attention/query_norm", {d});
        layer.attention.key_norm   = b.direct(p + "attention/key_norm", {d});
        layer.attention.output =
            b.parameter(p + "attention/output", {h, q}, {p + "attention_output"});
        layer.mlp = bind_dense(b, h, config.intermediate_size, p, true);
        out.layers.push_back(std::move(layer));
    }
    if (config.dflash2) { bind_dflash2(b, out, config, target); }
    if (config.dspine) {
        auto& bridge      = out.dspine.emplace();
        const auto p      = component + "/bridge/";
        bridge.codebook   = b.direct(p + "codebook", {target.vocab_size, 1024});
        bridge.gate_bias  = b.direct(p + "gate_bias", {512});
        bridge.projection = b.parameter(p + "projection", {1536, 1024}, {p + "codes"});
        const bool has_head_table   = b.binder.contains(p + "head_write_table");
        const bool has_sender_table = b.binder.contains(p + "sender_table");
        if (has_head_table != has_sender_table)
            throw artifact::ArtifactError("DSpine requires both precomputed tables or neither");
        if (has_head_table) {
            bridge.head_write_table = b.direct(p + "head_write_table", {target.vocab_size, 512});
            bridge.sender_table     = b.direct(p + "sender_table", {target.vocab_size, 1024});
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const auto l = p + "layers/" + std::to_string(i) + "/";
            bridge.reads.push_back(b.parameter(l + "read", {1024, h}, {l + "normalized"}));
            bridge.writes.push_back(b.parameter(l + "write", {h, 512}, {l + "message"}));
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
