#pragma once

#include "models/common/qwen/vision/parameters.h"

#include "core/weight_bank.h"
#include "models/common/weights.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct AttentionWeights {
    WeightId query, key, gate, value;
    WeightId query_norm, key_norm, output;
};

struct GdnWeights {
    WeightId query, key, value, z;
    WeightId a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

struct DenseWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    ExpertBankId bank;
    WeightId router, shared_score;
    std::vector<DenseWeights> experts;
    DenseWeights shared;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    std::variant<AttentionWeights, GdnWeights> mixer;
    std::variant<DenseWeights, MoeWeights> ffn;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    WeightUseId output_head_use;
    std::vector<BlockWeights> layers;
};

struct MtpWeights {
    WeightId input_projection, embedding_norm, hidden_norm, final_norm;
    BlockWeights layer;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct DraftAttentionWeights {
    // The draft attention consumes one fused query+key+value parent, but the artifact only
    // has to declare the three logical parameters: parameters.cpp assembles the parent with
    // ops::prepare_projection_weights. Requiring a fused parameter of its own would
    // reject every upstream-generated artifact, since the converter groups the three.
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
};

struct DynamicConvWeights {
    WeightId base_kernel, kernel_projection;
};

struct DraftBlockWeights {
    WeightId input_norm, post_attention_norm;
    DraftAttentionWeights attention;
    DenseWeights mlp;
    std::optional<DynamicConvWeights> attention_conv, mlp_conv;
};

struct SelectorWeights {
    WeightId hidden_projection, predecessor_codebook, successor_codebook;
};

struct DraftWeights {
    struct DSpine {
        WeightId codebook, projection, gate_bias;
        std::optional<WeightId> head_write_table, sender_table;
        std::vector<WeightId> reads, writes;
    };
    WeightId feature_projection, context_norm, final_norm;
    std::vector<DraftBlockWeights> layers;
    std::optional<SelectorWeights> selector;
    std::optional<DSpine> dspine;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct ProposalWeights {
    WeightId head;
    std::optional<WeightId> token_ids;
    std::uint32_t rows = 0;
    std::vector<std::int32_t> global_token_ids;
};

// Handles refer to the frozen model's weight array. No artifact ID lookup is needed in execution.
struct ModelWeights {
    TextWeights text;
    std::optional<qwen::VisionWeights> vision;
    std::optional<MtpWeights> mtp;
    std::optional<DraftWeights> draft;
    std::optional<ProposalWeights> proposal;
};

} // namespace ninfer::models::qwen3_5
