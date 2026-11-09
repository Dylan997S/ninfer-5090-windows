#include "ninfer/ops/dspine.h"
#include "ops/op_tester.h"
#include "ops/suite.h"

#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int Hidden     = 5120;
constexpr int Rank       = 512;
constexpr int Candidates = 16;
constexpr int Vocab      = 32;

std::vector<float> values(std::size_t count, unsigned seed) {
    std::vector<float> result(count);
    fill_uniform(result, seed, -.25f, .25f);
    round_to_bf16(result);
    return result;
}

int compare(const char* label, const std::vector<float>& actual,
            const std::vector<double>& expected) {
    double squared_error = 0, squared_norm = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i])) return 1;
        squared_error += std::pow(double(actual[i]) - expected[i], 2);
        squared_norm += expected[i] * expected[i];
    }
    const double error = std::sqrt(squared_error / std::max(squared_norm, 1e-20));
    if (error < 2e-5) return 0;
    std::cerr << label << " relative error " << error << '\n';
    return 1;
}

int table_case(int width, int batch) {
    const int rows = width * batch, positions = (width - 1) * batch;
    auto raw = values(rows * Hidden, 101), norm = values(Hidden, 103);
    auto receiver = values(rows * 3 * Rank, 107), bias = values(Rank, 109);
    auto senders = values(Vocab * 2 * Rank, 113), products = values(Vocab * Rank, 127);
    std::vector<float> rms(rows);
    for (int row = 0; row < rows; ++row) rms[row] = .1f + float(row % 7) * .1f;
    std::vector<signed char> head(Vocab * Hidden);
    for (std::size_t i = 0; i < head.size(); ++i)
        head[i] = static_cast<signed char>(int(i * 13 % 127) - 63);
    const auto scale_bits = __half_as_ushort(__float2half(.015625f));
    std::vector<std::uint16_t> scales(Vocab * Hidden / 32, scale_bits);
    std::vector<int> candidates(positions * Candidates), anchors(batch);
    for (int position = 0; position < positions; ++position)
        for (int candidate = 0; candidate < Candidates; ++candidate)
            candidates[position * Candidates + candidate] = (candidate * 7 + position * 3) % Vocab;
    for (int request = 0; request < batch; ++request) anchors[request] = (request * 11 + 3) % Vocab;

    auto dr = to_device_bf16(raw), dn = to_device_bf16(norm);
    auto dq = to_device_bf16(receiver), db = to_device_bf16(bias);
    auto ds = to_device_bf16(senders), dt = to_device_bf16(products);
    auto drm = to_device_f32(rms), dh = to_device(head), dsc = to_device(scales);
    auto dc = to_device_i32(candidates), da = to_device_i32(anchors);
    GuardedDeviceBuffer base(positions * Candidates * 4),
        scores(positions * Candidates * Candidates * 4);
    GuardedDeviceBuffer drafts(positions * 4);
    Tensor tr(dr.span(), DType::BF16, {Hidden, rows}), tn(dn.span(), DType::BF16, {Hidden});
    Tensor tq(dq.span(), DType::BF16, {3 * Rank, rows}), tb(db.span(), DType::BF16, {Rank});
    Tensor ts(ds.span(), DType::BF16, {2 * Rank, Vocab}), tt(dt.span(), DType::BF16, {Rank, Vocab});
    Tensor tm(drm.span(), DType::FP32, {rows}), tc(dc.span(), DType::I32, {Candidates, positions});
    Tensor ta(da.span(), DType::I32, {batch}),
        tbase(base.span(), DType::FP32, {Candidates, positions});
    Tensor tscore(scores.span(), DType::FP32, {Candidates * Candidates, positions});
    Tensor tdraft(drafts.span(), DType::I32, {width - 1, batch});
    Weight weight;
    weight.device = dr.device();
    weight.qtype  = QType::Q8_G32_FP16;
    weight.layout = QuantLayout::RowSplit;
    weight.n      = Vocab;
    weight.k      = Hidden;
    weight.group  = 32;
    weight.qdata  = dh.data();
    weight.scales = dsc.data();
    ops::dspine_table_greedy_path(tr, tm, tq, tb, ts, tt, tn, weight, tc, ta, width, tbase, tscore,
                                  tdraft, cuda_environment().default_execution);
    cuda_synchronize();

    // Independently evaluate the published table equation in double precision. Tables
    // are represented BF16 inputs; unlike the direct-write path, messages remain FP32.
    std::vector<double> expected_base(positions * Candidates),
        expected_scores(positions * Candidates * Candidates);
    for (int position = 0; position < positions; ++position) {
        const int step     = position % (width - 1);
        const int original = position / (width - 1) * width + step + 1;
        for (int candidate = 0; candidate < Candidates; ++candidate) {
            const int token = candidates[position * Candidates + candidate];
            double sum      = 0;
            for (int h = 0; h < Hidden; ++h) {
                const float dequantized =
                    bf16_to_f32(f32_to_bf16(head[token * Hidden + h] * .015625f));
                sum += double(dequantized) * norm[h] * raw[original * Hidden + h];
            }
            expected_base[position * Candidates + candidate] = sum;
        }
        for (int predecessor = 0; predecessor < Candidates; ++predecessor) {
            const int sender = step ? candidates[(position - 1) * Candidates + predecessor]
                                    : anchors[position / (width - 1)];
            std::vector<double> message(Rank);
            for (int r = 0; r < Rank; ++r) {
                const double gate = double(receiver[original * 3 * Rank + r]) +
                                    senders[sender * 2 * Rank + r] + bias[r];
                message[r]        = senders[sender * 2 * Rank + Rank + r] / (1 + std::exp(-gate));
            }
            for (int candidate = 0; candidate < Candidates; ++candidate) {
                const int token = candidates[position * Candidates + candidate];
                double dot      = 0;
                for (int r = 0; r < Rank; ++r) dot += message[r] * products[token * Rank + r];
                expected_scores[(position * Candidates + predecessor) * Candidates + candidate] =
                    expected_base[position * Candidates + candidate] + rms[original] * dot;
            }
        }
    }
    std::vector<int> expected_drafts(positions);
    for (int request = 0; request < batch; ++request) {
        int predecessor = 0;
        for (int step = 0; step < width - 1; ++step) {
            const int position = request * (width - 1) + step;
            const double* row =
                expected_scores.data() + (position * Candidates + predecessor) * Candidates;
            int best = 0;
            for (int candidate = 1; candidate < Candidates; ++candidate)
                if (row[candidate] > row[best] ||
                    (row[candidate] == row[best] && candidates[position * Candidates + candidate] <
                                                        candidates[position * Candidates + best]))
                    best = candidate;
            expected_drafts[position] = candidates[position * Candidates + best];
            predecessor               = best;
        }
    }
    int failures =
        compare("table base", from_device<float>(base.data(), expected_base.size()), expected_base);
    failures += compare("table scores", from_device<float>(scores.data(), expected_scores.size()),
                        expected_scores);
    if (from_device<int>(drafts.data(), positions) != expected_drafts) {
        std::cerr << "DSpine table chain mismatch\n";
        ++failures;
    }
    return failures + base.verify_guards("table base") + scores.verify_guards("table scores") +
           drafts.verify_guards("table drafts");
}

int post_write_norm_case(int rows) {
    constexpr float eps = 1e-6f;
    constexpr float raw_pattern[]{.5f, 1.5f, -2.f, .75f};
    constexpr float norm_pattern[]{.70703125f, 1.1328125f, .8984375f, 1.3046875f};
    std::vector<float> raw(rows * Hidden), norm(Hidden);
    for (int h = 0; h < Hidden; ++h) norm[h] = norm_pattern[h % 4];
    for (int row = 0; row < rows; ++row)
        for (int h = 0; h < Hidden; ++h)
            raw[row * Hidden + h] = std::ldexp(raw_pattern[h % 4], row % 5 - 2);
    auto dr = to_device_bf16(raw), dn = to_device_bf16(norm);
    GuardedDeviceBuffer output(raw.size() * 2);
    Tensor tr(dr.span(), DType::BF16, {Hidden, rows}), tn(dn.span(), DType::BF16, {Hidden});
    Tensor to(output.span(), DType::BF16, {Hidden, rows});
    ops::dspine_post_write_norm(tr, tn, eps, to, cuda_environment().default_execution);
    cuda_synchronize();
    const auto actual = from_device_bf16(output.data(), raw.size());
    int mismatches = 0, mutation_mismatches = 0;
    // Binary-exact square sums avoid accumulation ambiguity. Chosen values are
    // away from BF16 ties and make removing the first rounding observably fail.
    for (int row = 0; row < rows; ++row) {
        double square_sum = 0;
        for (int h = 0; h < Hidden; ++h)
            square_sum += double(raw[row * Hidden + h]) * raw[row * Hidden + h];
        const double inverse = 1.0 / std::sqrt(square_sum / Hidden + double(eps));
        for (int h = 0; h < Hidden; ++h) {
            const auto offset      = row * Hidden + h;
            const float normalized = bf16_to_f32(f32_to_bf16(float(raw[offset] * inverse)));
            const float expected   = bf16_to_f32(f32_to_bf16(normalized * norm[h]));
            const float single_round =
                bf16_to_f32(f32_to_bf16(float(raw[offset] * inverse * norm[h])));
            mismatches += actual[offset] != expected;
            mutation_mismatches += single_round != expected;
        }
    }
    if (mismatches || !mutation_mismatches) {
        std::cerr << "post-write norm mismatches=" << mismatches
                  << " one-round control mismatches=" << mutation_mismatches << '\n';
        return 1;
    }
    ops::dspine_post_write_norm(tr, tn, eps, tr, cuda_environment().default_execution);
    cuda_synchronize();
    if (from_device_bf16(dr, raw.size()) != actual) {
        std::cerr << "post-write in-place norm differs\n";
        return 1;
    }
    return output.verify_guards("post-write norm");
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("dspine/table/w2_b1", [] { return table_case(2, 1); });
    suite.add("dspine/table/w2_b8", [] { return table_case(2, 8); });
    suite.add("dspine/table/w8_b1", [] { return table_case(8, 1); });
    suite.add("dspine/table/w8_b8", [] { return table_case(8, 8); });
    suite.add("dspine/table/w16_b1", [] { return table_case(16, 1); });
    suite.add("dspine/table/w16_b8", [] { return table_case(16, 8); });
    suite.add("dspine/post_write_norm/rows2", [] { return post_write_norm_case(2); });
    suite.add("dspine/post_write_norm/rows16", [] { return post_write_norm_case(16); });
    suite.add("dspine/post_write_norm/rows128", [] { return post_write_norm_case(128); });
    return suite.main(argc, argv);
}
