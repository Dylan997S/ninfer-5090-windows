#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ninfer/ops/dspine.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/linear.h"
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;

namespace {
float rounded(float x) { return bf16_to_f32(f32_to_bf16(x)); }

std::vector<double> read_f32(const void* ptr, std::size_t count) {
    auto values = from_device<float>(ptr, count);
    return {values.begin(), values.end()};
}

std::vector<float> values(std::size_t count, unsigned seed) {
    std::vector<float> result(count);
    fill_uniform(result, seed, -.25f, .25f);
    round_to_bf16(result);
    return result;
}

int near(const char* name, const std::vector<double>& actual, const std::vector<double>& expected,
         double tolerance = .005) {
    double error = 0, norm = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) return 1;
        error += std::pow(actual[i] - expected[i], 2);
        norm += expected[i] * expected[i];
    }
    if (std::sqrt(error / std::max(norm, 1e-20)) <= tolerance) return 0;
    std::cerr << name << " relative error " << std::sqrt(error / norm) << '\n';
    return 1;
}

int bridge_case(int batch, int width = 16) {
    const int rows = width * batch, h = 5120, c = 1024, r = 512;
    auto raw      = values(rows * h, 19);
    auto codes    = values(rows * c, 21);
    auto codebook = values(32 * c, 23);
    auto packed   = values(rows * 3 * r, 29);
    auto bias     = values(r, 31);
    auto delta    = values(rows * h, 37);
    std::vector<int> anchors(batch);
    std::iota(anchors.begin(), anchors.end(), 3);
    auto dr = to_device_bf16(raw), dc = to_device_bf16(codes), db = to_device_bf16(codebook);
    auto dp = to_device_bf16(packed), dg = to_device_bf16(bias), dd = to_device_bf16(delta);
    auto da = to_device_i32(anchors);
    GuardedDeviceBuffer normalized(raw.size() * 2), rms(rows * 4), message(rows * r * 2);
    Tensor tr(dr.span(), DType::BF16, {h, rows}), tc(dc.span(), DType::BF16, {c, rows});
    Tensor tb(db.span(), DType::BF16, {c, 32}), tp(dp.span(), DType::BF16, {3 * r, rows});
    Tensor tg(dg.span(), DType::BF16, {r}), td(dd.span(), DType::BF16, {h, rows});
    Tensor ta(da.span(), DType::I32, {batch});
    Tensor tn(normalized.span(), DType::BF16, {h, rows}), ts(rms.span(), DType::FP32, {rows});
    Tensor tm(message.span(), DType::BF16, {r, rows});
    const auto e = cuda_environment().default_execution;
    ops::dspine_prepare_read(tr, tn, ts, width, 1e-6f, e);
    ops::dspine_normalize_codes(tc, tb, ta, width, 1e-6f, e);
    ops::dspine_message(tp, tg, width, tm, e);
    ops::dspine_finish(td, ts, width, tr, e);
    cuda_synchronize();
    std::vector<double> rn(raw.size()), rr(rows), rc(codes.size()), rm(rows * r), rf(raw.size());
    for (int row = 0; row < rows; ++row) {
        double sum = 0;
        for (int j = 0; j < h; ++j) sum += double(raw[row * h + j]) * raw[row * h + j];
        rr[row] = std::sqrt(sum / h);
        for (int j = 0; j < h; ++j) {
            rn[row * h + j] = raw[row * h + j] / (rr[row] + 1e-6);
            rf[row * h + j] =
                row % width ? raw[row * h + j] + rr[row] * delta[row * h + j] : raw[row * h + j];
        }
        sum = 0;
        for (int j = 0; j < c; ++j) sum += double(codes[row * c + j]) * codes[row * c + j];
        for (int j = 0; j < c; ++j)
            rc[row * c + j] = row % width ? 32 * codes[row * c + j] / (std::sqrt(sum) + 1e-6)
                                       : 32 * codebook[anchors[row / width] * c + j];
        for (int j = 0; j < r; ++j) {
            rm[row * r + j] =
                row % width ? packed[(row - 1) * 3 * r + 2 * r + j] /
                               (1 + std::exp(-double(packed[row * 3 * r + j] +
                                                     packed[(row - 1) * 3 * r + r + j] + bias[j])))
                         : 0;
        }
    }
    int failures = near("read", from_device_bf16(normalized.data(), rn.size()), rn);
    failures += near("rms", read_f32(rms.data(), rows), rr, 1e-5);
    failures += near("codes", from_device_bf16(dc, codes.size()), rc);
    failures += near("message", from_device_bf16(message.data(), rm.size()), rm);
    failures += near("finish", from_device_bf16(dr, raw.size()), rf);
    return failures + normalized.verify_guards("normalized") + rms.verify_guards("rms") +
           message.verify_guards("message");
}

int path_case(int width = 16, int batch = 1, bool penalties = false) {
    constexpr int h = 5120, vocab = 32;
    const int rows = width * batch, count = (width - 1) * batch;
    auto raw = values(rows * h, 103), delta = values(16 * count * h, 107), norm = values(h, 109);
    std::vector<float> rms(rows, .4f);
    std::vector<signed char> head(vocab * h);
    for (std::size_t i = 0; i < head.size(); ++i)
        head[i] = static_cast<signed char>(int(i * 13 % 127) - 63);
    const auto scale_bits = __half_as_ushort(__float2half(.015625f));
    std::vector<std::uint16_t> scales(vocab * h / 32, scale_bits);
    std::vector<int> candidates(count * 16);
    for (int i = 0; i < count; ++i)
        for (int c = 0; c < 16; ++c) candidates[i * 16 + c] = (c * 7 + i) % vocab;
    auto dr = to_device_bf16(raw), dd = to_device_bf16(delta), dn = to_device_bf16(norm);
    auto ds = to_device_f32(rms), dh = to_device(head), dscale = to_device(scales),
         dc = to_device_i32(candidates);
    GuardedDeviceBuffer scores(count * 256 * 4), drafts(count * 4);
    Tensor tr(dr.span(), DType::BF16, {h, rows}), td(dd.span(), DType::BF16, {h, count * 16});
    Tensor tn(dn.span(), DType::BF16, {h}), ts(ds.span(), DType::FP32, {rows});
    Tensor tc(dc.span(), DType::I32, {16, count}), tout(scores.span(), DType::FP32, {256, count});
    Tensor tpath(drafts.span(), DType::I32, {width - 1, batch});
    Weight w;
    w.device = dr.device();
    w.qtype  = QType::Q8_G32_FP16;
    w.layout = QuantLayout::RowSplit;
    w.n      = vocab;
    w.k      = h;
    w.group  = 32;
    w.qdata  = dh.data();
    w.scales = dscale.data();
    std::vector<int> counts(batch * vocab);
    for (int b = 0; b < batch; ++b)
        for (int v = 0; v < vocab; ++v) counts[b * vocab + v] = (v + b * 7) % 4;
    auto dcounts = to_device_i32(counts);
    std::vector<ops::SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].presence_penalty = b % 3 == 1 ? -1.5f : 1.5f;
        configs[b].frequency_penalty = b % 3 == 2 ? -.5f : .5f;
        configs[b].token_counts = b % 3 == 1 ? nullptr :
            static_cast<const int*>(dcounts.data()) + b * vocab;
    }
    auto dconfigs = to_device(configs);
    ops::dspine_greedy_path(tr, ts, td, tn, w, tc, width, tout, tpath,
                            cuda_environment().default_execution,
                            penalties ? static_cast<const ops::SamplingConfig*>(dconfigs.data()) : nullptr);
    cuda_synchronize();
    std::vector<double> reference(count * 256);
    std::vector<int> expected(count);
    int predecessor = 0;
    for (int i = 0; i < count; ++i) {
        if (i % (width - 1) == 0) predecessor = 0;
        const int original = i / (width - 1) * width + i % (width - 1) + 1;
        for (int p = 0; p < 16; ++p)
            for (int c = 0; c < 16; ++c) {
                double sum = 0, squared = 0;
                const int token = candidates[i * 16 + c];
                for (int j = 0; j < h; ++j) {
                    const double x = double(raw[original * h + j]) +
                        double(rms[original]) * delta[(i * 16 + p) * h + j];
                    sum += rounded(head[token * h + j] * .015625f) * double(norm[j]) * x;
                    squared += x * x;
                }
                if (penalties) sum /= std::sqrt(squared / h + 1e-6);
                reference[(i * 16 + p) * 16 + c] = sum;
            }
        auto adjusted = [&](int c) {
            double value = reference[(i * 16 + predecessor) * 16 + c];
            if (!penalties) return value;
            const int b = i / (width - 1), token = candidates[i * 16 + c];
            const auto& cfg = configs[b];
            int occurrences = cfg.token_counts ? counts[b * vocab + token] : 0;
            for (int earlier = b * (width - 1); earlier < i; ++earlier)
                occurrences += expected[earlier] == token;
            return value - (occurrences ? cfg.presence_penalty : 0.0f) -
                cfg.frequency_penalty * occurrences;
        };
        int best = 0;
        for (int c = 1; c < 16; ++c) {
            const auto value    = adjusted(c);
            const auto previous = adjusted(best);
            if (value > previous ||
                (value == previous && candidates[i * 16 + c] < candidates[i * 16 + best]))
                best = c;
        }
        expected[i] = candidates[i * 16 + best];
        predecessor = best;
    }
    int failures      = near("scores", read_f32(scores.data(), reference.size()), reference, 1e-5);
    const auto actual = from_device<int>(drafts.data(), count);
    if (actual != expected) {
        std::cerr << "DSpine serial path mismatch\n";
        ++failures;
    }
    if (from_device<int>(dcounts.data(), counts.size()) != counts) ++failures;
    return failures + scores.verify_guards("scores") + drafts.verify_guards("drafts");
}

int sender_case() {
    constexpr int width = 16, batch = 2, positions = 30, count = positions * 16;
    auto codebook = values(32 * 1024, 151);
    auto receiver = values(width * batch * 1536, 157);
    auto sender = values(count * 1536, 163);
    auto bias = values(512, 167);
    std::vector<int> candidates(positions * 16), anchors{3, 29};
    for (int i = 0; i < count; ++i) candidates[i] = (i * 7) % 32;
    auto db = to_device_bf16(codebook), dr = to_device_bf16(receiver);
    auto ds = to_device_bf16(sender), dg = to_device_bf16(bias);
    auto dc = to_device_i32(candidates), da = to_device_i32(anchors);
    GuardedDeviceBuffer codes(count * 1024 * 2), message(count * 512 * 2);
    Tensor tb(db.span(), DType::BF16, {1024, 32}), tr(dr.span(), DType::BF16, {1536, 32});
    Tensor ts(ds.span(), DType::BF16, {1536, count}), tg(dg.span(), DType::BF16, {512});
    Tensor tc(dc.span(), DType::I32, {16, positions}), ta(da.span(), DType::I32, {batch});
    Tensor to(codes.span(), DType::BF16, {1024, count}), tm(message.span(), DType::BF16, {512, count});
    auto e = cuda_environment().default_execution;
    ops::dspine_sender_codes(tb, tc, ta, width, to, e);
    ops::dspine_refine_message(tr, ts, tg, width, tm, e);
    cuda_synchronize();
    std::vector<double> expected_codes(count * 1024), expected_message(count * 512);
    for (int row = 0; row < count; ++row) {
        const int p = row / 16, step = p % 15, original = p / 15 * 16 + step + 1;
        const int token = step == 0 ? anchors[p / 15] : candidates[(p - 1) * 16 + row % 16];
        for (int c = 0; c < 1024; ++c) expected_codes[row * 1024 + c] = 32 * codebook[token * 1024 + c];
        for (int r = 0; r < 512; ++r) expected_message[row * 512 + r] = sender[row * 1536 + 1024 + r] /
            (1 + std::exp(-(double(receiver[original * 1536 + r]) + sender[row * 1536 + 512 + r] + bias[r])));
    }
    return near("sender codes", from_device_bf16(codes.data(), expected_codes.size()), expected_codes) +
           near("refined message", from_device_bf16(message.data(), expected_message.size()), expected_message) +
           codes.verify_guards("codes") + message.verify_guards("message");
}

int projection_case(int n, int k, int rows) {
    auto x       = values(rows * k, 131);
    auto weights = values(n * k, 137);
    auto dx = to_device_bf16(x), dw = to_device_bf16(weights);
    GuardedDeviceBuffer output(n * rows * 2);
    Tensor tx(dx.span(), DType::BF16, {k, rows}), ty(output.span(), DType::BF16, {n, rows});
    Weight w;
    w.device   = dx.device();
    w.qtype    = QType::BF16;
    w.layout   = QuantLayout::Contiguous;
    w.ndim     = 2;
    w.n        = n;
    w.k        = k;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.payload = w.qdata = dw.data();
    w.payload_bytes     = dw.size();
    ops::linear(tx, w, ty, cuda_environment().default_execution);
    cuda_synchronize();
    std::vector<double> expected(n * rows);
    for (int t = 0; t < rows; ++t)
        for (int i = 0; i < n; ++i) {
            double value = 0;
            for (int j = 0; j < k; ++j) value += double(x[t * k + j]) * weights[i * k + j];
            expected[t * n + i] = value;
        }
    return near("bridge projection", from_device_bf16(output.data(), expected.size()), expected) +
           output.verify_guards("projection");
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("dspine/bridge/b1", [] { return bridge_case(1); });
    suite.add("dspine/bridge/b8", [] { return bridge_case(8); });
    suite.add("dspine/bridge/w2/b1", [] { return bridge_case(1, 2); });
    suite.add("dspine/bridge/w8/b8", [] { return bridge_case(8, 8); });
    suite.add("dspine/greedy_path", [] { return path_case(); });
    suite.add("dspine/greedy_path/w2/b1", [] { return path_case(2, 1); });
    suite.add("dspine/greedy_path/w8/b8", [] { return path_case(8, 8); });
    suite.add("dspine/penalties/w2/b1", [] { return path_case(2, 1, true); });
    suite.add("dspine/penalties/w16/b1", [] { return path_case(16, 1, true); });
    suite.add("dspine/penalties/w8/b8", [] { return path_case(8, 8, true); });
    suite.add("dspine/senders", sender_case);
    suite.add("dspine/read_projection", [] { return projection_case(1024, 5120, 16); });
    suite.add("dspine/gate_projection", [] { return projection_case(1536, 1024, 128); });
    suite.add("dspine/write_projection", [] { return projection_case(5120, 512, 240); });
    suite.add("dspine/write_projection_small", [] { return projection_case(5120, 512, 16); });
    suite.add("dspine/write_projection_batch", [] { return projection_case(5120, 512, 128); });
    return suite.main(argc, argv);
}
