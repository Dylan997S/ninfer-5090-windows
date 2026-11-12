#include "serve/responses_crypto.h"
#include "serve/request_validation.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#    include <windows.h>
#    include <bcrypt.h>
#else
#    include <openssl/evp.h>
#    include <openssl/rand.h>
#endif

namespace ninfer::serve {
namespace {
using Bytes = std::vector<unsigned char>;

void check(bool ok) {
    if (!ok) throw std::runtime_error("reasoning encryption failed");
}

void random_bytes(unsigned char* bytes, std::size_t count) {
#ifdef _WIN32
    check(BCryptGenRandom(nullptr, bytes, static_cast<ULONG>(count),
                          BCRYPT_USE_SYSTEM_PREFERRED_RNG) >= 0);
#else
    check(RAND_bytes(bytes, static_cast<int>(count)) == 1);
#endif
}

const std::array<unsigned char, 32>& process_key() {
    static const auto key = [] {
        std::array<unsigned char, 32> value{};
        random_bytes(value.data(), value.size());
        return value;
    }();
    return key;
}

bool crypt(Bytes& input, Bytes& output, unsigned char* nonce, unsigned char* tag, bool encrypt) {
    auto key = process_key();
    output.resize(input.size() + 16);
#ifdef _WIN32
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    check(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_AES_ALGORITHM, nullptr, 0) >= 0);

    struct AlgorithmGuard {
        BCRYPT_ALG_HANDLE handle;

        ~AlgorithmGuard() { BCryptCloseAlgorithmProvider(handle, 0); }
    } algorithm_guard{algorithm};

    check(BCryptSetProperty(algorithm, BCRYPT_CHAINING_MODE,
                            reinterpret_cast<PUCHAR>(const_cast<wchar_t*>(BCRYPT_CHAIN_MODE_GCM)),
                            sizeof(BCRYPT_CHAIN_MODE_GCM), 0) >= 0);
    BCRYPT_KEY_HANDLE handle = nullptr;
    check(BCryptGenerateSymmetricKey(algorithm, &handle, nullptr, 0, key.data(),
                                     static_cast<ULONG>(key.size()), 0) >= 0);

    struct KeyGuard {
        BCRYPT_KEY_HANDLE handle;

        ~KeyGuard() { BCryptDestroyKey(handle); }
    } key_guard{handle};

    BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info;
    BCRYPT_INIT_AUTH_MODE_INFO(info);
    info.pbNonce  = nonce;
    info.cbNonce  = 12;
    info.pbTag    = tag;
    info.cbTag    = 16;
    ULONG written = 0;
    const auto status =
        encrypt
            ? BCryptEncrypt(handle, input.data(), static_cast<ULONG>(input.size()), &info, nullptr,
                            0, output.data(), static_cast<ULONG>(output.size()), &written, 0)
            : BCryptDecrypt(handle, input.data(), static_cast<ULONG>(input.size()), &info, nullptr,
                            0, output.data(), static_cast<ULONG>(output.size()), &written, 0);
    if (status < 0) return false;
    output.resize(written);
#else
    std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)> ctx(EVP_CIPHER_CTX_new(),
                                                                        EVP_CIPHER_CTX_free);
    check(ctx != nullptr);
    check(EVP_CipherInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, key.data(), nonce,
                            encrypt ? 1 : 0) == 1);
    if (!encrypt) check(EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, 16, tag) == 1);
    int written = 0, final = 0;
    check(EVP_CipherUpdate(ctx.get(), output.data(), &written, input.data(),
                           static_cast<int>(input.size())) == 1);
    if (EVP_CipherFinal_ex(ctx.get(), output.data() + written, &final) != 1) return false;
    output.resize(static_cast<std::size_t>(written + final));
    if (encrypt) check(EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, 16, tag) == 1);
#endif
    return true;
}
} // namespace

std::string encrypt_response_reasoning(std::string_view text) {
    std::array<unsigned char, 12> nonce{};
    std::array<unsigned char, 16> tag{};
    random_bytes(nonce.data(), nonce.size());
    Bytes input(text.begin(), text.end()), output;
    check(crypt(input, output, nonce.data(), tag.data(), true));
    std::string result   = "ninfer-gcm-v1.";
    constexpr char hex[] = "0123456789abcdef";
    const auto append    = [&](const auto& bytes) {
        for (auto byte : bytes) {
            result += hex[byte >> 4];
            result += hex[byte & 15];
        }
    };
    append(nonce);
    append(tag);
    append(output);
    return result;
}

std::string decrypt_response_reasoning(std::string_view envelope) {
    const auto invalid = [] {
        bad_request("invalid reasoning envelope or envelope belongs to another server process",
                    "input", "invalid_encrypted_content");
    };
    constexpr std::string_view prefix = "ninfer-gcm-v1.";
    if (!envelope.starts_with(prefix)) invalid();
    envelope.remove_prefix(prefix.size());
    if (envelope.size() < 56 || envelope.size() % 2 != 0) invalid();
    Bytes bytes;
    bytes.reserve(envelope.size() / 2);
    const auto nibble = [&](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        invalid();
        return 0;
    };
    for (std::size_t i = 0; i < envelope.size(); i += 2)
        bytes.push_back(
            static_cast<unsigned char>((nibble(envelope[i]) << 4) | nibble(envelope[i + 1])));
    Bytes input(bytes.begin() + 28, bytes.end()), output;
    if (!crypt(input, output, bytes.data(), bytes.data() + 12, false)) invalid();
    return std::string(output.begin(), output.end());
}
} // namespace ninfer::serve
