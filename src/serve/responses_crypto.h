#pragma once
#include <string>
#include <string_view>

namespace ninfer::serve {
// Authenticated, process-local envelopes. Keys never leave the serving process.
std::string encrypt_response_reasoning(std::string_view text);
std::string decrypt_response_reasoning(std::string_view envelope);
} // namespace ninfer::serve
