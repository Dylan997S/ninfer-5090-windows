#pragma once
#include <nlohmann/json.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {
struct BackgroundResponse {
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<bool> cancelled{false};
    nlohmann::json response;
    std::vector<std::string> events;
    std::size_t event_bytes = 0;
    std::size_t byte_limit  = 0;
    bool streaming          = false;
    bool done               = false;
    bool executing          = false;

    void append(const std::vector<std::string>& batch);
    void finish(nlohmann::json value, std::string terminal = {});
    void fail(const std::string& status, const std::string& message, const std::string& code);
    void cancel();
    nlohmann::json snapshot();
};

// Owns bounded process-local jobs and joins every worker before Engine teardown.
class BackgroundResponses {
public:
    BackgroundResponses(std::size_t max_records, std::size_t max_bytes, std::size_t workers);
    ~BackgroundResponses();
    std::shared_ptr<BackgroundResponse>
    submit(const std::string& id, nlohmann::json initial, bool streaming, std::size_t input_bytes,
           std::function<void(std::shared_ptr<BackgroundResponse>)> run);
    std::shared_ptr<BackgroundResponse> get(const std::string& id);
    bool erase(const std::string& id);
    void stop();
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::unordered_map<std::string, std::shared_ptr<BackgroundResponse>> records_;
    std::deque<std::string> order_;

    struct Task {
        std::shared_ptr<BackgroundResponse> job;
        std::function<void()> run;
    };

    std::deque<Task> queue_;
    std::vector<std::thread> workers_;
    std::size_t max_records_;
    std::size_t slot_bytes_;
    bool stopping_ = false;
};
} // namespace ninfer::serve
