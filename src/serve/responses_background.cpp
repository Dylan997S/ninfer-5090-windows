#include "serve/responses_background.h"
#include "serve/request_validation.h"
#include <algorithm>

namespace ninfer::serve {
void BackgroundResponse::append(const std::vector<std::string>& batch) {
    std::lock_guard lock(mutex);
    std::size_t added = 0;
    for (const auto& event : batch) added += event.size() + sizeof(std::string);
    if (added + event_bytes > byte_limit - std::min<std::size_t>(byte_limit, 4096))
        throw ApiException({.status  = 507,
                            .type    = "server_error",
                            .message = "background event buffer capacity exceeded",
                            .code    = "response_buffer_capacity"});
    events.insert(events.end(), batch.begin(), batch.end());
    event_bytes += added;
    changed.notify_all();
}

void BackgroundResponse::finish(nlohmann::json value, std::string terminal) {
    std::lock_guard lock(mutex);
    if (event_bytes + value.dump().size() + terminal.size() > byte_limit)
        throw ApiException({.status  = 507,
                            .type    = "server_error",
                            .message = "background response capacity exceeded",
                            .code    = "response_buffer_capacity"});
    if (!terminal.empty()) events.push_back(std::move(terminal));
    response = std::move(value);
    done     = true;
    changed.notify_all();
}

void BackgroundResponse::fail(const std::string& status, const std::string& message,
                              const std::string& code) {
    std::lock_guard lock(mutex);
    if (done) return;
    response = {
        {"id", response.at("id")},
        {"object", "response"},
        {"model", response.at("model")},
        {"created_at", response.at("created_at")},
        {"status", status},
        {"background", true},
        {"output", nlohmann::json::array()},
        {"usage", nullptr},
        {"error", status == "cancelled"
                      ? nlohmann::json(nullptr)
                      : nlohmann::json{{"code", code}, {"message", message.substr(0, 512)}}}};
    if (streaming) {
        const std::string type = status == "cancelled" ? "response.failed" : "response." + status;
        nlohmann::json event{
            {"type", type}, {"sequence_number", events.size()}, {"response", response}};
        events.push_back("event: " + type + "\ndata: " + event.dump() + "\n\n");
    }
    done = true;
    changed.notify_all();
}

void BackgroundResponse::cancel() {
    cancelled = true;
    bool queued;
    {
        std::lock_guard lock(mutex);
        queued = !executing && !done;
    }
    if (queued) fail("cancelled", "", "");
}

nlohmann::json BackgroundResponse::snapshot() {
    std::lock_guard lock(mutex);
    return response;
}

BackgroundResponses::BackgroundResponses(std::size_t max_records, std::size_t max_bytes,
                                         std::size_t workers)
    : max_records_(std::min(max_records, std::max<std::size_t>(1, max_bytes / (16ULL << 20)))),
      slot_bytes_(max_records_ ? max_bytes / max_records_ : 0) {
    for (std::size_t i = 0; i < std::min(max_records_, std::max<std::size_t>(1, workers)); ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> work;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                    if (queue_.empty()) return;
                    work = std::move(queue_.front().run);
                    queue_.pop_front();
                }
                work();
            }
        });
    }
}

BackgroundResponses::~BackgroundResponses() { stop(); }

void BackgroundResponses::stop() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        for (auto& [id, job] : records_) job->cancel();
    }
    ready_.notify_all();
    for (auto& worker : workers_)
        if (worker.joinable()) worker.join();
}

std::shared_ptr<BackgroundResponse>
BackgroundResponses::submit(const std::string& id, nlohmann::json initial, bool streaming,
                            std::size_t input_bytes,
                            std::function<void(std::shared_ptr<BackgroundResponse>)> run) {
    std::lock_guard lock(mutex_);
    // Cancelled queued tasks must release their captured input before another job is accepted.
    std::erase_if(queue_, [](const Task& task) { return task.job->cancelled.load(); });
    if (max_records_ == 0 || stopping_ || slot_bytes_ < 8192)
        throw ApiException({.status  = 503,
                            .type    = "server_error",
                            .message = "background storage unavailable",
                            .code    = "background_unavailable"});
    if (input_bytes + initial.dump().size() > slot_bytes_ / 2)
        throw ApiException({.status  = 413,
                            .message = "background input exceeds bounded job capacity",
                            .code    = "background_input_capacity"});
    if (records_.size() >= max_records_) {
        auto expired = std::find_if(order_.begin(), order_.end(), [&](const auto& key) {
            auto entry = records_.at(key);
            std::lock_guard job_lock(entry->mutex);
            return entry->done;
        });
        if (expired == order_.end())
            throw ApiException({.status  = 429,
                                .type    = "rate_limit_error",
                                .message = "background job capacity reached",
                                .code    = "server_overloaded"});
        records_.erase(*expired);
        order_.erase(expired);
    }
    auto job        = std::make_shared<BackgroundResponse>();
    job->response   = std::move(initial);
    job->streaming  = streaming;
    job->byte_limit = slot_bytes_ / 2;
    if (streaming) {
        nlohmann::json event{
            {"type", "response.created"}, {"sequence_number", 0}, {"response", job->response}};
        job->append({"event: response.created\ndata: " + event.dump() + "\n\n"});
    }
    records_.emplace(id, job);
    order_.push_back(id);
    queue_.push_back({job, [job, run = std::move(run)] {
                          {
                              std::lock_guard lock(job->mutex);
                              if (job->done) return;
                              job->executing = true;
                          }
                          run(job);
                      }});
    ready_.notify_one();
    return job;
}

std::shared_ptr<BackgroundResponse> BackgroundResponses::get(const std::string& id) {
    std::lock_guard lock(mutex_);
    auto found = records_.find(id);
    return found == records_.end() ? nullptr : found->second;
}

bool BackgroundResponses::erase(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (!records_.erase(id)) return false;
    std::erase(order_, id);
    return true;
}
} // namespace ninfer::serve
