#include "serve/responses_background.h"
#include "serve/request.h"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace ninfer::serve;
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Json queued(const char* id) {
    return {{"id", id}, {"model", "test"}, {"created_at", 1}, {"status", "queued"}};
}

int main() {
    try {
        std::promise<void> entered;
        BackgroundResponses manager(2, 32ULL << 20, 1);
        auto active = manager.submit("active", queued("active"), false, 1, [&](auto job) {
            entered.set_value();
            while (!job->cancelled) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            job->fail("cancelled", "", "");
        });
        entered.get_future().wait();
        auto pending = manager.submit("pending", queued("pending"), true, 1, [](auto) {
            throw std::runtime_error("cancelled queued work was executed");
        });
        bool full    = false;
        try {
            manager.submit("overflow", queued("overflow"), false, 1, [](auto) {});
        } catch (const ApiException& error) { full = error.error().status == 429; }
        require(full, "active jobs must never be evicted");
        pending->cancel();
        require(pending->snapshot()["status"] == "cancelled",
                "queued cancellation must settle immediately");
        {
            std::lock_guard lock(pending->mutex);
            require(pending->events.size() == 2,
                    "queued stream needs created and cancelled events");
        }
        std::promise<void> finished;
        auto replacement =
            manager.submit("replacement", queued("replacement"), false, 1, [&](auto job) {
                job->finish({{"id", "replacement"}, {"status", "completed"}});
                finished.set_value();
            });
        require(manager.get("active") != nullptr && manager.get("pending") == nullptr,
                "eviction must target a terminal job");
        active->cancel();
        require(finished.get_future().wait_for(std::chrono::seconds(3)) ==
                    std::future_status::ready,
                "queued work did not progress after cancellation");
        manager.stop();
        require(active->snapshot()["status"] == "cancelled", "shutdown must join active work");
        require(replacement->snapshot()["status"] == "completed", "replacement result lost");
        require(manager.erase("replacement") && !manager.get("replacement"), "delete failed");

        BackgroundResponse small;
        small.response   = queued("small");
        small.byte_limit = 8192;
        bool bounded     = false;
        try {
            small.append({std::string(8192, 'x')});
        } catch (const ApiException& error) {
            bounded = error.error().code == "response_buffer_capacity";
        }
        require(bounded, "event buffer must be bounded");
        small.fail("failed", "buffer full", "response_buffer_capacity");
        require(small.snapshot()["status"] == "failed", "overflow must settle as failed");
        std::cout << "background lifecycle, cancellation, eviction, bounds PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
