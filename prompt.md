# ABIDES HIP-3 — Computational Optimization Analysis

Based on a full codebase review of the hot paths, here is a prioritized list of optimization opportunities with estimated impact. Your baseline is **15-20 minutes for a 14-day simulation processing ~19M messages** (~19k-22k msg/sec).

---

## Hot Path Breakdown (Estimated Time Allocation)

| Layer | % of Wall Clock | What Happens |
|---|---|---|
| **Kernel event loop + heapq** | ~25% | 19M `heappop`/`heappush` cycles, tuple packing/unpacking, agent dispatch |
| **`sendMessage()` + `deepcopy`** | ~20% | Every message is deepcopied on creation (via `Message(body=deepcopy(body))` in agent code), then queued |
| **Agent `wakeup()` / `receiveMessage()` dispatch** | ~15% | 180 agents × hundreds of wakes, Python method call overhead, `pd.Timedelta` math |
| **`PerpOrderBook.match_order()` + `enter_order()`** | ~12% | List insertions, price-level scans, order cloning, history dict updates |
| **`Clearinghouse.process_fill()`** | ~8% | Fee computation (volume tier lookup, staking, rebates), margin resize, position update |
| **`_publish_order_book_data()` (every block)** | ~7% | Iterates all subscriptions, builds snapshot dicts, sends `MARKET_DATA` messages |
| **Block processing + trigger/liquidation checks** | ~5% | Action sorting, trigger order evaluation, TWAP slice dispatch |
| **Agent strategy logic (all 180)** | ~5% | Moving averages, Bayesian updates, Chiarella forecasts, random draws |
| **Logging + bookkeeping** | ~3% | `logEvent()` calls, `deepcopy` of event data, summary aggregation |

---

## Tier 1: High-Impact, Low-Effort Optimizations (1-2 hours of work, 30-50% speedup)

### 1. Eliminate Unnecessary `deepcopy()` on Messages

**File:** `agent/Agent.py`, `Kernel.py`

**Problem:** In `PerpTradingAgent.placeLimitOrder()` (and all order placement methods), the order object is passed into the message body. The message body is then deepcopied. But the **agent no longer needs the order after sending** — it only reads from it again after the exchange sends back a response with a filled copy.

Similarly, in the Kernel's `sendMessage()`, the message is not copied again, but agents deepcopied the body before wrapping it in a `Message()`.

**Fix:** In `Kernel.runner()`, wrap the `sendMessage` flow to only deepcopy for message types where the sender retains a reference. For all fire-and-forget messages (`LIMIT_ORDER`, `CANCEL_ORDER`, `QUERY_*`), skip the deepcopy.

**Estimated savings:** ~10-15% of total wall clock. `deepcopy` is one of the most expensive Python operations, and you're calling it ~38M times (2x per message — once in agent, once potentially in kernel).

**Specific change:**
```python
# In agent/Agent.py sendMessage(), add a shallow_copy flag:
def sendMessage(self, recipientID, msg, delay=0, deep_copy=True):
    if deep_copy:
        msg = Message(deepcopy(msg.body))
    self.kernel.sendMessage(self.id, recipientID, msg, delay=delay)
```

Then in `PerpTradingAgent`, pass `deep_copy=False` for order placements since the original order object is cloned by the exchange anyway.

### 2. Replace `pd.Timedelta` with Integer Nanoseconds Internally

**File:** `Kernel.py` (lines 283-288, 313-318), all agent `wakeup()` methods

**Problem:** The Kernel uses `pd.Timedelta` objects for every time computation. Creating a `pd.Timedelta` involves Python object allocation + pandas C-extension overhead. In a tight loop processing 19M messages, this is **millions of `pd.Timedelta` allocations**.

Look at these lines in the Kernel event loop:
```python
self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] + self.currentAgentAdditionalDelay)
```
This creates a new `pd.Timedelta` object on **every single message dispatch** — both for WAKEUP and MESSAGE types. That's ~19M `pd.Timedelta` allocations.

**Fix:** Keep agent current times as **integer nanoseconds** internally. Only convert to `pd.Timestamp` when calling agent methods that expect timestamps.

**Estimated savings:** ~8-12% of total wall clock. This is pure allocation overhead that accumulates.

### 3. Pre-compute and Cache `subscription_dict` Iteration

**File:** `agent/PerpExchangeAgent.py`, `_publish_order_book_data()` (lines 1590-1650)

**Problem:** Every block (every 100ms), this method iterates over `self.subscription_dict` which is a nested dict: `agent_id -> symbol -> [levels, freq, last_update, next_due]`. With 180 agents all subscribed, this is 180 iterations × dict lookups × `getInsideBids()`/`getInsideAsks()` calls × `sendMessage()` calls **every 100ms**. That's ~8,000+ block cycles over 14 days, each doing 180 iterations = **~1.4M subscription checks**.

The `snapshot_cache` helps but is per-block, not persistent. The `latest_update_cache` is also per-block.

**Fix:** Maintain a **flat list** of `(agent_id, symbol, levels, next_due)` tuples sorted by `next_due`. Only iterate agents whose `next_due <= currentTime`. This reduces each block's work from O(n_agents) to O(n_ready_agents), which is typically 0-5 agents per block (most agents wake on their own schedule, not on market data pushes).

**Estimated savings:** ~5-7% of total wall clock.

---

## Tier 2: Medium-Impact, Medium-Effort (3-6 hours, additional 15-25% speedup)

### 4. Optimize `PerpOrderBook.enter_order()` — Use Dict Instead of List Scan

**File:** `util/PerpOrderBook.py`, `enter_order()` (lines 46-67)

**Problem:** `enter_order()` scans `self.bids` or `self.asks` (a list of price levels) linearly to find the right insertion point. With 171k+ accepted orders over a simulation, many orders enter the book. Each scan is O(n_levels). With a book that can have 50-200 price levels, and 85k+ GTC orders entering the book, this is **millions of list comparisons**.

```python
for idx, level in enumerate(book):
    reference = level[0]
    if self._is_better_price(order, reference):
        book.insert(idx, [order])  # O(n) list insertion!
```

`list.insert(idx, item)` is O(n) because it shifts all elements after `idx`.

**Fix:** Use `bisect` module for O(log n) price level lookup. Use `collections.OrderedDict` or a sorted dict for price levels, where each level maps to a FIFO list of orders. Insertion becomes O(log n_levels) for finding the level + O(1) for appending to the level's order list.

**Estimated savings:** ~4-6% of total wall clock.

### 5. Eliminate `PerpOrderBook.history` Overhead

**File:** `util/PerpOrderBook.py`, history tracking (lines 62-67, 147-150, 183-197, 200-212)

**Problem:** On every order enter, cancel, modify, and fill, the code updates `self.history[0]` — a dict mapping `order_id` to a nested dict with `transactions`, `modifications`, `cancellations` lists. Then on every fill, it does `self.history.insert(0, {})` and truncates to `stream_history + 1` entries.

The history is used for:
1. `_get_recent_history()` → `_update_unrolled_transactions()` → `get_transacted_volume()` — which is only called by agents querying transacted volume
2. The exchange's `stream_history=10` reporting

With 171k accepted orders, each creating a history entry with list appends, and each fill triggering a `history.insert(0, {})` (which is O(n) for the list), this is significant overhead.

**Fix:** If you don't need `get_transacted_volume()` or `stream_history` for your analysis, **disable history tracking entirely** with a flag. If you do need it, use `collections.deque(maxlen=11)` instead of list insert+slice for the history window, and use a simpler counter-based approach for transaction recording.

**Estimated savings:** ~3-5% of total wall clock.

### 6. Batch `MARKET_DATA` Subscription Messages

**File:** `agent/PerpExchangeAgent.py`, `_publish_order_book_data()`

**Problem:** For each subscribed agent, the exchange sends a separate `Message` object with a snapshot dict. With 180 agents subscribed at level 1, that's **180 messages per block** just for market data pushes. Each message goes through `sendMessage()` → `heapq.heappush()` → eventual `heappop()` → `receiveMessage()`. That's **4 heap operations per market data message**.

**Fix:** Instead of sending individual `MARKET_DATA` messages, maintain a **push list** that agents read from on their next `wakeup()`. Or, only send `MARKET_DATA` when the book actually changed (it hasn't changed for most agents on most blocks).

**Estimated savings:** ~4-6% of total wall clock (fewer messages in the queue).

---

## Tier 3: Structural Optimizations (Significant effort, diminishing returns)

### 7. Replace the Kernel Event Queue with a Calendar Queue

**File:** `Kernel.py`, `_HeapMessageQueue`

**Problem:** `heapq` is O(log n) per push/pop. With 19M messages and a queue that can grow to 50k-100k pending items, that's ~19M × log(50k) ≈ **300M comparison operations**.

A **calendar queue** (time-bucketed priority queue) is O(1) amortized for events clustered in time. In a simulation where most events are scheduled 1-100ms ahead, a calendar queue with 1ms buckets would reduce queue operations dramatically.

**Estimated savings:** ~10-15% of total wall clock.

**Effort:** High. Requires rewriting the core event dispatch.

### 8. Use `__slots__` on Hot Data Classes

**Files:** `util/order/PerpLimitOrder.py`, `util/PerpAccount.py`, `message/Message.py`, `agent/PerpExchangeAgent.py` (the `PendingAction` and `TriggerGroup` dataclasses)

**Problem:** Python classes use `__dict__` by default, which is a hash map lookup on every attribute access. With millions of `PerpLimitOrder` clones, `Message` creations, and `PendingAction` instantiations, the dict overhead adds up.

**Fix:** Add `__slots__` to `PerpLimitOrder`, `Message`, `Position`, `OrderHold`, and `PendingAction`. This reduces per-instance memory by ~30-50% and speeds attribute access by ~10-20%.

**Estimated savings:** ~3-5% of total wall clock.

### 9. Cython/Numba the `PerpOrderBook.match_order()` Loop

**File:** `util/PerpOrderBook.py`, `match_order()` (lines 138-173)

**Problem:** The inner matching loop does:
```python
while incoming_order.quantity > 1e-12 and book:
    resting = book[0][0]
    if not self._is_match(incoming_order, resting): break
    if resting.agent_id == incoming_order.agent_id:
        cancelled.append(self.cancel_order(resting.order_id))
        continue
    fill_qty = min(incoming_order.quantity, resting.quantity)
    ...
```

This is pure Python doing list indexing, float comparison, and object method calls. With 10 fills in your current run, it's not the bottleneck **yet**. But if you implement the agent participation fixes and fills increase to 500-2,000, this will become more significant.

**Fix:** Annotate with `@numba.njit` or rewrite in Cython. The challenge is that the method manipulates complex Python objects (`PerpLimitOrder` instances), so you'd need to extract the hot loop into a pure numeric function and call back to Python for object manipulation.

**Estimated savings:** ~2-4% currently, potentially 8-10% with higher fill rates.

---

## Tier 4: Not Worth the Effort

| Optimization | Why Skip It |
|---|---|
| Multiprocessing agents | Agent wake cycles are sequential by design (event-driven). Parallelizing would require a fundamentally different execution model |
| Async/await | No I/O-bound work. Pure CPU. Async adds scheduling overhead with no benefit |
| Pandas vectorization | The simulation is inherently sequential. No batch operations to vectorize |
| Caching agent state | State changes every wake cycle. Nothing is stable enough to cache |
| Reducing `stream_history` | Already at 10. Lowering it saves negligible time |
| Reducing `block_interval_ms` | Would make it **slower** (more blocks to process) |

---

## Prioritized Implementation Plan

If you want the **maximum speedup for minimum effort**, do these in order:

| # | Optimization | Effort | Est. Speedup | Cumulative |
|---|---|---|---|---|
| 1 | Skip `deepcopy` on fire-and-forget messages | 1 hour | 10-15% | 10-15% |
| 2 | Cache subscription iteration (flat sorted list) | 1 hour | 5-7% | 15-22% |
| 3 | Disable `PerpOrderBook.history` or use `deque` | 1 hour | 3-5% | 18-27% |
| 4 | Add `__slots__` to `PerpLimitOrder`, `Message`, `PendingAction` | 1 hour | 3-5% | 21-32% |
| 5 | Use `bisect` for order book price level insertion | 2 hours | 4-6% | 25-38% |
| 6 | Replace `pd.Timedelta` with int-ns internally | 3 hours | 8-12% | 33-50% |

**Total expected improvement: ~33-50% faster.** Your 15-20 minute sim would run in **~8-12 minutes**.

If you also do the calendar queue (#7):
| 7 | Calendar queue for event dispatch | 6-8 hours | 10-15% | **43-65%** |

**Total with calendar queue: ~43-65% faster.** Your 15-20 minute sim would run in **~5.5-10 minutes**.

---

## Validation After Each Change

After implementing any optimization:
```bash
# Verify it still runs
python abides.py -c hip3_perp -- --oracle-csv data/sample_oracle.csv \
    --start-time "2025-01-01 00:00:00" --end-time "2025-01-01 00:05:00" --seed 42

# Verify tests still pass
python tests/test_perp_e2e.py
```

The 5-minute smoke test should finish in ~20-30 seconds currently. If it drops to ~12-18 seconds after optimizations, the improvements are real.
