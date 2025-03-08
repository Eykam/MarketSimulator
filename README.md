# High-Performance Stock Trading Engine

A real-time stock trading engine for matching buy and sell orders with lock-free data structures and high throughput.

## Table of Contents

- [Getting Started](#getting-started)
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
  - [Order Book Structure](#order-book-structure)
  - [Lock-Free Design](#lock-free-design)
- [Order Processing Algorithms](#order-processing-algorithms)
  - [addOrder Implementation](#addorder-implementation)
  - [matchOrder Implementation](#matchorder-implementation)
- [Simulation Framework](#simulation-framework)
  - [Thread Architecture](#thread-architecture)
  - [Order Generation](#order-generation)
- [Performance Metrics](#performance-metrics)
- [Testing](#testing-framework)

## Getting Started

### Building from Source

- Zig compiler (0.14.0)

```bash
zig build run
```
### Running Tests
- Zig compiler (0.14.0)

```bash
zig build test
```

### Running the Simulator

The produced binaries are located in zig-out/bin/* if built from source. Pre-built binaries are located [here](binaries)
To run the simulation, you can call the binary from any terminal.

Example: 

```bash
./MarketSimulator
```

### Command Line Options

The simulator supports several command-line options to configure its behavior:

```
Usage: MarketSimulator [options]

Options:
  --ordering-threads=N   Set number of order generator threads (default: 350)
  --matching-threads=N   Set number of order matching threads (default: 4)
  --order-interval=N     Set order interval in microseconds (default: 1)
  --runtime=N            Set runtime in seconds (default: indefinite)
  --help                 Display this help message
```

#### Examples

```bash
# Run with 500 order generator threads
./MarketSimulator --ordering-threads=500

# Run with 8 matcher threads
./MarketSimulator --matching-threads=8

# Run with 2 microsecond interval between orders (slower rate)
./MarketSimulator --order-interval=2

# Run for exactly 30 seconds then exit
./MarketSimulator --runtime=30

# Combine multiple options
./MarketSimulator --ordering-threads=400 --matching-threads=6 --order-interval=1
```

#### Performance Tuning

- **CPU-bound workloads**: Reduce `ordering-threads` and/or increase `order-interval`
- **Matching Bottleneck Detected**: Increase `matching-threads` to increase matching throughput


## Introduction

### Why Zig?

This trading engine is implemented in [Zig](https://ziglang.org/), a modern systems programming language that offers:

- **Performance**: Low-level control with minimal runtime overhead
- **Memory Safety**: Compile-time error detection without garbage collection
- **Concurrency**: First-class support for threading and atomic operations
- **Clarity**: Explicit allocation and error handling for critical financial systems

### Memory Model and Allocation Strategy

The engine uses a fixed-buffer approach with zero dynamic allocations during operation:

- All data structures are pre-allocated with fixed sizes
- No heap allocations during the order processing lifecycle
- All buffers are declared with known sizes at compile time
- Fixed arrays are used instead of dynamic lists or vectors

This approach delivers several benefits:
- Predictable memory usage (steady at ~100MB even under maximum load)
- No garbage collection or memory fragmentation concerns
- Eliminates allocation-related contention between threads
- Consistent performance characteristics regardless of runtime duration

### Benchmarks and Throughput

The engine has been benchmarked on a high-end desktop CPU (Intel Core i9-12900K with 16 cores):

| Metric | Performance |
|--------|-------------|
| Order throughput | 4,281,676 orders/sec |
| Match throughput | 4,150,732 matches/sec |
| Average execution time | 17.7ms |
| Max active orders | ~3.5 million concurrent orders |
| Tickers supported | 1,024 stocks |
| Memory usage | ~100MB steady state |

In a 60-second benchmark run, the system processed over **256 million orders** with a **match efficiency of 0.98**. The system was completely CPU-bound during these tests, demonstrating exceptional efficiency in memory usage while saturating all available CPU cores.

The configuration used to achieve this throughput:

```zig
const config = SimulationConfig{
    .num_ordering_threads = 350,
    .order_interval_ns = std.time.ns_per_us,
    .book_cleanup_interval = 500,
    .matches_per_cycle = 250,
    .num_matching_threads = 4,
};
```

## System Architecture

The trading engine is designed as a multi-tier system with clear separation of concerns:

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│                 │     │               │     │                 │
│ Order Generator │────▶│  Order Book   │────▶│ Order Matcher   │
│    Threads      │     │               │     │    Threads      │
│                 │     │               │     │                 │
└─────────────────┘     └───────────────┘     └─────────────────┘
                               │                       │
                               ▼                       ▼
                        ┌───────────────┐     ┌─────────────────┐
                        │    Logger     │     │   Performance   │
                        │               │     │    Metrics      │
                        └───────────────┘     └─────────────────┘
```

### Order Book Structure

The order book implements a two-level hierarchical structure with fixed-size buffers throughout:

1. **OrderBook**: Top-level container managing all tickers
   - Maintains fixed array of `TickerOrderBook` instances (one per ticker)
   - Fixed size of MAX_TICKERS (1,024) to avoid dynamic resizing
   - Manages matcher threads for parallel order processing
   - Provides thread-safe order submission interface
   - Uses atomic order ID generation for unique identification

2. **TickerOrderBook**: Per-ticker order management
   - Maintains two separate fixed-size arrays: `buy_orders[MAX_ORDERS_PER_TICKER]` and `sell_orders[MAX_ORDERS_PER_TICKER]`
   - Arrays have a fixed capacity of 10,000 orders each with no dynamic resizing
   - Each array is maintained in sorted order by price (descending for buys, ascending for sells)
   - Uses reference price with configurable deviation bands
   - Implements periodic cleanup to compact arrays and remove filled orders or orders outside price bands

```zig
pub const TickerOrderBook = struct {
    ticker_id: u16,
    reference_price: AtomicF32,
    max_price_deviation: f32,
    buy_orders: [MAX_ORDERS_PER_TICKER]Order,
    sell_orders: [MAX_ORDERS_PER_TICKER]Order,
    buy_count: AtomicU32,
    sell_count: AtomicU32,
    needs_matching: AtomicBool,
    // ...other fields...
};
```

Each order in the book is represented by:

```zig
const Order = struct {
    id: u64,
    ticker_index: u16,
    quantity: AtomicU32,
    price: f32,
    timestamp: u64,
    status: AtomicEnum,
    // ...methods...
};
```

### Lock-Free Design

The system utilizes lock-free data structures throughout to maximize concurrency:

1. **Atomic Values**: Uses Zig's atomic types for thread-safe operations:
   - `AtomicU32` for counters and quantities
   - `AtomicU64` for timestamps and IDs
   - `AtomicBool` for state flags
   - `AtomicEnum` for order status
   - `AtomicF32` for reference prices

2. **Thread Synchronization**:
   - No mutex locks or traditional synchronization primitives
   - All shared state is managed through atomic operations
   - Thread communication via atomic flags (e.g., `needs_matching`)

3. **Order Status Transitions**:
   - Atomic compare-and-swap for state transitions
   - Thread-safe quantity updates using atomic fetch-and-sub

4. **Array Compaction**:
   - Periodic cleanup and compaction of arrays
   - Atomic update of array size counters

This lock-free approach enables high throughput by minimizing thread contention and avoiding expensive lock operations.

## Order Processing Algorithms

### addOrder Implementation

The `addOrder` function implements a sorted insertion with binary search:

```zig
pub fn addOrder(self: *Self, order: Order, order_type: OrderType, logger: *Logger, metrics: *PerformanceMetrics) !void {
    // Price band validation
    const ref_price = self.reference_price.load(.acquire);
    if (ref_price > 0) {
        const min_acceptable = ref_price * (1.0 - self.max_price_deviation);
        const max_acceptable = ref_price * (1.0 + self.max_price_deviation);
        
        // Reject orders outside price bands
        if (order.price < min_acceptable or order.price > max_acceptable) {
            // ...logging and error return...
        }
    }

    // Determine array to insert into based on order type
    switch (order_type) {
        .BUY => {
            // Check capacity
            const current_count = self.buy_count.load(.acquire);
            if (current_count >= MAX_ORDERS_PER_TICKER) {
                return error.OrderBookFull;
            }

            // Find insertion point using binary search
            const insertion_idx = findInsertionPoint(self.buy_orders[0..current_count], order.price, true);

            // Shift elements to make room
            if (insertion_idx < current_count) {
                std.mem.copyBackwards(Order, self.buy_orders[insertion_idx + 1 .. current_count + 1], self.buy_orders[insertion_idx..current_count]);
            }

            // Insert at the right position
            self.buy_orders[insertion_idx] = order;

            // Atomically increment count
            _ = self.buy_count.fetchAdd(1, .release);
            metrics.recordBuyOrder();
        },
        .SELL => {
            // Similar logic for sell orders...
        },
    }

    // Signal matching is needed
    self.needs_matching.store(true, .release);
}
```

#### Binary Search for Insertion Point

```zig
fn findInsertionPoint(orders: []Order, price: f32, is_buy: bool) usize {
    var left: usize = 0;
    var right: usize = orders.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const mid_price = orders[mid].price;

        if ((is_buy and mid_price < price) or (!is_buy and mid_price > price)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}
```

#### Algorithmic Analysis

- **Time Complexity**: O(log n + n)
  - Binary search: O(log n) to find insertion point
  - Array shifting: O(n) worst case when inserting at the beginning
  - Best case: O(log n) when inserting at the end (no shifting needed)

- **Space Complexity**: O(1) extra space

- **Thread Safety**: Guaranteed by atomic operations
  - Atomic load/store for count checking
  - Atomic increment for count updating

### matchOrder Implementation

The `matchOrder` function continuously matches highest bids with lowest asks:

```zig
pub fn matchOrders(self: *TickerOrderBook, logger: *Logger, metrics: *PerformanceMetrics) void {
    if (!self.needs_matching.load(.acquire)) {
        return;
    }

    var match_count: u32 = 0;
    const max_matches_per_cycle = self.matches_per_cycle;

    while (match_count < max_matches_per_cycle) {
        // Find best bid and ask - O(1) in average case since they're at the front
        const best_buy_opt = self.findHighestBid();
        const best_sell_opt = self.findLowestAsk();

        if (best_buy_opt == null or best_sell_opt == null) {
            break;
        }

        const best_buy = best_buy_opt.?;
        const best_sell = best_sell_opt.?;

        // Check if price matches
        if (best_buy.price < best_sell.price) {
            break;
        }

        // Get quantities with atomic loads
        const buy_quantity = best_buy.quantity.load(.acquire);
        const sell_quantity = best_sell.quantity.load(.acquire);

        if (buy_quantity == 0 or sell_quantity == 0) {
            // Skip empty orders
            continue;
        }

        // Match the orders
        const match_quantity = @min(buy_quantity, sell_quantity);

        // Update quantities atomically
        const prev_buy_quantity = best_buy.quantity.fetchSub(match_quantity, .monotonic);
        const prev_sell_quantity = best_sell.quantity.fetchSub(match_quantity, .monotonic);
        
        // Update order status based on remaining quantities
        // ...status update logic with atomics...

        // Update reference price
        if (match_count > 0) {
            self.updateReferencePrice(best_sell.price);
        }

        match_count += 1;
        self.match_iterations += 1;
    }

    // Periodically clean up the order book
    if (self.match_iterations >= self.book_cleanup_interval) {
        self.cleanupOrderBook(logger);
        self.match_iterations = 0;
    }

    // Signal if more matching might be needed
    if (match_count >= max_matches_per_cycle) {
        self.needs_matching.store(true, .release);
    } else {
        self.needs_matching.store(false, .release);
    }
}
```

#### Finding Highest Bid and Lowest Ask

```zig
fn findHighestBid(self: *TickerOrderBook) ?*Order {
    const buy_count = self.buy_count.load(.acquire);
    if (buy_count == 0) return null;

    // Orders are sorted by price, so check from the beginning
    for (0..buy_count) |i| {
        const order = &self.buy_orders[i];
        const order_status = order.status.load(.acquire);
        const order_quantity = order.quantity.load(.acquire);

        if (order_status != .FILLED and order_quantity > 0) {
            return &self.buy_orders[i];
        }
    }

    return null;
}
```

#### Algorithmic Analysis

- **Time Complexity**: O(n) where n is the number of tickers
  - Each ticker's matching is O(1) in the average case
  - Finding highest bid/lowest ask is O(1) on average since they're at the front
  - Worst case is O(m) where m is orders in a single ticker (if many orders at front are filled)
  - Overall complexity is O(n) for n tickers processed in parallel

- **Space Complexity**: O(1) extra space

- **Thread Safety**: Guaranteed by atomic operations
  - Atomic loads for order quantity and status checks
  - Atomic fetch-and-subtract for quantity updates
  - Atomic compare-and-swap for status updates

#### Order Book Cleanup

Periodic cleanup removes filled orders and orders outside price bands:

```zig
fn cleanupOrderBook(self: *Self, logger: *Logger) void {
    const ref_price = self.reference_price.load(.acquire);
    const min_acceptable = ref_price * (1.0 - self.max_price_deviation);
    const max_acceptable = ref_price * (1.0 + self.max_price_deviation);

    // Process and compact buy orders
    const buy_count = self.buy_count.load(.acquire);
    var valid_buys: u32 = 0;

    for (0..buy_count) |i| {
        const order = &self.buy_orders[i];
        const order_status = order.status.load(.acquire);
        const order_quantity = order.quantity.load(.acquire);

        if (order_status != .FILLED and 
            order_status != .CANCELED and 
            order_quantity > 0)
        {
            var keep_order = true;

            // Price band check
            if (order.price < min_acceptable or order.price > max_acceptable) {
                order.status.store(.CANCELED, .release);
                // Log cancellation...
                keep_order = false;
            }

            if (keep_order) {
                if (i != valid_buys) {
                    self.buy_orders[valid_buys] = self.buy_orders[i];
                }
                valid_buys += 1;
            }
        }
    }

    // Atomically update count
    if (valid_buys < buy_count) {
        self.buy_count.store(valid_buys, .release);
    }

    // Similar process for sell orders...
}
```

## Simulation Framework

The simulation framework provides a comprehensive environment for testing and benchmarking the trading engine.

### Thread Architecture

The system utilizes multiple thread pools for different responsibilities:

1. **Order Generator Threads**:
   - Configurable number of threads (default 200)
   - Each independently generates random orders
   - Thread count defined by `num_ordering_threads`
   - Managed by the `MarketSimulator`

2. **Order Matcher Threads**:
   - Process order books to find and execute matches
   - Each thread handles a subset of tickers
   - Thread count defined by `num_matching_threads` (default 2)
   - Managed by the `OrderBook`

```zig
// Order generator thread creation
for (0..self.ordering_threads.len) |i| {
    self.ordering_threads[i] = try std.Thread.spawn(.{}, simulate, .{self});
}

// Order matcher thread creation
for (0..threads.len) |idx| {
    threads[idx] = try std.Thread.spawn(.{}, matcherThreadFn, .{ book, idx });
}
```

#### Thread Workload Distribution

Order matching threads divide the ticker space evenly:

```zig
fn matcherThreadFn(self: *Self, idx: usize) void {
    const tickers_per_thread = MAX_TICKERS / self.matcher_threads.len;
    const start_idx = idx * tickers_per_thread;
    const end_idx = start_idx + tickers_per_thread;

    while (self.run_matcher.load(.acquire)) {
        self.matchThreadsOrders(start_idx, end_idx);
        
        // Update metrics periodically on first thread
        if (idx == 0) {
            // Update metrics...
        }
    }
}
```

### Order Generation

Orders are generated with configurable parameters and realistic price distribution:

```zig
fn generateRandomOrder(self: *MarketSimulator) void {
    const random = self.prng.random();

    // Select random ticker
    const ticker_index = random.uintLessThan(u16, MAX_TICKERS);
    
    // Generate random quantity
    const quantity = random.uintLessThan(u32, self.config.max_quantity) + 1;

    // Get ticker reference price
    const ticker_book = self.order_book.ticker_books[ticker_index];
    const ticker_reference_price = ticker_book.reference_price.load(.acquire);

    // Generate price with normal distribution around reference price
    const price_norm = random.floatNorm(f32) * self.config.max_price_deviation + 1.0;
    const price = price_norm * ticker_reference_price;

    // Determine order type (buy or sell) with configurable ratio
    const buy_threshold = @as(f32, 0.5) * self.config.buy_sell_ratio;
    const order_type_int = @intFromBool(random.float(f32) >= buy_threshold);
    const order_type: OrderType = @enumFromInt(order_type_int);

    // Submit order to order book
    _ = self.order_book.addOrder(ticker_index, quantity, price, order_type);
}
```

Key aspects of the order generation process:

1. **Ticker Selection**: Uniform random selection across all tickers
2. **Price Generation**: Normal distribution around reference price
   - Uses `floatNorm()` to generate normally distributed values
   - Scaled by `max_price_deviation` to control volatility
   - Centered around current reference price

3. **Order Type Balancing**:
   - Controlled by `buy_sell_ratio` configuration
   - Ratio of 1.0 generates equal buys and sells
   - Values > 1.0 generate more buys, < 1.0 generate more sells

4. **Quantity Generation**: Uniform random up to configured maximum

### Configuration Options

The simulation behavior is highly configurable through the `SimulationConfig` struct:

```zig
const SimulationConfig = struct {
    timed_simulation: bool = false,        // Run for specific time or indefinitely
    runtime_seconds: ?u64 = null,          // Duration of simulation if timed
    order_interval_ns: u64 = 500 * std.time.ns_per_us, // Time between orders
    num_ordering_threads: u32 = 10,        // Threads generating orders
    num_matching_threads: u32 = 4,         // Threads matching orders
    seed: ?u64 = null,                     // Random seed for reproducibility
    max_price: u32 = 1000,                 // Maximum initial price
    max_price_deviation: f32 = 0.2,        // Price band percentage (±20%)
    book_cleanup_interval: u32 = 250,      // Matches before cleanup
    matches_per_cycle: u32 = 100,          // Max matches in one cycle
    max_quantity: u32 = 1000,              // Maximum order quantity
    performance_logging: bool = true,      // Enable performance metrics
    order_logging: bool = false,           // Enable detailed order logging
    buy_sell_ratio: f32 = 1.0,             // Ratio of buy to sell orders
};
```

## Performance Metrics

The system includes comprehensive real-time performance monitoring:

### Metrics Collection

The `PerformanceMetrics` struct collects and processes various metrics:

```zig
pub const PerformanceMetrics = struct {
    // Order counts
    total_buy_orders: AtomicU64,
    total_sell_orders: AtomicU64,
    pending_buy_orders: AtomicU64,
    pending_sell_orders: AtomicU64,

    // Order status tracking
    buy_pending_count: AtomicU64,
    buy_partial_count: AtomicU64,
    sell_pending_count: AtomicU64,
    sell_partial_count: AtomicU64,

    // Match statistics
    total_matches: AtomicU64,
    total_matched_quantity: AtomicU64,

    // Timing metrics
    start_timestamp: u64,
    last_timestamp: AtomicU64,
    total_execution_time_ms: AtomicU64,
    trade_count_for_avg: AtomicU64,

    // Flow analysis
    flow_analysis: FlowAnalysis,
    last_rate_update: AtomicU64,
    
    // Methods for recording metrics
    // ...
};
```

### Flow Analysis

The system performs real-time flow analysis to detect bottlenecks:

```zig
pub fn analyzeBottlenecks(self: *FlowAnalysis) struct {
    buy_sell_ratio: f64,
    match_efficiency: f64,
    bottleneck: []const u8,
} {
    const buy_rate = self.buy_rate_metric.getLastRate();
    const sell_rate = self.sell_rate_metric.getLastRate();
    const match_rate = self.match_rate_metric.getLastRate();

    // Calculate buy/sell ratio
    var buy_sell_ratio: f64 = 1.0;
    if (sell_rate > 0) {
        buy_sell_ratio = buy_rate / sell_rate;
    }

    // Calculate match efficiency
    var match_efficiency: f64 = 1.0;
    const order_rate = buy_rate + sell_rate;
    if (order_rate > 0) {
        match_efficiency = match_rate / order_rate;
    }

    // Identify bottleneck
    var bottleneck: []const u8 = "None";
    if (match_efficiency < 0.8) {
        if (buy_sell_ratio > 1.2) {
            bottleneck = "Excess buy orders, not enough sells";
        } else if (buy_sell_ratio < 0.8) {
            bottleneck = "Excess sell orders, not enough buys";
        } else {
            bottleneck = "Matching engine capacity";
        }
    }

    return .{
        .buy_sell_ratio = buy_sell_ratio,
        .match_efficiency = match_efficiency,
        .bottleneck = bottleneck,
    };
}
```

### Real-Time Dashboard

The metrics are visualized in a real-time dashboard:

```
╔══════════════════════════════════════════════════════════════════╗
║                   TRADING ENGINE METRICS                         ║
╚══════════════════════════════════════════════════════════════════╝

Runtime: 59.99 seconds

ORDER STATISTICS
  Total Submitted Orders: 256,857,760 (4,281,676.27 /sec)
    ├─ Buy Orders:  128,416,119 (2,140,625.42 /sec)
    └─ Sell Orders: 128,441,641 (2,141,050.85 /sec)

ACTIVE ORDER BREAKDOWN
  Buy Orders Status:
    Total Active: 1,874,065
    ├─ Pending:        1,303,170 (69.53%)
    └─ Partially Filled: 570,895 (30.46%)
    [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ]

  Sell Orders Status:
    Total Active: 1,594,652
    ├─ Pending:        1,023,885 (64.20%)
    └─ Partially Filled: 570,767 (35.79%)
    [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ]

COMPLETED ORDERS
  Total Completed Orders: 253,389,043 (4,223,854.69 /sec)
    ├─ Completed Buy Orders:  126,542,054
    └─ Completed Sell Orders: 126,846,989
  Total Match Events: 249,002,416 (4,150,732.05 /sec)
  Total Matched Quantity: 105,415,124,165
  Average Time to Completion: 17,678.28 ms

FLOW ANALYSIS
  Buy/Sell Ratio: 1.00
  Match Efficiency: 0.98
  Detected Bottleneck: None
```

## Testing Framework

The trading engine includes a comprehensive suite of automated tests to verify correctness, performance, and concurrency handling. Tests are designed to validate both the fundamental functionality of the matching engine and its behavior under high-load, concurrent scenarios.

### Running the Tests

Tests can be executed using the Zig build system:

```bash
zig build test
```

### Test Suite Overview

The test suite contains multiple targeted tests that validate different aspects of the trading engine:

#### 1. Simple Order Matching

Verifies the basic functionality of order matching by creating a single buy and sell order that should match. This test validates that:
- Orders are properly added to the order book
- The matching algorithm correctly identifies matching orders
- After matching, orders are properly removed from the active order book

#### 2. Partial Order Fill

Tests the correct handling of partial order fills by creating a large buy order and a smaller sell order. Validates that:
- The buy order is partially filled
- The sell order is completely filled
- The remaining quantity in the buy order is accurately tracked
- The buy order remains in the order book with correct remaining quantity

#### 3. Price Priority Matching

Ensures the matching algorithm respects price priority by creating multiple sell orders at different prices and a single buy order. Verifies that:
- The buy order matches with the lowest-priced sell order first
- Higher-priced sell orders remain in the book
- Price sorting is maintained correctly in the order book

#### 4. Time Priority Matching

Tests that the matching algorithm properly follows time priority when multiple orders exist at the same price. Creates two sell orders at identical prices but different times, then adds a buy order that should match with the first sell order. Verifies that:
- The first order (by time) is matched first when prices are equal
- The second order remains in the book
- FIFO ordering is properly maintained

#### 5. Multiple Order Matches

Validates that a single large order can match against multiple smaller orders. Creates several small sell orders and a larger buy order that should match with all of them. Verifies that:
- The buy order correctly matches against multiple sell orders
- All sell orders are completely filled
- The buy order has the correct remaining quantity
- Multiple match events are properly recorded

#### 6. Price Boundary Enforcement

Tests the system's price band enforcement mechanism, which rejects orders outside of configured price deviation limits. Verifies that:
- Orders within the allowed price band are accepted
- Orders outside the price band are rejected
- The reference price mechanism works as expected
- Price bands adapt correctly after trades occur

#### 7. Ticker Isolation

Ensures that orders in different tickers don't match with each other. Creates matching buy and sell orders in different tickers and verifies they remain unmatched, while matches do occur for orders in the same ticker. This test validates the correct partitioning of the order book by ticker.

#### 8. Concurrent Order Generation

This test validates the engine's performance under heavy load with concurrent order generation. It launches multiple threads that simultaneously add orders to the same ticker, then verifies:
- The order book correctly handles concurrent order insertion
- A high percentage of eligible orders are matched
- The matching engine keeps up with the order generation rate
- Thread safety is maintained throughout the system
- The final state of the order book is consistent and balanced

Each test is designed to isolate and verify a specific aspect of the trading engine, with comprehensive verification of both the logs and the actual state of the order book after operations complete.

