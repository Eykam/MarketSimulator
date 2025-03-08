const std = @import("std");
const MarketSimulator = @import("MarketSimulator.zig");

const AtomicU32 = std.atomic.Value(u32);
const AtomicU64 = std.atomic.Value(u64);

const OrderType = MarketSimulator.OrderType;
const OrderStatus = MarketSimulator.OrderStatus;
const TickerOrderBook = MarketSimulator.TickerOrderBook;
const SimulationConfig = MarketSimulator.SimulationConfig;

const MAX_TICKERS = MarketSimulator.MAX_TICKERS;
const MAX_LOG_ENTRIES = 100000;

const RATE_HISTORY_SIZE = 10; // Store last 10 time points for rate analysis
const RATE_WINDOW_MS = 1000; // Calculate rates over 1-second windows

const ANSI_GREEN = "\x1B[32m";
const ANSI_RED = "\x1B[31m";
const ANSI_YELLOW = "\x1B[33m";
const ANSI_BLUE = "\x1B[34m";
const ANSI_CYAN = "\x1B[36m";
const ANSI_MAGENTA = "\x1B[35m";
const ANSI_RESET = "\x1B[0m";
const ANSI_BOLD = "\x1B[1m";

const LogEventType = enum {
    ORDER_ADDED,
    ORDER_MATCHED,
    ORDER_CANCELED,
    ORDER_REJECTED,
    SYSTEM_INFO,
};

pub const LogEntry = struct {
    timestamp: u64,
    event_type: LogEventType,
    ticker_index: u16,
    order_id: u64,
    order_type: ?OrderType,
    quantity: u32,
    price: f32,
    message: [128]u8,

    pub fn init(event_type: LogEventType, ticker_index: u16, order_id: u64, order_type: ?OrderType, quantity: u32, price: f32) LogEntry {
        return LogEntry{
            .timestamp = @intCast(std.time.milliTimestamp()),
            .event_type = event_type,
            .ticker_index = ticker_index,
            .order_id = order_id,
            .order_type = order_type,
            .quantity = quantity,
            .price = price,
            .message = [_:0]u8{0} ** 128,
        };
    }

    pub fn formatMessage(self: *LogEntry, comptime fmt: []const u8, args: anytype) void {
        _ = std.fmt.bufPrint(&self.message, fmt, args) catch |err| {
            std.debug.print("Error formatting log message: {}\n", .{err});
        };
    }
};

// Thread-safe Logger
pub const Logger = struct {
    const Self = @This();

    log_entries: [MAX_LOG_ENTRIES]LogEntry,
    next_log_index: AtomicU32,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const logger = try allocator.create(Self);
        logger.* = Self{
            .log_entries = undefined,
            .next_log_index = AtomicU32.init(0),
        };
        return logger;
    }

    pub fn log(self: *Self, entry: LogEntry) void {
        const index = self.next_log_index.fetchAdd(1, .monotonic) % MAX_LOG_ENTRIES;
        self.log_entries[index] = entry;
    }

    pub fn generateReport(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        var report = std.ArrayList(u8).init(allocator);
        defer report.deinit();

        const writer = report.writer();

        try writer.print("=== Trading System Report ===\n\n", .{});

        const current_index: u32 = self.next_log_index.load(.acquire) % MAX_LOG_ENTRIES;
        const num_entries: u32 = @min(current_index, MAX_LOG_ENTRIES);

        var i: u32 = 0;
        while (i < num_entries) : (i += 1) {
            const entry = self.log_entries[i];
            const time_str = formatTimestamp(entry.timestamp);
            try writer.print("[{s}] ", .{time_str});

            switch (entry.event_type) {
                .ORDER_ADDED => {
                    const order_type = @tagName(entry.order_type.?);
                    try writer.print("Added {s} Order #{d}: Ticker {d}, Price {d:.2}, Qty {d}\n", .{ order_type, entry.order_id, entry.ticker_index, @abs(entry.price), entry.quantity });
                },
                .ORDER_MATCHED => {
                    try writer.print("Matched: Ticker {d}, Price {d:.2}, Qty {d}\n", .{ entry.ticker_index, entry.price, entry.quantity });
                },
                .ORDER_CANCELED => {
                    try writer.print("Canceled Order #{d}\n", .{entry.order_id});
                },
                .ORDER_REJECTED => {
                    try writer.print("Rejected Order #{d}, Price {d:.2}, Qty {d}\n", .{ entry.order_id, entry.price, entry.quantity });
                },
                .SYSTEM_INFO => {
                    try writer.print("System: {s}\n", .{entry.message});
                },
            }
        }

        return report.toOwnedSlice();
    }

    fn formatTimestamp(timestamp: u64) [24]u8 {
        var buf: [24]u8 = undefined;
        const millis = timestamp % 1000;
        const seconds = (timestamp / 1000) % 60;
        const minutes = (timestamp / 60000) % 60;
        const hours = (timestamp / 3600000) % 24;

        _ = std.fmt.bufPrint(&buf, "{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}", .{ hours, minutes, seconds, millis }) catch unreachable;

        return buf;
    }
};

const RateMetric = struct {
    timestamps: [RATE_HISTORY_SIZE]u64,
    counts: [RATE_HISTORY_SIZE]u64,
    current_index: usize,

    pub fn init() RateMetric {
        return RateMetric{
            .timestamps = [_]u64{0} ** RATE_HISTORY_SIZE,
            .counts = [_]u64{0} ** RATE_HISTORY_SIZE,
            .current_index = 0,
        };
    }

    pub fn recordSample(self: *RateMetric, timestamp: u64, count: u64) void {
        self.timestamps[self.current_index] = timestamp;
        self.counts[self.current_index] = count;
        self.current_index = (self.current_index + 1) % RATE_HISTORY_SIZE;
    }

    pub fn calculateRate(self: *RateMetric) f64 {
        if (self.timestamps[0] == 0) return 0.0;

        const oldest_index = (self.current_index + 1) % RATE_HISTORY_SIZE;
        const newest_index = if (self.current_index == 0) RATE_HISTORY_SIZE - 1 else self.current_index - 1;

        const oldest_time = self.timestamps[oldest_index];
        const newest_time = self.timestamps[newest_index];

        if (newest_time <= oldest_time) return 0.0;

        const oldest_count = self.counts[oldest_index];
        const newest_count = self.counts[newest_index];

        const time_diff_sec = @as(f64, @floatFromInt(newest_time - oldest_time)) / 1000.0;
        const count_diff = newest_count - oldest_count;

        return @as(f64, @floatFromInt(count_diff)) / time_diff_sec;
    }

    pub fn getLastRate(self: *RateMetric) f64 {
        return self.calculateRate();
    }
};

// Flow analysis for detecting bottlenecks
const FlowAnalysis = struct {
    buy_rate_metric: RateMetric,
    sell_rate_metric: RateMetric,
    match_rate_metric: RateMetric,

    pub fn init() FlowAnalysis {
        return FlowAnalysis{
            .buy_rate_metric = RateMetric.init(),
            .sell_rate_metric = RateMetric.init(),
            .match_rate_metric = RateMetric.init(),
        };
    }

    pub fn recordRates(self: *FlowAnalysis, timestamp: u64, buy_count: u64, sell_count: u64, match_count: u64) void {
        self.buy_rate_metric.recordSample(timestamp, buy_count);
        self.sell_rate_metric.recordSample(timestamp, sell_count);
        self.match_rate_metric.recordSample(timestamp, match_count);
    }

    pub fn analyzeBottlenecks(self: *FlowAnalysis) struct {
        buy_sell_ratio: f64,
        match_efficiency: f64,
        bottleneck: []const u8,
    } {
        const buy_rate = self.buy_rate_metric.getLastRate();
        const sell_rate = self.sell_rate_metric.getLastRate();
        const match_rate = self.match_rate_metric.getLastRate();

        // Calculate buy/sell ratio (>1 means more buys than sells)
        var buy_sell_ratio: f64 = 1.0;
        if (sell_rate > 0) {
            buy_sell_ratio = buy_rate / sell_rate;
        }

        // Calculate match efficiency (1.0 means perfect matching)
        // This measures how well the matching engine keeps up with incoming orders
        var match_efficiency: f64 = 1.0;
        const order_rate = buy_rate + sell_rate;
        if (order_rate > 0) {
            match_efficiency = match_rate / order_rate; //
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
};

pub const PerformanceMetrics = struct {
    const Self = @This();

    // Order counts
    total_buy_orders: AtomicU64,
    total_sell_orders: AtomicU64,

    pending_buy_orders: AtomicU64,
    pending_sell_orders: AtomicU64,

    // Match counts
    total_matches: AtomicU64,
    total_matched_quantity: AtomicU64,

    buy_pending_count: AtomicU64,
    buy_partial_count: AtomicU64,
    sell_pending_count: AtomicU64,
    sell_partial_count: AtomicU64,

    // Timing metrics
    start_timestamp: u64,
    last_timestamp: AtomicU64,

    // Trade execution timing
    total_execution_time_ms: AtomicU64,
    trade_count_for_avg: AtomicU64,

    // Flow analysis
    flow_analysis: FlowAnalysis,
    last_rate_update: AtomicU64,

    config_ordering_threads: u32,
    config_matching_threads: u32,
    config_order_interval_us: u64,

    pub fn init(allocator: std.mem.Allocator, config: SimulationConfig) !*Self {
        const current_time: u64 = @intCast(std.time.milliTimestamp());
        const metrics = try allocator.create(Self);
        metrics.* = Self{
            .total_buy_orders = AtomicU64.init(0),
            .total_sell_orders = AtomicU64.init(0),

            .pending_buy_orders = AtomicU64.init(0),
            .pending_sell_orders = AtomicU64.init(0),

            .total_matches = AtomicU64.init(0),
            .total_matched_quantity = AtomicU64.init(0),

            .buy_pending_count = AtomicU64.init(0),
            .buy_partial_count = AtomicU64.init(0),
            .sell_pending_count = AtomicU64.init(0),
            .sell_partial_count = AtomicU64.init(0),

            .start_timestamp = current_time,
            .last_timestamp = AtomicU64.init(current_time),
            .total_execution_time_ms = AtomicU64.init(0),
            .trade_count_for_avg = AtomicU64.init(0),
            .flow_analysis = FlowAnalysis.init(),
            .last_rate_update = AtomicU64.init(current_time),

            .config_ordering_threads = config.num_ordering_threads,
            .config_matching_threads = config.num_matching_threads,
            .config_order_interval_us = config.order_interval_ns / std.time.ns_per_us,
        };
        return metrics;
    }

    pub fn recordBuyOrder(self: *Self) void {
        _ = self.total_buy_orders.fetchAdd(1, .monotonic);
        _ = self.pending_buy_orders.fetchAdd(1, .monotonic);
        // Also increment the pending count for status tracking
        _ = self.buy_pending_count.fetchAdd(1, .monotonic);
        self.updateTimestamp();
        self.updateRateMetrics();
    }

    pub fn recordSellOrder(self: *Self) void {
        _ = self.total_sell_orders.fetchAdd(1, .monotonic);
        _ = self.pending_sell_orders.fetchAdd(1, .monotonic);
        // Also increment the pending count for status tracking
        _ = self.sell_pending_count.fetchAdd(1, .monotonic);
        self.updateTimestamp();
        self.updateRateMetrics();
    }

    pub fn recordMatch(self: *Self, quantity: u32, execution_time_ms: u64) void {
        _ = self.total_matches.fetchAdd(1, .monotonic);
        _ = self.total_matched_quantity.fetchAdd(quantity, .monotonic);
        _ = self.total_execution_time_ms.fetchAdd(execution_time_ms, .monotonic);
        _ = self.trade_count_for_avg.fetchAdd(1, .monotonic);
        self.updateTimestamp();
        self.updateRateMetrics();
    }

    pub fn updateOrderStatus(self: *Self, order_type: OrderType, old_status: OrderStatus, new_status: OrderStatus) void {
        switch (order_type) {
            .BUY => {
                switch (old_status) {
                    .PENDING => {
                        _ = self.buy_pending_count.fetchSub(1, .monotonic);
                    },
                    .PARTIALLY_FILLED => {
                        _ = self.buy_partial_count.fetchSub(1, .monotonic);
                    },
                    else => {},
                }

                switch (new_status) {
                    .PENDING => {
                        _ = self.buy_pending_count.fetchAdd(1, .monotonic);
                    },
                    .PARTIALLY_FILLED => {
                        _ = self.buy_partial_count.fetchAdd(1, .monotonic);
                    },
                    else => {},
                }
            },
            .SELL => {
                switch (old_status) {
                    .PENDING => {
                        _ = self.sell_pending_count.fetchSub(1, .monotonic);
                    },
                    .PARTIALLY_FILLED => {
                        _ = self.sell_partial_count.fetchSub(1, .monotonic);
                    },
                    else => {},
                }

                switch (new_status) {
                    .PENDING => {
                        _ = self.sell_pending_count.fetchAdd(1, .monotonic);
                    },
                    .PARTIALLY_FILLED => {
                        _ = self.sell_partial_count.fetchAdd(1, .monotonic);
                    },
                    else => {},
                }
            },
        }
    }

    pub fn updateOrderBookMetrics(self: *Self, ticker_books: [MAX_TICKERS]*TickerOrderBook) void {
        // Reset counters first
        self.buy_pending_count.store(0, .release);
        self.buy_partial_count.store(0, .release);
        self.sell_pending_count.store(0, .release);
        self.sell_partial_count.store(0, .release);

        // Scan through all tickers
        for (0..MAX_TICKERS) |i| {
            const ticker_book = ticker_books[i];

            // Scan buy orders
            const buy_orders = ticker_book.getBuyOrders();
            for (buy_orders) |*order| {
                const status = order.status.load(.acquire);
                if (status == .PENDING) {
                    _ = self.buy_pending_count.fetchAdd(1, .monotonic);
                } else if (status == .PARTIALLY_FILLED) {
                    _ = self.buy_partial_count.fetchAdd(1, .monotonic);
                }
            }

            // Scan sell orders
            const sell_orders = ticker_book.getSellOrders();
            for (sell_orders) |*order| {
                const status = order.status.load(.acquire);
                if (status == .PENDING) {
                    _ = self.sell_pending_count.fetchAdd(1, .monotonic);
                } else if (status == .PARTIALLY_FILLED) {
                    _ = self.sell_partial_count.fetchAdd(1, .monotonic);
                }
            }
        }
    }

    pub fn safeDecrementPendingOrders(self: *Self, order_type: OrderType) void {
        var pending_orders: AtomicU64 = undefined;
        switch (order_type) {
            .BUY => pending_orders = self.pending_buy_orders,
            .SELL => pending_orders = self.pending_sell_orders,
        }

        var current = pending_orders.load(.acquire);
        while (current > 0) {
            if (pending_orders.cmpxchgWeak(current, current - 1, .acquire, .monotonic)) |actual| {
                current = actual;
            } else {
                break;
            }
        }
    }

    fn updateTimestamp(self: *Self) void {
        self.last_timestamp.store(@intCast(std.time.milliTimestamp()), .release);
    }

    fn updateRateMetrics(self: *Self) void {
        const current_time: u64 = @intCast(std.time.milliTimestamp());
        const last_update = self.last_rate_update.load(.acquire);

        // Only update rates periodically to avoid too frequent updates
        if (current_time - last_update >= RATE_WINDOW_MS / 2) {
            self.last_rate_update.store(current_time, .release);

            const buy_count = self.total_buy_orders.load(.acquire);
            const sell_count = self.total_sell_orders.load(.acquire);
            const match_count = self.total_matches.load(.acquire);

            self.flow_analysis.recordRates(current_time, buy_count, sell_count, match_count);
        }
    }

    // Example of using the simplified formatters in the dashboard

    pub fn generateDashboard(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        var report = std.ArrayList(u8).init(allocator);
        defer report.deinit();

        const writer = report.writer();

        // Extract statistics
        const current_time: u64 = @intCast(std.time.milliTimestamp());
        const elapsed_ms = current_time - self.start_timestamp;
        const elapsed_seconds = @as(f64, @floatFromInt(elapsed_ms)) / 1000.0;

        // Get order status counts
        const buy_pending = self.buy_pending_count.load(.acquire);
        const buy_partial = self.buy_partial_count.load(.acquire);
        const sell_pending = self.sell_pending_count.load(.acquire);
        const sell_partial = self.sell_partial_count.load(.acquire);

        const total_buy_orders = self.total_buy_orders.load(.acquire);
        const total_sell_orders = self.total_sell_orders.load(.acquire);
        const total_orders = total_buy_orders + total_sell_orders;

        const total_matches = self.total_matches.load(.acquire);
        const total_matched_quantity = self.total_matched_quantity.load(.acquire);

        const orders_per_second = @as(f64, @floatFromInt(total_orders)) / elapsed_seconds;
        const buys_per_second = @as(f64, @floatFromInt(total_buy_orders)) / elapsed_seconds;
        const sells_per_second = @as(f64, @floatFromInt(total_sell_orders)) / elapsed_seconds;
        const matches_per_second = @as(f64, @floatFromInt(total_matches)) / elapsed_seconds;

        // Calculate average execution time
        var avg_execution_time: f64 = 0;
        const trade_count = self.trade_count_for_avg.load(.acquire);
        if (trade_count > 0) {
            const total_execution_time = self.total_execution_time_ms.load(.acquire);
            avg_execution_time = @as(f64, @floatFromInt(total_execution_time)) /
                @as(f64, @floatFromInt(trade_count));
        }

        const bottleneck_analysis = self.flow_analysis.analyzeBottlenecks();

        // Calculate percentages for visualization
        var buy_pending_pct: f64 = 0;
        var buy_partial_pct: f64 = 0;
        const buy_total = buy_pending + buy_partial;
        if (buy_total > 0) {
            buy_pending_pct = @as(f64, @floatFromInt(buy_pending)) / @as(f64, @floatFromInt(buy_total)) * 100.0;
            buy_partial_pct = @as(f64, @floatFromInt(buy_partial)) / @as(f64, @floatFromInt(buy_total)) * 100.0;
        }

        var sell_pending_pct: f64 = 0;
        var sell_partial_pct: f64 = 0;
        const sell_total = sell_pending + sell_partial;
        if (sell_total > 0) {
            sell_pending_pct = @as(f64, @floatFromInt(sell_pending)) / @as(f64, @floatFromInt(sell_total)) * 100.0;
            sell_partial_pct = @as(f64, @floatFromInt(sell_partial)) / @as(f64, @floatFromInt(sell_total)) * 100.0;
        }

        // Dashboard header
        try writer.print("{s}╔══════════════════════════════════════════════════════════════════╗{s}\n", .{ ANSI_CYAN, ANSI_RESET });
        try writer.print("{s}║                   {s}TRADING ENGINE METRICS{s}                         {s}║{s}\n", .{ ANSI_CYAN, ANSI_BOLD, ANSI_RESET, ANSI_CYAN, ANSI_CYAN });
        try writer.print("{s}╚══════════════════════════════════════════════════════════════════╝{s}\n\n", .{ ANSI_CYAN, ANSI_RESET });

        // Config summary
        try writer.print("{s}Configuration:{s}\n", .{ ANSI_BOLD, ANSI_RESET });
        try writer.print("  ├─ Ordering Threads:          {s}{d}{s}\n", .{ ANSI_BLUE, self.config_ordering_threads, ANSI_RESET });
        try writer.print("  └─ Matching Threads:          {s}{d}{s}\n", .{ ANSI_BLUE, self.config_matching_threads, ANSI_RESET });
        try writer.print("  └─ Order Generation Interval: {s}{d}{s} µs\n\n", .{ ANSI_BLUE, self.config_order_interval_us, ANSI_RESET });

        // Runtime information
        try writer.print("{s}Runtime:{s} {}{s}\n\n", .{ ANSI_BOLD, ANSI_RESET, commaFloat2(elapsed_seconds), " seconds" });

        // Order statistics section with comma formatting for both integers and floats
        try writer.print("{s}ORDER STATISTICS{s}\n", .{ ANSI_BOLD, ANSI_RESET });
        try writer.print("  Total Submitted Orders: {s}{}{s} ({s}{}{s} /sec)\n", .{ ANSI_GREEN, commaFormat(total_orders), ANSI_RESET, ANSI_YELLOW, commaFloat2(orders_per_second), ANSI_RESET });
        try writer.print("    ├─ Buy Orders:  {s}{}{s} ({s}{}{s} /sec)\n", .{ ANSI_BLUE, commaFormat(total_buy_orders), ANSI_RESET, ANSI_YELLOW, commaFloat2(buys_per_second), ANSI_RESET });
        try writer.print("    └─ Sell Orders: {s}{}{s} ({s}{}{s} /sec)\n\n", .{ ANSI_RED, commaFormat(total_sell_orders), ANSI_RESET, ANSI_YELLOW, commaFloat2(sells_per_second), ANSI_RESET });

        // Active order breakdown
        try writer.print("{s}ACTIVE ORDER BREAKDOWN{s}\n", .{ ANSI_BOLD, ANSI_RESET });

        // Buy Orders Status
        try writer.print("  Buy Orders Status:\n", .{});
        try writer.print("    Total Active: {s}{}{s}\n", .{ ANSI_BLUE, commaFormat(buy_total), ANSI_RESET });
        try writer.print("    ├─ Pending:        {s}{}{s} ({}{s})\n", .{ ANSI_GREEN, commaFormat(buy_pending), ANSI_RESET, commaFloat2(buy_pending_pct), "%" });
        try writer.print("    └─ Partially Filled: {s}{}{s} ({}{s})\n", .{ ANSI_YELLOW, commaFormat(buy_partial), ANSI_RESET, commaFloat2(buy_partial_pct), "%" });

        // Visualization of buy orders status
        try writer.print("    [", .{});
        const buy_pending_bars = @as(usize, @intFromFloat(buy_pending_pct / 2.5)); // Each bar represents 2.5%
        const buy_partial_bars = @as(usize, @intFromFloat(buy_partial_pct / 2.5));

        for (0..buy_pending_bars) |_| {
            try writer.print("{s}▓{s}", .{ ANSI_GREEN, ANSI_RESET });
        }

        for (0..buy_partial_bars) |_| {
            try writer.print("{s}▓{s}", .{ ANSI_YELLOW, ANSI_RESET });
        }

        const remaining_buy_bars = 40 - buy_pending_bars - buy_partial_bars;
        for (0..remaining_buy_bars) |_| {
            try writer.print(" ", .{});
        }
        try writer.print("]\n\n", .{});

        // Sell Orders Status
        try writer.print("  Sell Orders Status:\n", .{});
        try writer.print("    Total Active: {s}{}{s}\n", .{ ANSI_RED, commaFormat(sell_total), ANSI_RESET });
        try writer.print("    ├─ Pending:        {s}{}{s} ({}{s})\n", .{ ANSI_GREEN, commaFormat(sell_pending), ANSI_RESET, commaFloat2(sell_pending_pct), "%" });
        try writer.print("    └─ Partially Filled: {s}{}{s} ({}{s})\n", .{ ANSI_YELLOW, commaFormat(sell_partial), ANSI_RESET, commaFloat2(sell_partial_pct), "%" });

        // Visualization of sell orders status
        try writer.print("    [", .{});
        const sell_pending_bars = @as(usize, @intFromFloat(sell_pending_pct / 2.5));
        const sell_partial_bars = @as(usize, @intFromFloat(sell_partial_pct / 2.5));

        for (0..sell_pending_bars) |_| {
            try writer.print("{s}▓{s}", .{ ANSI_GREEN, ANSI_RESET });
        }

        for (0..sell_partial_bars) |_| {
            try writer.print("{s}▓{s}", .{ ANSI_YELLOW, ANSI_RESET });
        }

        const remaining_sell_bars = 40 - sell_pending_bars - sell_partial_bars;
        for (0..remaining_sell_bars) |_| {
            try writer.print(" ", .{});
        }
        try writer.print("]\n\n", .{});

        // Calculate fully completed (not partial) orders
        const completed_buy_orders = total_buy_orders - (buy_pending + buy_partial);
        const completed_sell_orders = total_sell_orders - (sell_pending + sell_partial);
        const total_completed_orders = completed_buy_orders + completed_sell_orders;

        // Calculate completion rate per second
        const completed_orders_per_second = @as(f64, @floatFromInt(total_completed_orders)) / elapsed_seconds;

        // Completed orders section
        try writer.print("{s}COMPLETED ORDERS{s}\n", .{ ANSI_BOLD, ANSI_RESET });
        try writer.print("  Total Completed Orders: {s}{}{s} ({s}{}{s} /sec)\n", .{ ANSI_GREEN, commaFormat(total_completed_orders), ANSI_RESET, ANSI_YELLOW, commaFloat2(completed_orders_per_second), ANSI_RESET });
        try writer.print("    ├─ Completed Buy Orders:  {s}{}{s}\n", .{ ANSI_BLUE, commaFormat(completed_buy_orders), ANSI_RESET });
        try writer.print("    └─ Completed Sell Orders: {s}{}{s}\n", .{ ANSI_RED, commaFormat(completed_sell_orders), ANSI_RESET });

        // Match events information
        try writer.print("  Total Match Events: {s}{}{s} ({s}{}{s} /sec)\n", .{ ANSI_GREEN, commaFormat(total_matches), ANSI_RESET, ANSI_YELLOW, commaFloat2(matches_per_second), ANSI_RESET });
        try writer.print("  Total Matched Quantity: {s}{}{s}\n", .{ ANSI_GREEN, commaFormat(total_matched_quantity), ANSI_RESET });

        var time_color = ANSI_GREEN;
        if (avg_execution_time > 100) {
            time_color = ANSI_YELLOW;
        }
        if (avg_execution_time > 500) {
            time_color = ANSI_RED;
        }
        try writer.print("  Average Time to Completion: {s}{}{s} ms\n\n", .{ time_color, commaFloat2(avg_execution_time), ANSI_RESET });

        // Flow analysis section
        try writer.print("{s}FLOW ANALYSIS{s}\n", .{ ANSI_BOLD, ANSI_RESET });

        var ratio_color = ANSI_GREEN;
        if (bottleneck_analysis.buy_sell_ratio > 1.2 or bottleneck_analysis.buy_sell_ratio < 0.8) {
            ratio_color = ANSI_YELLOW;
        }
        if (bottleneck_analysis.buy_sell_ratio > 2.0 or bottleneck_analysis.buy_sell_ratio < 0.5) {
            ratio_color = ANSI_RED;
        }
        try writer.print("  Buy/Sell Ratio: {s}{}{s}\n", .{ ratio_color, commaFloat2(bottleneck_analysis.buy_sell_ratio), ANSI_RESET });

        var efficiency_color = ANSI_GREEN;
        if (bottleneck_analysis.match_efficiency < 0.9) {
            efficiency_color = ANSI_YELLOW;
        }
        if (bottleneck_analysis.match_efficiency < 0.7) {
            efficiency_color = ANSI_RED;
        }
        try writer.print("  Match Efficiency: {s}{}{s}\n", .{ efficiency_color, commaFloat2(bottleneck_analysis.match_efficiency), ANSI_RESET });

        var bottleneck_color = ANSI_GREEN;
        if (!std.mem.eql(u8, bottleneck_analysis.bottleneck, "None")) {
            bottleneck_color = ANSI_RED;
        }
        try writer.print("  Detected Bottleneck: {s}{s}{s}\n", .{ bottleneck_color, bottleneck_analysis.bottleneck, ANSI_RESET });

        try writer.print("\n{s}╔══════════════════════════════════════════════════════════════════╗{s}\n", .{ ANSI_CYAN, ANSI_RESET });
        try writer.print("{s}║                     {s}Press Ctrl+C to exit{s}                         {s}║{s}\n", .{ ANSI_CYAN, ANSI_BOLD, ANSI_RESET, ANSI_CYAN, ANSI_CYAN });
        try writer.print("{s}╚══════════════════════════════════════════════════════════════════╝{s}\n", .{ ANSI_CYAN, ANSI_RESET });

        return report.toOwnedSlice();
    }

    // Add this method to the PerformanceMetrics struct
    pub fn printDashboard(self: *Self, allocator: std.mem.Allocator) !void {
        const report = try self.generateDashboard(allocator);
        defer allocator.free(report);

        // Count the number of lines in the report
        var line_count: usize = 0;
        for (report) |char| {
            if (char == '\n') line_count += 1;
        }

        // ANSI escape sequence to clear the entire screen and move cursor to home
        std.debug.print("\x1B[2J\x1B[H", .{});

        std.debug.print("{s}", .{report});
    }
};

const CommaFormatter = struct {
    value: u64,

    pub fn format(self: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        var temp_buf: [32]u8 = undefined;
        const num_str = try std.fmt.bufPrint(&temp_buf, "{d}", .{self.value});

        // If the number is small enough, no need for commas
        if (num_str.len <= 3) {
            try writer.writeAll(num_str);
            return;
        }

        var i: usize = 0;
        while (i < num_str.len) : (i += 1) {
            // Add a comma every 3 digits, starting from the right
            const pos_from_right = num_str.len - i;
            if (i > 0 and pos_from_right % 3 == 0) {
                try writer.writeByte(',');
            }
            try writer.writeByte(num_str[i]);
        }
    }
};

// Float formatter with comma separators for integer part
const CommaFloat2Formatter = struct {
    value: f64,

    pub fn format(self: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        const abs_value = if (self.value >= 0) self.value else -self.value;

        if (self.value < 0) {
            try writer.writeByte('-');
        }

        const int_part = @as(u64, @intFromFloat(@floor(abs_value)));

        const formatter = CommaFormatter{ .value = int_part };
        try formatter.format("", .{}, writer);

        try writer.writeByte('.');

        const decimal_part = @as(u8, @intFromFloat(@mod(abs_value * 100.0, 100.0)));

        if (decimal_part < 10) {
            try writer.writeByte('0');
        }

        var dec_buf: [2]u8 = undefined;
        const dec_str = try std.fmt.bufPrint(&dec_buf, "{d}", .{decimal_part});
        try writer.writeAll(dec_str);
    }
};

// Helper functions
fn commaFormat(value: u64) CommaFormatter {
    return CommaFormatter{ .value = value };
}

fn commaFloat2(value: f64) CommaFloat2Formatter {
    return CommaFloat2Formatter{ .value = value };
}
