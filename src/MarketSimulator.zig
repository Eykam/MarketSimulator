const std = @import("std");
const Logging = @import("Logging.zig");

const Random = std.Random;
const AtomicU32 = std.atomic.Value(u32);
const AtomicU64 = std.atomic.Value(u64);
const AtomicBool = std.atomic.Value(bool);
const AtomicEnum = std.atomic.Value(OrderStatus);
const AtomicF32 = std.atomic.Value(f32);

const Logger = Logging.Logger;
const LogEntry = Logging.LogEntry;
const PerformanceMetrics = Logging.PerformanceMetrics;

pub const MAX_TICKERS = 1024;
pub const MAX_ORDERS_PER_TICKER = 10000;

pub const OrderType = enum {
    BUY,
    SELL,
};

pub const OrderStatus = enum(u8) {
    PENDING,
    PARTIALLY_FILLED,
    FILLED,
    CANCELED,
};

const Order = struct {
    id: u64,
    ticker_index: u16,
    quantity: AtomicU32,
    price: f32,
    timestamp: u64,
    status: AtomicEnum,

    pub fn init(id: u64, ticker_index: u16, quantity: u32, price: f32) Order {
        return Order{
            .id = id,
            .ticker_index = ticker_index,
            .quantity = AtomicU32.init(quantity),
            .price = price,
            .timestamp = @intCast(std.time.milliTimestamp()),
            .status = AtomicEnum.init(OrderStatus.PENDING),
        };
    }
};

pub const TickerOrderBook = struct {
    const Self = @This();

    ticker_id: u16,
    reference_price: AtomicF32,
    max_price_deviation: f32,
    buy_orders: [MAX_ORDERS_PER_TICKER]Order,
    sell_orders: [MAX_ORDERS_PER_TICKER]Order,
    buy_count: AtomicU32,
    sell_count: AtomicU32,
    needs_matching: AtomicBool,
    match_iterations: u32 = 0,
    matches_per_cycle: u32,
    book_cleanup_interval: u32,

    pub fn init(allocator: std.mem.Allocator, ticker_id: u16, config: SimulationConfig) !*Self {
        const seed = config.seed orelse @as(u64, @intCast(std.time.timestamp()));
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const reference_price = (@as(f32, @floatFromInt(random.uintLessThan(u32, config.max_price * 100))) + 1) / 100.0;

        const ticker_order_book = try allocator.create(Self);
        ticker_order_book.* = Self{
            .ticker_id = ticker_id,
            .max_price_deviation = config.max_price_deviation,
            .reference_price = AtomicF32.init(reference_price),
            .buy_orders = undefined,
            .sell_orders = undefined,
            .buy_count = AtomicU32.init(0),
            .sell_count = AtomicU32.init(0),
            .needs_matching = AtomicBool.init(false),
            .matches_per_cycle = config.matches_per_cycle,
            .book_cleanup_interval = config.book_cleanup_interval,
        };

        return ticker_order_book;
    }

    pub fn addOrder(self: *Self, order: Order, order_type: OrderType, logger: *Logger, metrics: *PerformanceMetrics) !void {
        const ref_price = self.reference_price.load(.acquire);

        if (ref_price > 0) {
            const min_acceptable = ref_price * (1.0 - self.max_price_deviation);
            const max_acceptable = ref_price * (1.0 + self.max_price_deviation);

            if (order.price < min_acceptable or order.price > max_acceptable) {
                const log_entry = LogEntry.init(
                    .ORDER_REJECTED,
                    self.ticker_id,
                    order.id,
                    order_type,
                    order.quantity.load(.acquire),
                    order.price,
                );
                logger.log(log_entry);
                return error.OrderOutsidePriceBands;
            }
        }

        const log_entry = LogEntry.init(
            .ORDER_ADDED,
            self.ticker_id,
            order.id,
            order_type,
            order.quantity.load(.acquire),
            order.price,
        );

        switch (order_type) {
            .BUY => {
                const current_count = self.buy_count.load(.acquire);
                if (current_count >= MAX_ORDERS_PER_TICKER) {
                    return error.OrderBookFull;
                }

                const insertion_idx = findInsertionPoint(self.buy_orders[0..current_count], order.price, true);

                if (insertion_idx < current_count) {
                    std.mem.copyBackwards(Order, self.buy_orders[insertion_idx + 1 .. current_count + 1], self.buy_orders[insertion_idx..current_count]);
                }

                self.buy_orders[insertion_idx] = order;

                _ = self.buy_count.fetchAdd(1, .release);
                metrics.recordBuyOrder();
            },
            .SELL => {
                const current_count = self.sell_count.load(.acquire);
                if (current_count >= MAX_ORDERS_PER_TICKER) {
                    return error.OrderBookFull;
                }

                const insertion_idx = findInsertionPoint(self.sell_orders[0..current_count], order.price, false);

                if (insertion_idx < current_count) {
                    std.mem.copyBackwards(Order, self.sell_orders[insertion_idx + 1 .. current_count + 1], self.sell_orders[insertion_idx..current_count]);
                }

                self.sell_orders[insertion_idx] = order;

                _ = self.sell_count.fetchAdd(1, .release);
                metrics.recordSellOrder();
            },
        }

        logger.log(log_entry);
        self.needs_matching.store(true, .release);
    }

    pub fn matchOrders(self: *TickerOrderBook, logger: *Logger, metrics: *PerformanceMetrics) void {
        if (!self.needs_matching.load(.acquire)) {
            return;
        }

        var match_count: u32 = 0;
        const max_matches_per_cycle = self.matches_per_cycle;

        while (match_count < max_matches_per_cycle) {
            const best_buy_opt = self.findHighestBid();
            const best_sell_opt = self.findLowestAsk();

            if (best_buy_opt == null or best_sell_opt == null) {
                break;
            }

            const best_buy = best_buy_opt.?;
            const best_sell = best_sell_opt.?;

            if (best_buy.price < best_sell.price) {
                break;
            }

            const buy_quantity = best_buy.quantity.load(.acquire);
            const sell_quantity = best_sell.quantity.load(.acquire);

            if (buy_quantity == 0 or sell_quantity == 0) {
                if (buy_quantity == 0) {
                    best_buy.status.store(.FILLED, .release);
                }
                if (sell_quantity == 0) {
                    best_sell.status.store(.FILLED, .release);
                }

                continue;
            }

            const match_quantity = @min(buy_quantity, sell_quantity);

            const current_time = std.time.milliTimestamp();
            const buy_time = best_buy.timestamp;
            const sell_time = best_sell.timestamp;
            const buy_execution_time = @as(u64, @intCast(current_time)) - buy_time;
            const sell_execution_time = @as(u64, @intCast(current_time)) - sell_time;
            const execution_time_ms: u64 = (buy_execution_time + sell_execution_time) / 2;

            const buy_old_status = best_buy.status.load(.acquire);
            const sell_old_status = best_sell.status.load(.acquire);

            _ = best_buy.quantity.fetchSub(match_quantity, .monotonic);
            if (best_buy.quantity.load(.acquire) == 0) {
                best_buy.status.store(.FILLED, .release);
                metrics.safeDecrementPendingOrders(.BUY);
            } else {
                best_buy.status.store(.PARTIALLY_FILLED, .release);
                if (buy_old_status == .PENDING) {
                    metrics.updateOrderStatus(.BUY, .PENDING, .PARTIALLY_FILLED);
                }
            }

            // Update sell order
            _ = best_sell.quantity.fetchSub(match_quantity, .monotonic);
            if (best_sell.quantity.load(.acquire) == 0) {
                best_sell.status.store(.FILLED, .release);
                metrics.safeDecrementPendingOrders(.SELL);
            } else {
                best_sell.status.store(.PARTIALLY_FILLED, .release);
                if (sell_old_status == .PENDING) {
                    metrics.updateOrderStatus(.SELL, .PENDING, .PARTIALLY_FILLED);
                }
            }

            // Log the match
            var log_entry = LogEntry.init(
                .ORDER_MATCHED,
                self.ticker_id,
                0,
                null,
                match_quantity,
                best_sell.price,
            );

            log_entry.formatMessage("Matched: Buy #{d} with Sell #{d}, Price: {d:.2}, Qty: {d}", .{
                best_buy.id,
                best_sell.id,
                best_sell.price,
                match_quantity,
            });
            logger.log(log_entry);

            // Record the match in metrics
            metrics.recordMatch(match_quantity, execution_time_ms);

            if (match_count > 0) {
                self.updateReferencePrice(best_sell.price);
            }

            match_count += 1;
            self.match_iterations += 1;

            if (self.match_iterations >= self.book_cleanup_interval) {
                self.cleanupOrderBook(logger);
                self.match_iterations = 0;
            }
        }

        if (match_count >= max_matches_per_cycle) {
            self.needs_matching.store(true, .release);
        } else {
            self.needs_matching.store(false, .release);
        }
    }

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

    fn findHighestBid(self: *TickerOrderBook) ?*Order {
        const buy_count = self.buy_count.load(.acquire);
        if (buy_count == 0) return null;

        for (0..buy_count) |i| {
            const order = &self.buy_orders[i];
            const order_status = order.status.load(.acquire);
            const order_quantity = order.quantity.load(.acquire);

            if (order_status != .FILLED and order_status != .CANCELED and
                order_quantity > 0)
            {
                return order;
            }
        }

        return null;
    }

    fn findLowestAsk(self: *TickerOrderBook) ?*Order {
        const sell_count = self.sell_count.load(.acquire);
        if (sell_count == 0) return null;

        for (0..sell_count) |i| {
            const order = &self.sell_orders[i];
            const order_status = order.status.load(.acquire);
            const order_quantity = order.quantity.load(.acquire);

            if (order_status != .FILLED and order_status != .CANCELED and
                order_quantity > 0)
            {
                return order;
            }
        }

        return null;
    }

    pub fn cleanupOrderBook(self: *Self, logger: *Logger) void {
        const ref_price = self.reference_price.load(.acquire);
        const min_acceptable = ref_price * (1.0 - self.max_price_deviation);
        const max_acceptable = ref_price * (1.0 + self.max_price_deviation);

        const buy_count = self.buy_count.load(.acquire);
        var valid_buys: u32 = 0;

        // Process buy orders
        for (0..buy_count) |i| {
            const order = &self.buy_orders[i];
            const order_status = order.status.load(.acquire);
            const order_quantity = order.quantity.load(.acquire);

            if (order_status != .FILLED and
                order_status != .CANCELED and
                order_quantity > 0)
            {
                var keep_order = true;

                if (order.price < min_acceptable or order.price > max_acceptable) {
                    order.status.store(.CANCELED, .release);

                    var log_entry = LogEntry.init(
                        .ORDER_CANCELED,
                        self.ticker_id,
                        order.id,
                        null,
                        order_quantity,
                        order.price,
                    );
                    log_entry.formatMessage("Order canceled: outside price band ({d:.2} outside [{d:.2}, {d:.2}])", .{
                        order.price,
                        min_acceptable,
                        max_acceptable,
                    });
                    logger.log(log_entry);

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

        if (valid_buys < buy_count) {
            self.buy_count.store(valid_buys, .release);
        }

        const sell_count = self.sell_count.load(.acquire);
        var valid_sells: u32 = 0;

        for (0..sell_count) |i| {
            const order = &self.sell_orders[i];
            const order_status = order.status.load(.acquire);
            const order_quantity = order.quantity.load(.acquire);

            if (order_status != .FILLED and
                order_status != .CANCELED and
                order_quantity > 0)
            {
                var keep_order = true;

                if (order.price < min_acceptable or order.price > max_acceptable) {
                    order.status.store(.CANCELED, .release);

                    var log_entry = LogEntry.init(
                        .ORDER_CANCELED,
                        self.ticker_id,
                        order.id,
                        null,
                        order_quantity,
                        order.price,
                    );
                    log_entry.formatMessage("Order canceled: outside price band ({d:.2} outside [{d:.2}, {d:.2}])", .{
                        order.price,
                        min_acceptable,
                        max_acceptable,
                    });
                    logger.log(log_entry);

                    keep_order = false;
                }

                if (keep_order) {
                    if (i != valid_sells) {
                        self.sell_orders[valid_sells] = self.sell_orders[i];
                    }
                    valid_sells += 1;
                }
            }
        }

        if (valid_sells < sell_count) {
            self.sell_count.store(valid_sells, .release);
        }
    }

    pub fn getBuyOrders(self: *Self) []Order {
        const count = self.buy_count.load(.acquire);
        return self.buy_orders[0..count];
    }

    pub fn getSellOrders(self: *Self) []Order {
        const count = self.sell_count.load(.acquire);
        return self.sell_orders[0..count];
    }

    fn updateReferencePrice(self: *Self, match_price: f32) void {
        self.reference_price.store(match_price, .release);
    }
};

pub const OrderBook = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ticker_books: [MAX_TICKERS]*TickerOrderBook,
    next_order_id: AtomicU32,
    logger: *Logger,
    metrics: *PerformanceMetrics,
    matcher_threads: []std.Thread,
    run_matcher: AtomicBool,

    pub fn init(allocator: std.mem.Allocator, config: SimulationConfig) !*Self {
        const book = try allocator.create(Self);
        const logger = Logger.init(allocator) catch |err| {
            std.debug.print("Failed to initialize logger => {any}\n", .{err});
            return error.FailedToInitializeLogger;
        };
        const metrics = try PerformanceMetrics.init(allocator, config);

        book.* = Self{
            .allocator = allocator,
            .ticker_books = undefined,
            .next_order_id = AtomicU32.init(1),
            .logger = logger,
            .metrics = metrics,
            .matcher_threads = undefined,
            .run_matcher = AtomicBool.init(true),
        };

        for (0..MAX_TICKERS) |i| {
            book.ticker_books[i] = try TickerOrderBook.init(allocator, @intCast(i), config);
        }
        const threads = try allocator.alloc(std.Thread, config.num_matching_threads);
        book.matcher_threads = threads;

        for (0..threads.len) |idx| {
            threads[idx] = try std.Thread.spawn(.{}, matcherThreadFn, .{ book, idx });
        }

        var log_entry = LogEntry.init(
            .SYSTEM_INFO,
            0,
            0,
            null,
            0,
            0,
        );
        log_entry.formatMessage("Order book initialized with {d} tickers", .{MAX_TICKERS});
        book.logger.log(log_entry);

        return book;
    }

    pub fn deinit(self: *Self) void {
        self.run_matcher.store(false, .release);

        for (self.matcher_threads) |thread| {
            thread.join();
        }

        // Clean up ticker books
        for (0..MAX_TICKERS) |i| {
            self.allocator.destroy(self.ticker_books[i]);
        }

        self.allocator.free(self.matcher_threads);
        self.allocator.destroy(self.logger);
        self.allocator.destroy(self.metrics);
        self.allocator.destroy(self);
    }

    pub fn addOrder(self: *OrderBook, ticker_index: u16, quantity: u32, price: f32, order_type: OrderType) u64 {
        if (ticker_index >= MAX_TICKERS or quantity == 0 or price <= 0) {
            return 0;
        }

        const order_id = self.next_order_id.fetchAdd(1, .monotonic);
        const order = Order.init(order_id, ticker_index, quantity, price);

        self.ticker_books[ticker_index].addOrder(order, order_type, self.logger, self.metrics) catch {
            return 0;
        };

        return order_id;
    }

    fn matcherThreadFn(self: *Self, idx: usize) void {
        const tickers_per_thread = MAX_TICKERS / self.matcher_threads.len;
        const start_idx = idx * tickers_per_thread;
        const end_idx = start_idx + tickers_per_thread;

        var last_status_update = std.time.milliTimestamp();
        const STATUS_UPDATE_INTERVAL_MS = 1000; // Update order status metrics every second

        while (self.run_matcher.load(.acquire)) {
            self.matchThreadsOrders(start_idx, end_idx);

            if (idx == 0) {
                const current_time = std.time.milliTimestamp();
                if (current_time - last_status_update >= STATUS_UPDATE_INTERVAL_MS) {
                    self.metrics.updateOrderBookMetrics(self.ticker_books);
                    last_status_update = current_time;
                }
            }
        }
    }

    pub fn matchThreadsOrders(self: *OrderBook, start: usize, end: usize) void {
        for (start..end) |i| {
            self.ticker_books[i].matchOrders(self.logger, self.metrics);
        }
    }
};

// New simulation configuration struct
pub const SimulationConfig = struct {
    timed_simulation: bool = false,
    runtime_seconds: ?u64 = null,
    order_interval_ns: u64 = 250 * std.time.ns_per_us,
    num_ordering_threads: u32 = 10,
    num_matching_threads: u32 = 1,
    seed: ?u64 = null,
    max_price: u32 = 1000,
    max_price_deviation: f32 = 0.2,
    book_cleanup_interval: u32 = 250,
    matches_per_cycle: u32 = 100,
    max_quantity: u32 = 1000,
    performance_logging: bool = true,
    order_logging: bool = false,
    buy_sell_ratio: f32 = 1.0,
};

const MarketSimulator = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    order_book: *OrderBook,
    prng: std.Random.DefaultPrng,
    running: AtomicBool,
    ordering_threads: []std.Thread,
    config: SimulationConfig,

    pub fn init(allocator: std.mem.Allocator, config: SimulationConfig) !*Self {
        const market_simulator = try allocator.create(Self);

        const order_book = try OrderBook.init(allocator, config);
        const threads = try allocator.alloc(std.Thread, config.num_ordering_threads);

        const seed = config.seed orelse @as(u64, @intCast(std.time.milliTimestamp()));

        market_simulator.* = MarketSimulator{
            .allocator = allocator,
            .order_book = order_book,
            .prng = std.Random.DefaultPrng.init(seed),
            .running = AtomicBool.init(false),
            .ordering_threads = threads,
            .config = config,
        };
        return market_simulator;
    }

    pub fn deinit(self: *Self) void {
        for (self.ordering_threads) |thread| {
            thread.join();
        }

        self.allocator.free(self.ordering_threads);
        self.order_book.deinit();
        self.allocator.destroy(self);
    }

    fn generateRandomOrder(self: *MarketSimulator) void {
        const random = self.prng.random();

        const ticker_index = random.uintLessThan(u16, MAX_TICKERS);
        const quantity = random.uintLessThan(u32, self.config.max_quantity) + 1;

        const ticker_book = self.order_book.ticker_books[ticker_index];
        const ticker_reference_price = ticker_book.reference_price.load(.acquire);

        const price_norm = random.floatNorm(f32) * self.config.max_price_deviation + 1.0;
        const price = price_norm * ticker_reference_price;

        const buy_threshold = @as(f32, 0.5) * self.config.buy_sell_ratio;
        const order_type_int = @intFromBool(random.float(f32) >= buy_threshold);
        const order_type: OrderType = @enumFromInt(order_type_int);

        _ = self.order_book.addOrder(ticker_index, quantity, price, order_type);
    }

    pub fn simulate(self: *Self) void {
        while (self.running.load(.acquire)) {
            self.generateRandomOrder();
            std.time.sleep(self.config.order_interval_ns);
        }
    }

    pub fn start(self: *Self) !void {
        self.running.store(true, .release);

        for (0..self.ordering_threads.len) |i| {
            self.ordering_threads[i] = try std.Thread.spawn(.{}, simulate, .{self});
        }

        var log_entry = LogEntry.init(
            .SYSTEM_INFO,
            0,
            0,
            null,
            0,
            0,
        );
        log_entry.formatMessage("Simulation started with {d} threads", .{self.ordering_threads.len});
        self.order_book.logger.log(log_entry);
    }

    pub fn stop(self: *MarketSimulator) void {
        if (self.running.load(.acquire)) {
            self.running.store(false, .release);

            var log_entry = LogEntry.init(
                .SYSTEM_INFO,
                0,
                0,
                null,
                0,
                0,
            );
            log_entry.formatMessage("Simulation stopped", .{});
            self.order_book.logger.log(log_entry);
        }
    }

    pub fn run(self: *Self) !void {
        try self.start();

        switch (self.config.timed_simulation) {
            true => {
                const runtime = self.config.runtime_seconds orelse 10;
                const start_time = std.time.milliTimestamp();

                while (std.time.milliTimestamp() - start_time < runtime * 1000) {
                    if (self.config.performance_logging) {
                        try self.order_book.metrics.printDashboard(self.allocator);
                    }

                    std.time.sleep(250 * std.time.ns_per_ms);
                }

                self.stop();
                std.debug.print("\n\nSimulation complete!\n", .{});

                if (self.config.performance_logging) try self.order_book.metrics.printDashboard(self.allocator);
            },
            false => {
                while (true) {
                    if (self.config.performance_logging) {
                        try self.order_book.metrics.printDashboard(self.allocator);
                    }
                    std.time.sleep(100 * std.time.ns_per_ms);
                }
            },
        }
    }

    pub fn printOrderReport(self: *Self) !void {
        const report = try self.order_book.logger.generateReport(self.allocator);
        defer self.allocator.free(report);

        std.debug.print("{s}\n", .{report});
    }
};

fn parseCommandLineArgs(allocator: std.mem.Allocator, config: *SimulationConfig) !void {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip the program name
    _ = args.skip();

    // Process arguments
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--ordering-threads=")) {
            const value_str = arg["--ordering-threads=".len..];
            const value = try std.fmt.parseInt(u32, value_str, 10);
            config.*.num_ordering_threads = value;
        } else if (std.mem.startsWith(u8, arg, "--matching-threads=")) {
            const value_str = arg["--matching-threads=".len..];
            const value = try std.fmt.parseInt(u32, value_str, 10);
            config.*.num_matching_threads = value;
        } else if (std.mem.startsWith(u8, arg, "--order-interval=")) {
            const value_str = arg["--order-interval=".len..];
            const value = try std.fmt.parseInt(u64, value_str, 10);
            config.*.order_interval_ns = value * std.time.ns_per_us; // Convert microseconds to nanoseconds
        } else if (std.mem.startsWith(u8, arg, "--runtime=")) {
            const value_str = arg["--runtime=".len..];
            const value = try std.fmt.parseInt(u64, value_str, 10);
            config.*.timed_simulation = true;
            config.*.runtime_seconds = value;
        } else if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            std.process.exit(0);
        }
    }

    // Print configuration summary
    std.debug.print("Running with configuration:\n", .{});
    std.debug.print("  - Ordering threads: {d}\n", .{config.num_ordering_threads});
    std.debug.print("  - Matching threads: {d}\n", .{config.num_matching_threads});
    std.debug.print("  - Order interval: {d} µs\n", .{config.order_interval_ns / std.time.ns_per_us});
    std.debug.print("  - Cleanup interval: {d} matches\n", .{config.book_cleanup_interval});
    std.debug.print("  - Matches per cycle: {d}\n", .{config.matches_per_cycle});
    if (config.timed_simulation) {
        std.debug.print("  - Runtime: {d} seconds\n", .{config.runtime_seconds.?});
    } else {
        std.debug.print("  - Runtime: indefinite (Ctrl+C to stop)\n", .{});
    }
    std.debug.print("\n", .{});
}

fn printUsage() void {
    const usage =
        \\Usage: MarketSimulator [options]
        \\
        \\Options:
        \\  --ordering-threads=N   Set number of order generator threads (default: 10)
        \\  --matching-threads=N   Set number of order matching threads (default: 1)
        \\  --order-interval=N     Set order interval in microseconds (default: 250)
        \\  --runtime=N            Set runtime in seconds (default: indefinite)
        \\  --help                 Display this help message
        \\
    ;
    std.debug.print("{s}\n", .{usage});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Default configuration
    var config = SimulationConfig{};

    // Parse command line arguments
    try parseCommandLineArgs(allocator, &config);

    var simulator = try MarketSimulator.init(
        allocator,
        config,
    );
    defer simulator.deinit();
    try simulator.run();
}
