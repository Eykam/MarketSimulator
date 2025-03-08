// Helper function to check if a string contains a substring
const std = @import("std");
const MarketSimulator = @import("MarketSimulator.zig");

const AtomicU32 = std.atomic.Value(u32);
const AtomicU64 = std.atomic.Value(u64);

const OrderType = MarketSimulator.OrderType;
const OrderStatus = MarketSimulator.OrderStatus;
const OrderBook = MarketSimulator.OrderBook;
const TickerOrderBook = MarketSimulator.TickerOrderBook;
const SimulationConfig = MarketSimulator.SimulationConfig;

fn stringContains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

// Helper function to count active orders in the order book
fn countActiveOrders(ticker_book: anytype, order_type: OrderType) u32 {
    switch (order_type) {
        .BUY => return ticker_book.buy_count.load(.acquire),
        .SELL => return ticker_book.sell_count.load(.acquire),
    }
}

fn createTestConfig() SimulationConfig {
    return SimulationConfig{
        .timed_simulation = false,
        .order_interval_ns = 1 * std.time.ns_per_ms,
        .num_ordering_threads = 0, // No automatic order generation for tests
        .num_matching_threads = 1,
        .seed = 12345, // Fixed seed for reproducibility
        .max_price = 1000,
        .max_price_deviation = 0.2, // 20% deviation allowed
        .book_cleanup_interval = 500,
        .matches_per_cycle = 250,
        .max_quantity = 100,
        .performance_logging = false,
        .order_logging = true,
    };
}

// Test 1: Simple order matching
test "Simple Order Matching" {
    std.debug.print("\n================= Test: 1 Simple Order Matching =================\n", .{});
    // Set up test allocator
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create order book with test configuration
    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker = 0;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    const ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("Ticker reference price: {d:.2}\n", .{ref_price});

    // Use prices within the valid range
    const buy_price = ref_price * 1.05; // 5% above reference
    const sell_price = ref_price * 0.95; // 5% below reference

    // Add a buy order
    const buy_id = order_book.addOrder(ticker, 100, buy_price, .BUY);
    std.debug.print("Created buy order #{d} @ {d:.2}\n", .{ buy_id, buy_price });

    // Add a sell order (should match with buy)
    const sell_id = order_book.addOrder(ticker, 100, sell_price, .SELL);
    std.debug.print("Created sell order #{d} @ {d:.2}\n", .{ sell_id, sell_price });

    // Confirm both orders were added successfully
    try std.testing.expect(buy_id != 0);
    try std.testing.expect(sell_id != 0);

    // Pre-match state
    const pre_match_buys = countActiveOrders(ticker_book, .BUY);
    const pre_match_sells = countActiveOrders(ticker_book, .SELL);
    std.debug.print("Pre-match: {d} buys, {d} sells\n", .{ pre_match_buys, pre_match_sells });

    // Give time for matching to occur
    std.time.sleep(100 * std.time.ns_per_ms);

    // Verify result by checking log
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);
    std.debug.print("{s}\n", .{report});

    // Look for a match event in the log
    try std.testing.expect(stringContains(report, "Matched:"));
    try std.testing.expect(stringContains(report, "Ticker 0"));
    try std.testing.expect(stringContains(report, "Qty 100"));
    ticker_book.cleanupOrderBook(order_book.logger);

    // Post-match state
    const post_match_buys = countActiveOrders(ticker_book, .BUY);
    const post_match_sells = countActiveOrders(ticker_book, .SELL);
    std.debug.print("Post-match: {d} buys, {d} sells\n", .{ post_match_buys, post_match_sells });

    // We expect orders to have matched
    try std.testing.expect(post_match_buys == 0);
    try std.testing.expect(post_match_sells == 0);

    std.debug.print("✅ Passed Test Case\n", .{});
}

// Test 2: Partial order fill
test "Partial Order Fill" {
    std.debug.print("\n================= Test: 2 Partial Order Fill =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker = 1;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    const ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("\nTicker reference price: {d:.2}\n", .{ref_price});

    // Use prices within the valid range
    const buy_price = ref_price * 1.05; // 5% above reference
    const sell_price = ref_price * 0.95; // 5% below reference

    // Add buy order for 200 units
    const buy_id = order_book.addOrder(ticker, 200, buy_price, .BUY);
    std.debug.print("Created buy order #{d} for 200 units @ {d:.2}\n", .{ buy_id, buy_price });

    // Add sell order for 50 units (should match, but partially)
    const sell_id = order_book.addOrder(ticker, 50, sell_price, .SELL);
    std.debug.print("Created sell order #{d} for 50 units @ {d:.2}\n", .{ sell_id, sell_price });

    // Confirm both orders were added successfully
    try std.testing.expect(buy_id != 0);
    try std.testing.expect(sell_id != 0);

    // Give time for matching to occur
    std.time.sleep(100 * std.time.ns_per_ms);

    // Verify results via logs
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);
    std.debug.print("{s}\n", .{report});

    // Check for match events
    try std.testing.expect(stringContains(report, "Matched:"));
    try std.testing.expect(stringContains(report, "Ticker 1"));
    try std.testing.expect(stringContains(report, "Qty 50"));

    ticker_book.cleanupOrderBook(order_book.logger);

    // After partial matching, buy order should still exist but sell should be gone
    const buy_count = countActiveOrders(ticker_book, .BUY);
    const sell_count = countActiveOrders(ticker_book, .SELL);

    std.debug.print("Post-match: {d} buy orders, {d} sell orders\n", .{ buy_count, sell_count });

    // We expect the buy order to still be active
    try std.testing.expectEqual(@as(u32, 1), buy_count);
    // The sell order should be gone (filled)
    try std.testing.expectEqual(@as(u32, 0), sell_count);

    // Check that the buy order has the remaining quantity of 150
    const buy_orders = ticker_book.getBuyOrders();
    if (buy_orders.len > 0) {
        for (buy_orders) |*order| {
            if (order.id == buy_id) {
                const remaining_qty = order.quantity.load(.acquire);
                std.debug.print("Buy order #{d} has {d} units remaining\n", .{ buy_id, remaining_qty });
                try std.testing.expectEqual(@as(u32, 150), remaining_qty);
            }
        }
    }

    std.debug.print("✅ Passed Test Case\n", .{});
}

// Test 3: Price priority matching
test "Price Priority Matching" {
    std.debug.print("\n================= Test: 3 Price Priority Matching =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker = 2;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    const ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("\nTicker reference price: {d:.2}\n", .{ref_price});

    // Calculate valid price range
    const max_deviation = ticker_book.max_price_deviation;
    const min_price = ref_price * (1.0 - max_deviation);
    const max_price = ref_price * (1.0 + max_deviation);
    std.debug.print("Valid price range: {d:.2} - {d:.2}\n", .{ min_price, max_price });

    // Create sell orders at different prices (from highest to lowest)
    const sell_high_price = ref_price * 0.95; // 5% below reference
    const sell_mid_price = ref_price * 0.90; // 10% below reference
    const sell_low_price = ref_price * 0.85; // 15% below reference

    const sell_high_id = order_book.addOrder(ticker, 100, sell_high_price, .SELL);
    const sell_mid_id = order_book.addOrder(ticker, 100, sell_mid_price, .SELL);
    const sell_low_id = order_book.addOrder(ticker, 100, sell_low_price, .SELL);

    std.debug.print("Created sell orders: #{d} @ {d:.2}, #{d} @ {d:.2}, #{d} @ {d:.2}\n", .{ sell_high_id, sell_high_price, sell_mid_id, sell_mid_price, sell_low_id, sell_low_price });

    // Confirm orders were added successfully
    try std.testing.expect(sell_high_id != 0);
    try std.testing.expect(sell_mid_id != 0);
    try std.testing.expect(sell_low_id != 0);

    // Verify we have 3 sell orders before matching
    try std.testing.expectEqual(@as(u32, 3), countActiveOrders(ticker_book, .SELL));

    // Add a buy order at a high price - should match with lowest sell first
    const buy_price = ref_price * 1.1; // 10% above reference
    const buy_id = order_book.addOrder(ticker, 100, buy_price, .BUY);
    std.debug.print("Created buy order #{d} @ {d:.2}\n", .{ buy_id, buy_price });
    try std.testing.expect(buy_id != 0);

    // Give time for matching to occur
    std.time.sleep(100 * std.time.ns_per_ms);

    // Verify results via logs
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);

    // Check that a match occurred
    try std.testing.expect(stringContains(report, "Matched:"));

    ticker_book.cleanupOrderBook(order_book.logger);

    // After matching, we should have 2 sell orders left (the higher priced ones)
    // and the buy order should be gone
    const remaining_sells = countActiveOrders(ticker_book, .SELL);
    const remaining_buys = countActiveOrders(ticker_book, .BUY);

    std.debug.print("Post-match: {d} buy orders, {d} sell orders\n", .{ remaining_buys, remaining_sells });

    // One sell order should be gone (the lowest priced one)
    try std.testing.expectEqual(@as(u32, 2), remaining_sells);
    // The buy order should be gone (filled)
    try std.testing.expectEqual(@as(u32, 0), remaining_buys);

    // Verify which sell orders remain in the book
    const sell_orders = ticker_book.getSellOrders();

    var found_low = false;
    var found_mid = false;
    var found_high = false;

    for (sell_orders) |*order| {
        if (order.id == sell_low_id) found_low = true;
        if (order.id == sell_mid_id) found_mid = true;
        if (order.id == sell_high_id) found_high = true;
    }

    std.debug.print("Remaining orders: low={}, mid={}, high={}\n", .{ found_low, found_mid, found_high });

    // The lowest price sell should be gone (matched)
    try std.testing.expect(!found_low);

    // The higher priced sells should still be there
    try std.testing.expect(found_mid);
    try std.testing.expect(found_high);

    std.debug.print("✅ Passed Test Case\n", .{});
}

test "Time Priority Matching" {
    std.debug.print("\n================= Test: 4 Time Priority Matching =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker = 3;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    const ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("\nTicker reference price: {d:.2}\n", .{ref_price});

    // Use a price within valid range for sell orders
    const sell_price = ref_price * 0.95; // 5% below reference

    // Add two sell orders at the same price
    const sell_first_id = order_book.addOrder(ticker, 100, sell_price, .SELL);
    std.debug.print("Created first sell order #{d} @ {d:.2}\n", .{ sell_first_id, sell_price });
    try std.testing.expect(sell_first_id != 0);

    // Sleep to ensure timestamp difference
    std.time.sleep(100 * std.time.ns_per_ms);

    const sell_second_id = order_book.addOrder(ticker, 100, sell_price, .SELL);
    std.debug.print("Created second sell order #{d} @ {d:.2}\n", .{ sell_second_id, sell_price });
    try std.testing.expect(sell_second_id != 0);

    // Verify we have 2 sell orders before matching
    try std.testing.expectEqual(@as(u32, 2), countActiveOrders(ticker_book, .SELL));

    // Add a buy order for less than total available - should match with first order
    const buy_price = ref_price * 1.05; // 5% above reference
    const buy_id = order_book.addOrder(ticker, 100, buy_price, .BUY);
    std.debug.print("Created buy order #{d} @ {d:.2}\n", .{ buy_id, buy_price });
    try std.testing.expect(buy_id != 0);

    // Give time for matching to occur
    std.time.sleep(250 * std.time.ns_per_ms);

    // Verify results via logs
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);
    std.debug.print("{s}\n", .{report});

    // Check for a match
    try std.testing.expect(stringContains(report, "Matched:"));
    try std.testing.expect(stringContains(report, "Ticker 3"));
    try std.testing.expect(stringContains(report, "Qty 100"));

    ticker_book.cleanupOrderBook(order_book.logger);

    // After matching, one sell order should remain (the second one)
    // and the buy order should be gone (filled)
    const remaining_sells = countActiveOrders(ticker_book, .SELL);
    const remaining_buys = countActiveOrders(ticker_book, .BUY);

    std.debug.print("Post-match: {d} buy orders, {d} sell orders\n", .{ remaining_buys, remaining_sells });

    // The buy order should be gone (filled)
    try std.testing.expectEqual(@as(u32, 0), remaining_buys);
    // One sell order should remain
    try std.testing.expectEqual(@as(u32, 1), remaining_sells);

    // Verify which sell order remains (should be the second one)
    const sell_orders = ticker_book.getSellOrders();

    var found_first = false;
    var found_second = false;

    for (sell_orders) |*order| {
        if (order.id == sell_first_id) found_first = true;
        if (order.id == sell_second_id) found_second = true;
    }

    std.debug.print("Remaining orders: first={}, second={}\n", .{ found_first, found_second });

    // The first sell order should be gone (matched)
    try std.testing.expect(!found_first);

    // The second sell order should still be active
    try std.testing.expect(found_second);

    std.debug.print("✅ Passed Test Case\n", .{});
}

// Test 5: Multiple matches from a single large order
test "Multiple Order Matches" {
    std.debug.print("\n================= Test: 5 Multiple Order Matches =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker = 4;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    const ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("\nTicker reference price: {d:.2}\n", .{ref_price});

    // Use a price within valid range for sell orders
    const sell_price_base = ref_price * 0.95; // 10% below reference

    // Add several small sell orders at slightly different prices
    const sell_ids = [_]u64{
        order_book.addOrder(ticker, 10, sell_price_base * 0.98, .SELL),
        order_book.addOrder(ticker, 20, sell_price_base * 0.99, .SELL),
        order_book.addOrder(ticker, 15, sell_price_base, .SELL),
    };

    std.debug.print("Created sell orders: #{d} (10 units), #{d} (20 units), #{d} (15 units)\n", .{ sell_ids[0], sell_ids[1], sell_ids[2] });

    // Verify all orders were added
    for (sell_ids) |id| {
        try std.testing.expect(id != 0);
    }

    // Verify we have 3 sell orders before matching
    try std.testing.expectEqual(@as(u32, 3), countActiveOrders(ticker_book, .SELL));

    // Add a large buy order that should match with all the sells
    const buy_price = ref_price * 1.05; // 10% above reference
    const buy_id = order_book.addOrder(ticker, 50, buy_price, .BUY);
    std.debug.print("Created buy order #{d} for 50 units @ {d:.2}\n", .{ buy_id, buy_price });
    try std.testing.expect(buy_id != 0);

    // Give time for matching to occur
    std.time.sleep(200 * std.time.ns_per_ms);

    // Verify results - check the log
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);
    std.debug.print("{s}\n", .{report});

    // Count match events in the log
    var match_count: usize = 0;
    var iter = std.mem.splitSequence(u8, report, "Matched:");
    // Skip the first part (before any matches)
    _ = iter.next();

    // Count remaining parts which represent matches
    while (iter.next()) |_| {
        match_count += 1;
    }

    std.debug.print("Found {d} matches in the log\n", .{match_count});
    try std.testing.expect(match_count >= 3);

    ticker_book.cleanupOrderBook(order_book.logger);

    // After multiple matches, all sell orders and the buy should be gone
    // (total sell quantity is 45, buy quantity is 50, so buy should have 5 remaining)
    const remaining_sell_count = countActiveOrders(ticker_book, .SELL);
    const remaining_buy_count = countActiveOrders(ticker_book, .BUY);

    std.debug.print("Remaining sell orders: {d}, buy orders: {d}\n", .{ remaining_sell_count, remaining_buy_count });

    // No sell orders should remain
    try std.testing.expectEqual(@as(u32, 0), remaining_sell_count);

    // The buy order should remain with 5 quantity
    try std.testing.expectEqual(@as(u32, 1), remaining_buy_count);

    // Check the remaining quantity
    const buy_orders = ticker_book.getBuyOrders();
    if (buy_orders.len > 0) {
        for (buy_orders) |*order| {
            if (order.id == buy_id) {
                const remaining_qty = order.quantity.load(.acquire);
                std.debug.print("Buy order #{d} has {d} units remaining\n", .{ buy_id, remaining_qty });
                try std.testing.expectEqual(@as(u32, 5), remaining_qty);
            }
        }
    }

    std.debug.print("✅ Passed Test Case\n", .{});
}

// Test 6: Price boundary enforcement
test "Price Boundary Enforcement" {
    std.debug.print("\n================= Test: 6 Price Boundary Enforcement =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker = 7;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    var ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("\nInitial reference price: {d:.2}\n", .{ref_price});

    // Add orders at the current reference price to establish trading history
    const init_buy_id = order_book.addOrder(ticker, 100, ref_price, .BUY);
    const init_sell_id = order_book.addOrder(ticker, 100, ref_price, .SELL);

    try std.testing.expect(init_buy_id != 0);
    try std.testing.expect(init_sell_id != 0);

    // Give time for matching and reference price establishment
    std.time.sleep(100 * std.time.ns_per_ms);

    // Update our reference price (it may have changed after matching)
    ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("Updated reference price after match: {d:.2}\n", .{ref_price});

    // Calculate the valid price range
    const max_deviation = ticker_book.max_price_deviation;
    const min_price = ref_price * (1.0 - max_deviation);
    const max_price = ref_price * (1.0 + max_deviation);

    std.debug.print("Valid price range: {d:.2} - {d:.2} (deviation: {d:.2})\n", .{ min_price, max_price, max_deviation });

    // Try orders at extreme prices (should be rejected)
    const extreme_buy_price = ref_price * (1.0 + max_deviation + 0.1); // Just beyond max
    const extreme_sell_price = ref_price * (1.0 - max_deviation - 0.1); // Just beyond min

    const far_buy_id = order_book.addOrder(ticker, 50, extreme_buy_price, .BUY);
    const far_sell_id = order_book.addOrder(ticker, 50, extreme_sell_price, .SELL);

    std.debug.print("Created buy order #{d} @ {d:.2} (should be rejected)\n", .{ far_buy_id, extreme_buy_price });
    std.debug.print("Created sell order #{d} @ {d:.2} (should be rejected)\n", .{ far_sell_id, extreme_sell_price });

    // Now try orders that are just within bounds
    const valid_buy_price = ref_price * (1.0 + max_deviation * 0.9); // Just inside max
    const valid_sell_price = ref_price * (1.0 - max_deviation * 0.9); // Just inside min

    const valid_buy_id = order_book.addOrder(ticker, 50, valid_buy_price, .BUY);
    const valid_sell_id = order_book.addOrder(ticker, 50, valid_sell_price, .SELL);

    std.debug.print("Created buy order #{d} @ {d:.2} (should be accepted)\n", .{ valid_buy_id, valid_buy_price });
    std.debug.print("Created sell order #{d} @ {d:.2} (should be accepted)\n", .{ valid_sell_id, valid_sell_price });

    // Give time for processing
    std.time.sleep(100 * std.time.ns_per_ms);

    // Verify results via logs
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);
    std.debug.print("{s}\n", .{report});

    // Check for rejections
    const has_rejections = stringContains(report, "ORDER_REJECTED") or
        stringContains(report, "Rejected");
    try std.testing.expect(has_rejections);

    ticker_book.cleanupOrderBook(order_book.logger);

    // Extreme orders should be rejected (returned with ID 0 or not in the book)
    try std.testing.expect(far_buy_id == 0);
    try std.testing.expect(far_sell_id == 0);

    // Valid orders should be in the book
    try std.testing.expect(valid_buy_id != 0);
    try std.testing.expect(valid_sell_id != 0);

    // Check if valid orders are in the book
    var found_valid_buy = false;
    var found_valid_sell = false;

    const buy_orders = ticker_book.getBuyOrders();
    for (buy_orders) |*order| {
        if (order.id == valid_buy_id) found_valid_buy = true;
    }

    const sell_orders = ticker_book.getSellOrders();
    for (sell_orders) |*order| {
        if (order.id == valid_sell_id) found_valid_sell = true;
    }

    // Valid orders should be in the book (unless they matched)
    if (!found_valid_buy or !found_valid_sell) {
        // If they're not in the book, they might have matched
        try std.testing.expect(stringContains(report, "Matched:"));
    }

    std.debug.print("✅ Passed Test Case\n", .{});
}

// Test 7: Ticker isolation - orders in different tickers don't match
test "Ticker Isolation" {
    std.debug.print("\n================= Test: 7 Ticker Isolation =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var order_book = try OrderBook.init(allocator, createTestConfig());
    defer order_book.deinit();

    const ticker1 = 10;
    const ticker2 = 11;

    const ticker_book1 = order_book.ticker_books[ticker1];
    const ticker_book2 = order_book.ticker_books[ticker2];

    // Get reference prices for both tickers
    const ref_price1 = ticker_book1.reference_price.load(.acquire);
    const ref_price2 = ticker_book2.reference_price.load(.acquire);

    std.debug.print("\nTicker1 ref price: {d:.2}, Ticker2 ref price: {d:.2}\n", .{ ref_price1, ref_price2 });

    // Add orders for ticker1
    const buy_price1 = ref_price1 * 1.05; // 5% above reference
    const buy_id1 = order_book.addOrder(ticker1, 100, buy_price1, .BUY);
    std.debug.print("Created buy order #{d} for ticker {d} @ {d:.2}\n", .{ buy_id1, ticker1, buy_price1 });

    // Add matching sell for ticker2 (shouldn't match with buy from ticker1)
    const sell_price2 = ref_price2 * 0.95; // 5% below reference
    const sell_id2 = order_book.addOrder(ticker2, 100, sell_price2, .SELL);
    std.debug.print("Created sell order #{d} for ticker {d} @ {d:.2}\n", .{ sell_id2, ticker2, sell_price2 });

    // Now add a matching sell for ticker1 (should match with buy from ticker1)
    const sell_price1 = ref_price1 * 0.95; // 5% below reference
    const sell_id1 = order_book.addOrder(ticker1, 100, sell_price1, .SELL);
    std.debug.print("Created sell order #{d} for ticker {d} @ {d:.2}\n", .{ sell_id1, ticker1, sell_price1 });

    // Verify orders were added
    try std.testing.expect(buy_id1 != 0);
    try std.testing.expect(sell_id1 != 0);
    try std.testing.expect(sell_id2 != 0);

    // Record pre-match state
    const pre_ticker1_buys = countActiveOrders(ticker_book1, .BUY);
    const pre_ticker1_sells = countActiveOrders(ticker_book1, .SELL);
    const pre_ticker2_sells = countActiveOrders(ticker_book2, .SELL);

    std.debug.print("Pre-match: Ticker1 has {d} buys, {d} sells; Ticker2 has {d} sells\n", .{ pre_ticker1_buys, pre_ticker1_sells, pre_ticker2_sells });

    // Give time for matching to occur
    std.time.sleep(100 * std.time.ns_per_ms);

    // Verify results via logs
    const report = try order_book.logger.generateReport(allocator);
    defer allocator.free(report);
    std.debug.print("{s}\n", .{report});

    // Look for matches for ticker1
    var ticker1_matched = false;
    var ticker2_matched = false;
    var lines = std.mem.splitSequence(u8, report, "\n");

    while (lines.next()) |line| {
        // Check if the line shows a match for ticker1
        if (stringContains(line, "Matched:") and stringContains(line, "Ticker 10")) {
            ticker1_matched = true;
        }
        // Check if the line shows a match for ticker2
        if (stringContains(line, "Matched:") and stringContains(line, "Ticker 11")) {
            ticker2_matched = true;
        }
    }

    // We expect matches for ticker1 but not ticker2
    try std.testing.expect(ticker1_matched);
    try std.testing.expect(!ticker2_matched);

    ticker_book1.cleanupOrderBook(order_book.logger);
    ticker_book2.cleanupOrderBook(order_book.logger);

    // Post-match state
    const post_ticker1_buys = countActiveOrders(ticker_book1, .BUY);
    const post_ticker1_sells = countActiveOrders(ticker_book1, .SELL);
    const post_ticker2_sells = countActiveOrders(ticker_book2, .SELL);

    std.debug.print("Post-match: Ticker1 has {d} buys, {d} sells; Ticker2 has {d} sells\n", .{ post_ticker1_buys, post_ticker1_sells, post_ticker2_sells });

    // Ticker 1's orders should have matched
    try std.testing.expect(post_ticker1_buys < pre_ticker1_buys);
    try std.testing.expect(post_ticker1_sells < pre_ticker1_sells);

    // Ticker 2's orders should remain unchanged
    try std.testing.expectEqual(pre_ticker2_sells, post_ticker2_sells);

    std.debug.print("✅ Passed Test Case\n", .{});
}

const ThreadContext = struct {
    order_book: *OrderBook,
    ticker: u16,
    ref_price: f32,
    is_buy: bool,
    num_orders: u32,
    thread_id: u32,
    order_count: *AtomicU32,
    match_count: *AtomicU32,
};

// Thread function for order generation
fn generateOrdersThread(ctx: *ThreadContext) void {
    // Create a deterministic PRNG for this thread
    var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(ctx.thread_id * 1000 + 1)));
    const random = prng.random();

    var orders_created: u32 = 0;
    for (0..ctx.num_orders) |i| {
        // Use a price that will definitely match
        var price: f32 = undefined;
        if (ctx.is_buy) {
            price = ctx.ref_price * (1.0 + random.float(f32) * 0.05); // 0-5% above reference
        } else {
            price = ctx.ref_price * (1.0 - random.float(f32) * 0.05); // 0-5% below reference
        }

        // Use small quantities for higher throughput
        const quantity = random.intRangeAtMost(u32, 5, 20);

        // Add the order
        const order_type: OrderType = if (ctx.is_buy) .BUY else .SELL;
        const id = ctx.order_book.addOrder(ctx.ticker, quantity, price, order_type);

        if (id != 0) {
            orders_created += 1;
            _ = ctx.order_count.fetchAdd(1, .monotonic);
        }

        // Occasionally check for matches
        if (i % 100 == 0) {
            // Log progress every 100 orders
            std.debug.print("Thread {d} ({s}) created {d} orders\n", .{ ctx.thread_id, if (ctx.is_buy) "BUY" else "SELL", orders_created });

            // Sleep briefly to give matcher a chance to catch up
            std.time.sleep(1 * std.time.ns_per_ms);
        }
    }

    std.debug.print("Thread {d} ({s}) completed creating {d} orders\n", .{ ctx.thread_id, if (ctx.is_buy) "BUY" else "SELL", orders_created });
}

test "Concurrent Order Generation" {
    std.debug.print("\n================= Test 8: Concurrent Order Generation =================\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create the order book with standard configuration
    var config = createTestConfig();
    config.num_matching_threads = 4;
    config.matches_per_cycle = 100; // Allow matching to process batches

    var order_book = try OrderBook.init(allocator, config);
    defer order_book.deinit();

    // Use a single ticker for all orders to test concurrency on that ticker
    const ticker = 90;
    const ticker_book = order_book.ticker_books[ticker];

    // Get the current reference price
    const ref_price = ticker_book.reference_price.load(.acquire);
    std.debug.print("Reference price: {d:.2}\n", .{ref_price});

    // Create initial orders to establish a reference price
    const init_buy_id = order_book.addOrder(ticker, 100, ref_price, .BUY);
    const init_sell_id = order_book.addOrder(ticker, 100, ref_price, .SELL);

    try std.testing.expect(init_buy_id != 0);
    try std.testing.expect(init_sell_id != 0);

    // Allow initial orders to match
    std.time.sleep(100 * std.time.ns_per_ms);

    // Launch multiple threads for concurrent order generation
    const num_threads = 6; // 3 buy threads, 3 sell threads
    const orders_per_thread = 1000; // Each thread generates 1000 orders

    // Create atomic counters for tracking
    var buy_count = AtomicU32.init(0);
    var sell_count = AtomicU32.init(0);
    var match_count = AtomicU32.init(0);

    var contexts: [num_threads]ThreadContext = undefined;
    var threads: [num_threads]std.Thread = undefined;

    std.debug.print("Launching {d} threads to generate orders\n", .{num_threads});

    // Initialize and launch threads
    for (0..num_threads) |i| {
        const is_buy = i % 2 == 0;
        contexts[i] = ThreadContext{
            .order_book = order_book,
            .ticker = ticker,
            .ref_price = ref_price,
            .is_buy = is_buy,
            .num_orders = orders_per_thread,
            .thread_id = @intCast(i),
            .order_count = if (is_buy) &buy_count else &sell_count,
            .match_count = &match_count,
        };

        threads[i] = try std.Thread.spawn(.{}, generateOrdersThread, .{&contexts[i]});
    }

    // Wait for all generator threads to complete
    for (threads) |thread| {
        thread.join();
    }

    std.debug.print("\nAll order generation threads completed\n", .{});

    // Wait for matching to catch up
    std.debug.print("Waiting for matcher to catch up...\n", .{});
    std.time.sleep(2000 * std.time.ns_per_ms);

    // Retrieve performance metrics
    const metrics = order_book.metrics;
    const total_matches = metrics.total_matches.load(.acquire);
    const total_buy_orders = metrics.total_buy_orders.load(.acquire);
    const total_sell_orders = metrics.total_sell_orders.load(.acquire);

    // Get the current state of the order book
    const active_buys = countActiveOrders(ticker_book, .BUY);
    const active_sells = countActiveOrders(ticker_book, .SELL);

    // Force a cleanup of the order book
    ticker_book.cleanupOrderBook(order_book.logger);

    // Get the state after cleanup
    const post_cleanup_buys = countActiveOrders(ticker_book, .BUY);
    const post_cleanup_sells = countActiveOrders(ticker_book, .SELL);

    // Print final stats
    std.debug.print("\nFinal Results:\n", .{});
    std.debug.print("  Total orders created: {d} buys, {d} sells\n", .{ buy_count.load(.acquire), sell_count.load(.acquire) });
    std.debug.print("  Orders in metrics: {d} buys, {d} sells\n", .{ total_buy_orders, total_sell_orders });
    std.debug.print("  Total matches: {d}\n", .{total_matches});
    std.debug.print("  Active orders before cleanup: {d} buys, {d} sells\n", .{ active_buys, active_sells });
    std.debug.print("  Active orders after cleanup: {d} buys, {d} sells\n", .{ post_cleanup_buys, post_cleanup_sells });

    // Expected total orders (allowing for some rejections)
    const expected_orders = (num_threads / 2) * orders_per_thread; // Half buy, half sell

    // Verify metrics
    try std.testing.expect(buy_count.load(.acquire) > expected_orders * 0.9);
    try std.testing.expect(sell_count.load(.acquire) > expected_orders * 0.9);

    // Verify matching - at least 80% of orders should have matched
    const total_expected_matches: f32 = @floatFromInt(@min(buy_count.load(.acquire), sell_count.load(.acquire)));
    try std.testing.expect(total_matches > @as(u64, @intFromFloat(total_expected_matches * 0.8)));

    // Check balance - there shouldn't be a huge imbalance of buy vs sell orders
    const order_imbalance: f32 = if (post_cleanup_buys > post_cleanup_sells)
        @floatFromInt(post_cleanup_buys - post_cleanup_sells)
    else
        @floatFromInt(post_cleanup_sells - post_cleanup_buys);

    // The imbalance should not be more than a small percentage of total orders
    try std.testing.expect(order_imbalance < @as(f32, @floatFromInt(total_buy_orders + total_sell_orders)) * 0.1);

    std.debug.print("✅ Passed Test Case\n", .{});
}
