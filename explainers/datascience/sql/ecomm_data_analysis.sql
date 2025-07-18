-- =====================================================
-- SAMPLE E-COMMERCE DATABASE SCHEMA
-- =====================================================

-- Table definitions with partitioning and indexing strategies
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    registration_date DATE,
    country VARCHAR(50),
    city VARCHAR(100),
    birth_date DATE,
    customer_tier VARCHAR(20) DEFAULT 'BRONZE' -- BRONZE, SILVER, GOLD, PLATINUM
);

-- Partitioned orders table by date for performance
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    customer_id BIGINT,
    order_date DATE,
    total_amount DECIMAL(12,2),
    order_status VARCHAR(20), -- PENDING, PROCESSING, SHIPPED, DELIVERED, CANCELLED
    shipping_country VARCHAR(50),
    payment_method VARCHAR(50),
    discount_amount DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

CREATE TABLE order_items (
    item_id BIGINT PRIMARY KEY,
    order_id BIGINT,
    product_id BIGINT,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(12,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    product_name VARCHAR(255),
    category_id INT,
    brand VARCHAR(100),
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    stock_quantity INT,
    weight_kg DECIMAL(8,3),
    created_date DATE
);

CREATE TABLE categories (
    category_id INT PRIMARY KEY,
    category_name VARCHAR(100),
    parent_category_id INT,
    commission_rate DECIMAL(5,4)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Customer indexes
CREATE INDEX idx_customers_country_city ON customers(country, city);
CREATE INDEX idx_customers_registration_date ON customers(registration_date);
CREATE INDEX idx_customers_tier ON customers(customer_tier);

-- Order indexes
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_orders_status_date ON orders(order_status, order_date);
CREATE INDEX idx_orders_country_date ON orders(shipping_country, order_date);
CREATE INDEX idx_orders_amount ON orders(total_amount);

-- Order items indexes
CREATE INDEX idx_order_items_product ON order_items(product_id, order_id);
CREATE INDEX idx_order_items_order ON order_items(order_id);

-- Product indexes
CREATE INDEX idx_products_category_price ON products(category_id, price);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_stock ON products(stock_quantity);

-- =====================================================
-- QUERY 1: CUSTOMER LIFETIME VALUE WITH RANKING
-- =====================================================

WITH customer_metrics AS (
    SELECT
        c.customer_id,
        c.email,
        c.country,
        c.customer_tier,
        COUNT(o.order_id) as total_orders,
        SUM(o.total_amount) as lifetime_value,
        AVG(o.total_amount) as avg_order_value,
        MAX(o.order_date) as last_order_date,
        MIN(o.order_date) as first_order_date,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) as customer_lifespan_days
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_status IN ('DELIVERED', 'SHIPPED')
    GROUP BY c.customer_id, c.email, c.country, c.customer_tier
),
ranked_customers AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY lifetime_value DESC) as global_rank,
        ROW_NUMBER() OVER (PARTITION BY country ORDER BY lifetime_value DESC) as country_rank,
        ROW_NUMBER() OVER (PARTITION BY customer_tier ORDER BY lifetime_value DESC) as tier_rank,
        NTILE(10) OVER (ORDER BY lifetime_value DESC) as value_decile,
        PERCENT_RANK() OVER (ORDER BY lifetime_value) as value_percentile
    FROM customer_metrics
    WHERE total_orders >= 2
)
SELECT
    customer_id,
    email,
    country,
    customer_tier,
    total_orders,
    ROUND(lifetime_value, 2) as lifetime_value,
    ROUND(avg_order_value, 2) as avg_order_value,
    customer_lifespan_days,
    global_rank,
    country_rank,
    tier_rank,
    value_decile,
    ROUND(value_percentile * 100, 2) as value_percentile
FROM ranked_customers
WHERE global_rank <= 1000 OR country_rank <= 50
ORDER BY lifetime_value DESC;

-- =====================================================
-- QUERY 2: COHORT ANALYSIS WITH RETENTION METRICS
-- =====================================================

WITH monthly_cohorts AS (
    SELECT
        customer_id,
        DATE_FORMAT(MIN(order_date), '%Y-%m') as cohort_month,
        MIN(order_date) as first_order_date
    FROM orders
    WHERE order_status IN ('DELIVERED', 'SHIPPED')
    GROUP BY customer_id
),
customer_orders AS (
    SELECT
        o.customer_id,
        o.order_date,
        mc.cohort_month,
        mc.first_order_date,
        TIMESTAMPDIFF(MONTH, mc.first_order_date, o.order_date) as period_offset
    FROM orders o
    JOIN monthly_cohorts mc ON o.customer_id = mc.customer_id
    WHERE o.order_status IN ('DELIVERED', 'SHIPPED')
),
cohort_data AS (
    SELECT
        cohort_month,
        period_offset,
        COUNT(DISTINCT customer_id) as customers_in_period
    FROM customer_orders
    GROUP BY cohort_month, period_offset
),
cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM monthly_cohorts
    GROUP BY cohort_month
)
SELECT
    cd.cohort_month,
    cs.cohort_size,
    cd.period_offset,
    cd.customers_in_period,
    ROUND(cd.customers_in_period * 100.0 / cs.cohort_size, 2) as retention_rate,
    SUM(cd.customers_in_period) OVER (
        PARTITION BY cd.cohort_month
        ORDER BY cd.period_offset
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_customers
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
WHERE cs.cohort_size >= 50 -- Only include cohorts with meaningful size
ORDER BY cd.cohort_month, cd.period_offset;

-- =====================================================
-- QUERY 3: PRODUCT PERFORMANCE WITH ADVANCED ANALYTICS
-- =====================================================

WITH product_sales AS (
    SELECT
        p.product_id,
        p.product_name,
        p.category_id,
        c.category_name,
        p.brand,
        p.price,
        p.cost,
        SUM(oi.quantity) as total_quantity_sold,
        SUM(oi.total_price) as total_revenue,
        COUNT(DISTINCT o.order_id) as total_orders,
        COUNT(DISTINCT o.customer_id) as unique_customers,
        AVG(oi.unit_price) as avg_selling_price,
        SUM(oi.total_price - (p.cost * oi.quantity)) as total_profit
    FROM products p
    JOIN categories c ON p.category_id = c.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status IN ('DELIVERED', 'SHIPPED')
      AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
    GROUP BY p.product_id, p.product_name, p.category_id, c.category_name, p.brand, p.price, p.cost
),
product_rankings AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) as revenue_rank,
        ROW_NUMBER() OVER (ORDER BY total_profit DESC) as profit_rank,
        ROW_NUMBER() OVER (ORDER BY total_quantity_sold DESC) as volume_rank,
        ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY total_revenue DESC) as category_rank,
        ROW_NUMBER() OVER (PARTITION BY brand ORDER BY total_revenue DESC) as brand_rank,
        ROUND(total_profit / total_revenue * 100, 2) as profit_margin_pct,
        ROUND(total_revenue / total_quantity_sold, 2) as revenue_per_unit,
        LAG(total_revenue) OVER (ORDER BY total_revenue DESC) as prev_product_revenue,
        LEAD(total_revenue) OVER (ORDER BY total_revenue DESC) as next_product_revenue
    FROM product_sales
)
SELECT
    product_id,
    product_name,
    category_name,
    brand,
    price,
    total_quantity_sold,
    ROUND(total_revenue, 2) as total_revenue,
    ROUND(total_profit, 2) as total_profit,
    profit_margin_pct,
    revenue_per_unit,
    unique_customers,
    revenue_rank,
    profit_rank,
    volume_rank,
    category_rank,
    brand_rank,
    CASE
        WHEN revenue_rank <= 100 THEN 'TOP_PERFORMER'
        WHEN profit_margin_pct >= 30 THEN 'HIGH_MARGIN'
        WHEN total_quantity_sold >= 1000 THEN 'HIGH_VOLUME'
        ELSE 'STANDARD'
    END as performance_tier
FROM product_rankings
WHERE revenue_rank <= 500 OR profit_rank <= 500
ORDER BY total_revenue DESC;

-- =====================================================
-- QUERY 4: SEASONAL SALES ANALYSIS WITH FORECASTING
-- =====================================================

WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(o.order_date, '%Y-%m') as sales_month,
        YEAR(o.order_date) as sales_year,
        MONTH(o.order_date) as sales_month_num,
        COUNT(DISTINCT o.order_id) as total_orders,
        COUNT(DISTINCT o.customer_id) as unique_customers,
        SUM(o.total_amount) as total_revenue,
        AVG(o.total_amount) as avg_order_value
    FROM orders o
    WHERE o.order_status IN ('DELIVERED', 'SHIPPED')
    GROUP BY DATE_FORMAT(o.order_date, '%Y-%m'), YEAR(o.order_date), MONTH(o.order_date)
),
seasonal_metrics AS (
    SELECT *,
        LAG(total_revenue, 1) OVER (ORDER BY sales_month) as prev_month_revenue,
        LAG(total_revenue, 12) OVER (ORDER BY sales_month) as same_month_prev_year,
        AVG(total_revenue) OVER (
            ORDER BY sales_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as three_month_avg,
        AVG(total_revenue) OVER (
            ORDER BY sales_month
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) as twelve_month_avg
    FROM monthly_sales
),
growth_analysis AS (
    SELECT *,
        CASE
            WHEN prev_month_revenue IS NOT NULL
            THEN ROUND((total_revenue - prev_month_revenue) / prev_month_revenue * 100, 2)
            ELSE NULL
        END as mom_growth_pct,
        CASE
            WHEN same_month_prev_year IS NOT NULL
            THEN ROUND((total_revenue - same_month_prev_year) / same_month_prev_year * 100, 2)
            ELSE NULL
        END as yoy_growth_pct,
        ROUND(total_revenue / three_month_avg * 100, 2) as seasonal_index
    FROM seasonal_metrics
)
SELECT
    sales_month,
    sales_year,
    sales_month_num,
    total_orders,
    unique_customers,
    ROUND(total_revenue, 2) as total_revenue,
    ROUND(avg_order_value, 2) as avg_order_value,
    mom_growth_pct,
    yoy_growth_pct,
    seasonal_index,
    ROUND(three_month_avg, 2) as three_month_avg,
    ROUND(twelve_month_avg, 2) as twelve_month_avg,
    CASE
        WHEN seasonal_index >= 110 THEN 'PEAK'
        WHEN seasonal_index <= 90 THEN 'LOW'
        ELSE 'NORMAL'
    END as seasonal_category
FROM growth_analysis
ORDER BY sales_month DESC;

-- =====================================================
-- QUERY 5: ADVANCED CUSTOMER SEGMENTATION
-- =====================================================

WITH customer_behavior AS (
    SELECT
        c.customer_id,
        c.registration_date,
        c.country,
        c.customer_tier,
        COUNT(o.order_id) as total_orders,
        SUM(o.total_amount) as total_spent,
        AVG(o.total_amount) as avg_order_value,
        MAX(o.order_date) as last_order_date,
        MIN(o.order_date) as first_order_date,
        DATEDIFF(CURRENT_DATE, MAX(o.order_date)) as days_since_last_order,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) as customer_lifespan_days,
        COUNT(DISTINCT YEAR(o.order_date)) as active_years,
        COUNT(DISTINCT p.category_id) as categories_purchased
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status IN ('DELIVERED', 'SHIPPED')
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    GROUP BY c.customer_id, c.registration_date, c.country, c.customer_tier
),
rfm_analysis AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY days_since_last_order) as recency_score,
        NTILE(5) OVER (ORDER BY total_orders DESC) as frequency_score,
        NTILE(5) OVER (ORDER BY total_spent DESC) as monetary_score
    FROM customer_behavior
    WHERE total_orders > 0
),
customer_segments AS (
    SELECT *,
        CONCAT(recency_score, frequency_score, monetary_score) as rfm_score,
        CASE
            WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'CHAMPIONS'
            WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'LOYAL_CUSTOMERS'
            WHEN recency_score >= 4 AND frequency_score <= 2 THEN 'NEW_CUSTOMERS'
            WHEN recency_score >= 3 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'POTENTIAL_LOYALISTS'
            WHEN recency_score >= 3 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'PROMISING'
            WHEN recency_score <= 2 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'AT_RISK'
            WHEN recency_score <= 2 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'CANNOT_LOSE'
            WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'HIBERNATING'
            ELSE 'LOST'
        END as segment
    FROM rfm_analysis
)
SELECT
    segment,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as segment_percentage,
    ROUND(AVG(total_spent), 2) as avg_total_spent,
    ROUND(AVG(avg_order_value), 2) as avg_order_value,
    ROUND(AVG(total_orders), 1) as avg_total_orders,
    ROUND(AVG(days_since_last_order), 1) as avg_days_since_last_order,
    ROUND(AVG(customer_lifespan_days), 1) as avg_customer_lifespan,
    ROUND(SUM(total_spent), 2) as total_segment_revenue,
    ROUND(SUM(total_spent) * 100.0 / SUM(SUM(total_spent)) OVER (), 2) as revenue_percentage
FROM customer_segments
GROUP BY segment
ORDER BY total_segment_revenue DESC;

-- =====================================================
-- QUERY 6: INVENTORY OPTIMIZATION WITH DEMAND FORECASTING
-- =====================================================

WITH product_demand AS (
    SELECT
        p.product_id,
        p.product_name,
        p.category_id,
        c.category_name,
        p.stock_quantity,
        p.price,
        p.cost,
        COUNT(DISTINCT o.order_id) as order_frequency,
        SUM(oi.quantity) as total_quantity_sold,
        AVG(oi.quantity) as avg_quantity_per_order,
        STDDEV(oi.quantity) as quantity_stddev,
        SUM(oi.total_price) as total_revenue,
        MAX(o.order_date) as last_sold_date,
        DATEDIFF(CURRENT_DATE, MAX(o.order_date)) as days_since_last_sale
    FROM products p
    JOIN categories c ON p.category_id = c.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.order_status IN ('DELIVERED', 'SHIPPED')
        AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 6 MONTH)
    GROUP BY p.product_id, p.product_name, p.category_id, c.category_name,
             p.stock_quantity, p.price, p.cost
),
demand_metrics AS (
    SELECT *,
        CASE
            WHEN total_quantity_sold IS NULL THEN 0
            ELSE ROUND(total_quantity_sold / 26.0, 2) -- 26 weeks in 6 months
        END as avg_weekly_demand,
        CASE
            WHEN stock_quantity > 0 AND total_quantity_sold > 0
            THEN ROUND(stock_quantity / (total_quantity_sold / 26.0), 1)
            ELSE 9999
        END as weeks_of_stock,
        ROUND((total_revenue - (cost * total_quantity_sold)) / total_revenue * 100, 2) as profit_margin_pct
    FROM product_demand
),
inventory_classification AS (
    SELECT *,
        CASE
            WHEN weeks_of_stock <= 2 THEN 'CRITICAL_LOW'
            WHEN weeks_of_stock <= 4 THEN 'LOW_STOCK'
            WHEN weeks_of_stock <= 12 THEN 'OPTIMAL'
            WHEN weeks_of_stock <= 26 THEN 'OVERSTOCKED'
            ELSE 'EXCESS'
        END as stock_status,
        CASE
            WHEN avg_weekly_demand >= 10 AND profit_margin_pct >= 20 THEN 'HIGH_PRIORITY'
            WHEN avg_weekly_demand >= 5 AND profit_margin_pct >= 15 THEN 'MEDIUM_PRIORITY'
            WHEN avg_weekly_demand >= 1 THEN 'LOW_PRIORITY'
            ELSE 'SLOW_MOVING'
        END as reorder_priority,
        NTILE(4) OVER (ORDER BY total_revenue DESC) as revenue_quartile,
        NTILE(4) OVER (ORDER BY avg_weekly_demand DESC) as demand_quartile
    FROM demand_metrics
)
SELECT
    product_id,
    product_name,
    category_name,
    stock_quantity,
    price,
    total_quantity_sold,
    avg_weekly_demand,
    weeks_of_stock,
    stock_status,
    reorder_priority,
    profit_margin_pct,
    revenue_quartile,
    demand_quartile,
    days_since_last_sale,
    CASE
        WHEN stock_status = 'CRITICAL_LOW' THEN avg_weekly_demand * 8
        WHEN stock_status = 'LOW_STOCK' THEN avg_weekly_demand * 6
        WHEN stock_status = 'EXCESS' THEN 0
        ELSE avg_weekly_demand * 4
    END as suggested_reorder_quantity
FROM inventory_classification
WHERE (stock_status IN ('CRITICAL_LOW', 'LOW_STOCK') AND reorder_priority != 'SLOW_MOVING')
   OR (stock_status = 'EXCESS' AND reorder_priority = 'SLOW_MOVING')
ORDER BY
    CASE stock_status
        WHEN 'CRITICAL_LOW' THEN 1
        WHEN 'LOW_STOCK' THEN 2
        WHEN 'EXCESS' THEN 3
        ELSE 4
    END,
    avg_weekly_demand DESC;

-- =====================================================
-- QUERY 7: GEOGRAPHICAL SALES PERFORMANCE
-- =====================================================

WITH country_sales AS (
    SELECT
        o.shipping_country,
        COUNT(DISTINCT o.customer_id) as unique_customers,
        COUNT(o.order_id) as total_orders,
        SUM(o.total_amount) as total_revenue,
        AVG(o.total_amount) as avg_order_value,
        SUM(o.discount_amount) as total_discounts,
        COUNT(DISTINCT DATE_FORMAT(o.order_date, '%Y-%m')) as active_months,
        MIN(o.order_date) as first_order_date,
        MAX(o.order_date) as last_order_date
    FROM orders o
    WHERE o.order_status IN ('DELIVERED', 'SHIPPED')
      AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
    GROUP BY o.shipping_country
),
country_metrics AS (
    SELECT *,
        ROUND(total_revenue / unique_customers, 2) as revenue_per_customer,
        ROUND(total_orders / unique_customers, 2) as orders_per_customer,
        ROUND(total_discounts / total_revenue * 100, 2) as discount_rate_pct,
        ROUND(total_revenue / active_months, 2) as avg_monthly_revenue,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) as revenue_rank,
        ROW_NUMBER() OVER (ORDER BY unique_customers DESC) as customer_rank,
        ROW_NUMBER() OVER (ORDER BY avg_order_value DESC) as aov_rank
    FROM country_sales
),
performance_tiers AS (
    SELECT *,
        CASE
            WHEN revenue_rank <= 5 THEN 'TIER_1'
            WHEN revenue_rank <= 15 THEN 'TIER_2'
            WHEN revenue_rank <= 30 THEN 'TIER_3'
            ELSE 'TIER_4'
        END as market_tier,
        PERCENT_RANK() OVER (ORDER BY total_revenue) as revenue_percentile,
        total_revenue / SUM(total_revenue) OVER () * 100 as revenue_share_pct
    FROM country_metrics
)
SELECT
    shipping_country,
    market_tier,
    unique_customers,
    total_orders,
    ROUND(total_revenue, 2) as total_revenue,
    ROUND(revenue_share_pct, 2) as revenue_share_pct,
    ROUND(avg_order_value, 2) as avg_order_value,
    revenue_per_customer,
    orders_per_customer,
    discount_rate_pct,
    ROUND(avg_monthly_revenue, 2) as avg_monthly_revenue,
    revenue_rank,
    customer_rank,
    aov_rank,
    ROUND(revenue_percentile * 100, 2) as revenue_percentile,
    active_months,
    first_order_date,
    last_order_date
FROM performance_tiers
ORDER BY total_revenue DESC;

-- =====================================================
-- QUERY 8: CUSTOMER CHURN PREDICTION ANALYSIS
-- =====================================================

WITH customer_activity AS (
    SELECT
        c.customer_id,
        c.registration_date,
        c.country,
        c.customer_tier,
        COUNT(o.order_id) as total_orders,
        SUM(o.total_amount) as total_spent,
        AVG(o.total_amount) as avg_order_value,
        MAX(o.order_date) as last_order_date,
        MIN(o.order_date) as first_order_date,
        DATEDIFF(CURRENT_DATE, MAX(o.order_date)) as days_since_last_order,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) as customer_lifespan_days,
        AVG(DATEDIFF(o.order_date, LAG(o.order_date) OVER (PARTITION BY c.customer_id ORDER BY o.order_date))) as avg_days_between_orders,
        COUNT(DISTINCT YEAR(o.order_date)) as active_years,
        COUNT(DISTINCT MONTH(o.order_date)) as active_months
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.order_status IN ('DELIVERED', 'SHIPPED')
    WHERE c.registration_date <= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
    GROUP BY c.customer_id, c.registration_date, c.country, c.customer_tier
),
churn_indicators AS (
    SELECT *,
        CASE
            WHEN total_orders = 0 THEN 1
            WHEN days_since_last_order > (avg_days_between_orders * 2) THEN 1
            WHEN days_since_last_order > 180 THEN 1
            ELSE 0
        END as is_likely_churned,
        CASE
            WHEN total_orders = 0 THEN 'NEVER_PURCHASED'
            WHEN days_since_last_order > 365 THEN 'LONG_DORMANT'
            WHEN days_since_last_order > 180 THEN 'DORMANT'
            WHEN days_since_last_order > (avg_days_between_orders * 2) THEN 'AT_RISK'
            WHEN days_since_last_order > avg_days_between_orders THEN 'DECLINING'
            ELSE 'ACTIVE'
        END as churn_risk_category,
        CASE
            WHEN total_spent >= 1000 THEN 'HIGH_VALUE'
            WHEN total_spent >= 500 THEN 'MEDIUM_VALUE'
            WHEN total_spent >= 100 THEN 'LOW_VALUE'
            ELSE 'MINIMAL_VALUE'
        END as value_segment
    FROM customer_activity
),
churn_scores AS (
    SELECT *,
        CASE
            WHEN churn_risk_category = 'NEVER_PURCHASED' THEN 10
            WHEN churn_risk_category = 'LONG_DORMANT' THEN 9
            WHEN churn_risk_category = 'DORMANT' THEN 8
            WHEN churn_risk_category = 'AT_RISK' THEN 6
            WHEN churn_risk_category = 'DECLINING' THEN 4
            ELSE 1
        END as churn_risk_score,
        CASE
            WHEN value_segment = 'HIGH_VALUE' THEN 4
            WHEN value_segment = 'MEDIUM_VALUE' THEN 3
            WHEN value_segment = 'LOW_VALUE' THEN 2
            ELSE 1
        END as value_score
    FROM churn_indicators
),
priority_matrix AS (
    SELECT *,
        churn_risk_score + value_score as total_priority_score,
        CASE
            WHEN churn_risk_score >= 8 AND value_score >= 3 THEN 'CRITICAL'
            WHEN churn_risk_score >= 6 AND value_score >= 2 THEN 'HIGH'
            WHEN churn_risk_score >= 4 OR value_score >= 3 THEN 'MEDIUM'
            ELSE 'LOW'
        END as intervention_priority
    FROM churn_scores
)
SELECT
    customer_id,
    country,
    customer_tier,
    total_orders,
    ROUND(total_spent, 2) as total_spent,
    ROUND(avg_order_value, 2) as avg_order_value,
    days_since_last_order,
    ROUND(avg_days_between_orders, 1) as avg_days_between_orders,
    churn_risk_category,
    value_segment,
    intervention_priority,
    churn_risk_score,
    value_score,
    total_priority_score,
    last_order_date,
    customer_lifespan_days
FROM priority_matrix
WHERE intervention_priority IN ('CRITICAL', 'HIGH', 'MEDIUM')
ORDER BY total_priority_score DESC, total_spent DESC;

-- =====================================================
-- QUERY 9: PROMOTIONAL CAMPAIGN EFFECTIVENESS
-- =====================================================

WITH campaign_periods AS (
    SELECT
        'BLACK_FRIDAY_2024' as campaign_name,
        '2024-11-24' as start_date,
        '2024-11-30' as end_date
    UNION ALL
    SELECT
        'SUMMER_SALE_2024' as campaign_name,
        '2024-06-15' as start_date,
        '2024-06-30' as end_date
    UNION ALL
    SELECT
        'HOLIDAY_2024' as campaign_name,
        '2024-12-20' as start_date,
        '2024-12-31' as end_date
),
campaign_performance AS (
    SELECT
        cp.campaign_name,
        cp.start_date,
        cp.end_date,
        COUNT(DISTINCT o.order_id) as campaign_orders,
        COUNT(DISTINCT o.customer_id) as campaign_customers,
        SUM(o.total_amount) as campaign_revenue,
        SUM(o.discount_amount) as total_discounts_given,
        AVG(o.total_amount) as avg_order_value,
        SUM(o.total_amount - o.discount_amount) as net_revenue,
        COUNT(DISTINCT CASE WHEN customer_first_order.first_order_date BETWEEN cp.start_date AND cp.end_date THEN o.customer_id END) as new_customers_acquired
    FROM campaign_periods cp
    LEFT JOIN orders o ON o.order_date BETWEEN cp.start_date AND cp.end_date
        AND o.order_status IN ('DELIVERED', 'SHIPPED')
    LEFT JOIN (
        SELECT customer_id, MIN(order_date) as first_order_date
        FROM orders
        WHERE order_status IN ('DELIVERED', 'SHIPPED')
        GROUP BY customer_id
    ) customer_first_order ON o.customer_id = customer_first_order.customer_id
    GROUP BY cp.campaign_name, cp.start_date, cp.end_date
),
baseline_comparison AS (
    SELECT
        cp.campaign_name,
        cp.start_date,
        cp.end_date,
        DATEDIFF(cp.end_date, cp.start_date) + 1 as campaign_days,
        -- Compare with same period previous year
        COUNT(DISTINCT prev_year.order_id) as prev_year_orders,
        SUM(prev_year.total_amount) as prev_year_revenue,
        COUNT(DISTINCT prev_year.customer_id) as prev_year_customers,
        AVG(prev_year.total_amount) as prev_year_avg_order_value
    FROM campaign_periods cp
    LEFT JOIN orders prev_year ON prev_year.order_date BETWEEN
        DATE_SUB(cp.start_date, INTERVAL 1 YEAR) AND
        DATE_SUB(cp.end_date, INTERVAL 1 YEAR)
        AND prev_year.order_status IN ('DELIVERED', 'SHIPPED')
    GROUP BY cp.campaign_name, cp.start_date, cp.end_date
),
campaign_analysis AS (
    SELECT
        cp.*,
        bc.prev_year_orders,
        bc.prev_year_revenue,
        bc.prev_year_customers,
        bc.campaign_days,
        ROUND(cp.campaign_revenue / bc.campaign_days, 2) as daily_revenue,
        ROUND(cp.total_discounts_given / cp.campaign_revenue * 100, 2) as discount_rate_pct,
        ROUND((cp.campaign_revenue - bc.prev_year_revenue) / bc.prev_year_revenue * 100, 2) as revenue_growth_pct,
        ROUND((cp.campaign_orders - bc.prev_year_orders) / bc.prev_year_orders * 100, 2) as order_growth_pct,
        ROUND(cp.campaign_revenue / cp.total_discounts_given, 2) as revenue_per_discount_dollar,
        ROUND(cp.new_customers_acquired / cp.total_discounts_given * 100, 2) as acquisition_cost_per_customer
    FROM campaign_performance cp
    JOIN baseline_comparison bc ON cp.campaign_name = bc.campaign_name
)
SELECT
    campaign_name,
    start_date,
    end_date,
    campaign_days,
    campaign_orders,
    campaign_customers,
    ROUND(campaign_revenue, 2) as campaign_revenue,
    ROUND(total_discounts_given, 2) as total_discounts_given,
    ROUND(avg_order_value, 2) as avg_order_value,
    new_customers_acquired,
    daily_revenue,
    discount_rate_pct,
    revenue_growth_pct,
    order_growth_pct,
    revenue_per_discount_dollar,
    acquisition_cost_per_customer,
    ROUND(prev_year_revenue, 2) as prev_year_revenue,
    CASE
        WHEN revenue_growth_pct >= 50 THEN 'EXCELLENT'
        WHEN revenue_growth_pct >= 25 THEN 'GOOD'
        WHEN revenue_growth_pct >= 0 THEN 'FAIR'
        ELSE 'POOR'
    END as campaign_performance_rating
FROM campaign_analysis
ORDER BY revenue_growth_pct DESC;

-- =====================================================
-- QUERY 10: MULTI-DIMENSIONAL BUSINESS INTELLIGENCE DASHBOARD
-- =====================================================

WITH time_dimensions AS (
    SELECT
        o.order_date,
        DATE_FORMAT(o.order_date, '%Y-%m') as year_month,
        YEAR(o.order_date) as year,
        MONTH(o.order_date) as month,
        QUARTER(o.order_date) as quarter,
        DAYOFWEEK(o.order_date) as day_of_week,
        CASE
            WHEN DAYOFWEEK(o.order_date) IN (1, 7) THEN 'WEEKEND'
            ELSE 'WEEKDAY'
        END as day_type,
        o.order_id,
        o.customer_id,
        o.total_amount,
        o.discount_amount,
        o.shipping_country,
        o.order_status
    FROM orders o
    WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 24 MONTH)
),
customer_dimensions AS (
    SELECT
        c.customer_id,
        c.customer_tier,
        c.country,
        CASE
            WHEN DATEDIFF(CURRENT_DATE, c.birth_date) / 365 < 25 THEN 'GEN_Z'
            WHEN DATEDIFF(CURRENT_DATE, c.birth_date) / 365 < 40 THEN 'MILLENNIAL'
            WHEN DATEDIFF(CURRENT_DATE, c.birth_date) / 365 < 55 THEN 'GEN_X'
            ELSE 'BOOMER'
        END as age_group,
        DATEDIFF(CURRENT_DATE, c.registration_date) as days_since_registration
    FROM customers c
),
product_dimensions AS (
    SELECT
        p.product_id,
        p.category_id,
        c.category_name,
        p.brand,
        CASE
            WHEN p.price < 50 THEN 'LOW_PRICE'
            WHEN p.price < 200 THEN 'MID_PRICE'
            WHEN p.price < 500 THEN 'HIGH_PRICE'
            ELSE 'PREMIUM'
        END as price_segment
    FROM products p
    JOIN categories c ON p.category_id = c.category_id
),
comprehensive_metrics AS (
    SELECT
        td.year_month,
        td.year,
        td.quarter,
        td.day_type,
        cd.customer_tier,
        cd.age_group,
        cd.country,
        pd.category_name,
        pd.brand,
        pd.price_segment,
        COUNT(DISTINCT td.order_id) as total_orders,
        COUNT(DISTINCT td.customer_id) as unique_customers,
        COUNT(DISTINCT oi.product_id) as unique_products_sold,
        SUM(td.total_amount) as total_revenue,
        SUM(td.discount_amount) as total_discounts,
        SUM(oi.quantity) as total_units_sold,
        AVG(td.total_amount) as avg_order_value,
        AVG(oi.quantity) as avg_units_per_order,
        SUM(oi.total_price) as gross_merchandise_value
    FROM time_dimensions td
    JOIN customer_dimensions cd ON td.customer_id = cd.customer_id
    JOIN order_items oi ON td.order_id = oi.order_id
    JOIN product_dimensions pd ON oi.product_id = pd.product_id
    WHERE td.order_status IN ('DELIVERED', 'SHIPPED')
    GROUP BY
        td.year_month, td.year, td.quarter, td.day_type,
        cd.customer_tier, cd.age_group, cd.country,
        pd.category_name, pd.brand, pd.price_segment
),
ranked_metrics AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY total_revenue DESC) as monthly_revenue_rank,
        ROW_NUMBER() OVER (PARTITION BY category_name ORDER BY total_revenue DESC) as category_revenue_rank,
        ROW_NUMBER() OVER (PARTITION BY brand ORDER BY total_revenue DESC) as brand_revenue_rank,
        PERCENT_RANK() OVER (PARTITION BY year_month ORDER BY total_revenue) as monthly_revenue_percentile,
        LAG(total_revenue) OVER (
            PARTITION BY customer_tier, age_group, country, category_name, brand, price_segment
            ORDER BY year_month
        ) as prev_month_revenue,
        SUM(total_revenue) OVER (
            PARTITION BY customer_tier, age_group, country, category_name, brand, price_segment
            ORDER BY year_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as rolling_3month_revenue
    FROM comprehensive_metrics
),
final_dashboard AS (
    SELECT
        year_month,
        year,
        quarter,
        day_type,
        customer_tier,
        age_group,
        country,
        category_name,
        brand,
        price_segment,
        total_orders,
        unique_customers,
        unique_products_sold,
        ROUND(total_revenue, 2) as total_revenue,
        ROUND(total_discounts, 2) as total_discounts,
        total_units_sold,
        ROUND(avg_order_value, 2) as avg_order_value,
        ROUND(avg_units_per_order, 2) as avg_units_per_order,
        ROUND(gross_merchandise_value, 2) as gross_merchandise_value,
        monthly_revenue_rank,
        category_revenue_rank,
        brand_revenue_rank,
        ROUND(monthly_revenue_percentile * 100, 2) as monthly_revenue_percentile,
        ROUND(total_discounts / total_revenue * 100, 2) as discount_rate_pct,
        ROUND(total_revenue / unique_customers, 2) as revenue_per_customer,
        ROUND(total_units_sold / unique_customers, 2) as units_per_customer,
        CASE
            WHEN prev_month_revenue IS NOT NULL AND prev_month_revenue > 0
            THEN ROUND((total_revenue - prev_month_revenue) / prev_month_revenue * 100, 2)
            ELSE NULL
        END as mom_growth_pct,
        ROUND(rolling_3month_revenue / 3, 2) as rolling_3month_avg_revenue
    FROM ranked_metrics
)
SELECT *
FROM final_dashboard
WHERE year_month >= DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH), '%Y-%m')
  AND total_revenue >= 1000  -- Filter for meaningful segments
ORDER BY year_month DESC, total_revenue DESC
LIMIT 1000;

-- =====================================================
-- PERFORMANCE OPTIMIZATION RECOMMENDATIONS
-- =====================================================

/*
PARTITIONING STRATEGY:
1. Orders table partitioned by year for time-based queries
2. Consider sub-partitioning by country for geographical analysis
3. Archive old partitions to maintain performance

INDEXING RECOMMENDATIONS:
1. Composite indexes on frequently filtered columns
2. Covering indexes for read-heavy queries
3. Partial indexes for specific query patterns

MAINTENANCE TASKS:
1. Regular ANALYZE TABLE for updated statistics
2. Archive old data beyond retention period
3. Monitor query performance with EXPLAIN plans
4. Consider read replicas for reporting queries

SCALING CONSIDERATIONS:
1. Implement horizontal partitioning for very large datasets
2. Use materialized views for complex aggregations
3. Consider column-store indexes for analytics
4. Implement proper connection pooling
*/