WITH base_data AS (
SELECT
  rep_id,
  battle_card_conversion,
  battle_card_expired,
  avg_days_to_win,
  cxi_improvement,
  margin,
  net_adds,
  disconnects,
  quality_of_sale,
  zero_usage_lines,
  ltv
FROM `project.dataset.sales_rep_metrics`
),

top10 AS (
SELECT *
FROM base_data
ORDER BY net_adds DESC
LIMIT 10
),

team_metrics AS (
SELECT 'battle_card_conversion' metric, battle_card_conversion value FROM base_data
UNION ALL SELECT 'battle_card_expired', battle_card_expired FROM base_data
UNION ALL SELECT 'avg_days_to_win', avg_days_to_win FROM base_data
UNION ALL SELECT 'cxi_improvement', cxi_improvement FROM base_data
UNION ALL SELECT 'margin', margin FROM base_data
UNION ALL SELECT 'net_adds', net_adds FROM base_data
UNION ALL SELECT 'disconnects', disconnects FROM base_data
UNION ALL SELECT 'quality_of_sale', quality_of_sale FROM base_data
UNION ALL SELECT 'zero_usage_lines', zero_usage_lines FROM base_data
UNION ALL SELECT 'ltv', ltv FROM base_data
),

top10_metrics AS (
SELECT 'battle_card_conversion' metric, battle_card_conversion value FROM top10
UNION ALL SELECT 'battle_card_expired', battle_card_expired FROM top10
UNION ALL SELECT 'avg_days_to_win', avg_days_to_win FROM top10
UNION ALL SELECT 'cxi_improvement', cxi_improvement FROM top10
UNION ALL SELECT 'margin', margin FROM top10
UNION ALL SELECT 'net_adds', net_adds FROM top10
UNION ALL SELECT 'disconnects', disconnects FROM top10
UNION ALL SELECT 'quality_of_sale', quality_of_sale FROM top10
UNION ALL SELECT 'zero_usage_lines', zero_usage_lines FROM top10
UNION ALL SELECT 'ltv', ltv FROM top10
),

team_avg AS (
SELECT
metric,
AVG(value) AS team_avg
FROM team_metrics
GROUP BY metric
),

top10_avg AS (
SELECT
metric,
AVG(value) AS top10_avg,
MAX(value) AS max_value,
MIN(value) AS min_value
FROM top10_metrics
GROUP BY metric
),

rep_distribution AS (
SELECT
tm.metric,
COUNTIF(tm.value > ta.team_avg) AS reps_above_avg
FROM top10_metrics tm
JOIN team_avg ta
ON tm.metric = ta.metric
GROUP BY tm.metric
)

SELECT
ta.metric,
ta.team_avg,
t10.top10_avg,

ROUND(t10.top10_avg / ta.team_avg,2) AS multiplier_vs_base,

CASE
WHEN (t10.top10_avg / ta.team_avg) >= 1
THEN CONCAT(ROUND(t10.top10_avg / ta.team_avg,2),'x higher')
ELSE CONCAT(ROUND((1 - (t10.top10_avg / ta.team_avg))*100,1),'% lower')
END AS insight_difference,

rd.reps_above_avg,
t10.max_value,
t10.min_value

FROM team_avg ta
JOIN top10_avg t10
ON ta.metric = t10.metric
JOIN rep_distribution rd
ON ta.metric = rd.metric
ORDER BY metric
