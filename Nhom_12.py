from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, year, avg
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Khởi tạo Spark session
spark = SparkSession.builder \
  .appName("StoreSalesAnalysis") \
  .config("spark.executor.memory", "4g") \
  .config("spark.driver.memory", "4g") \
  .config("spark.memory.fraction", "0.8") \
  .config("spark.executor.heartbeatInterval", "60s") \
  .config("spark.network.timeout", "120s") \
  .getOrCreate()




#Đọc file CSV từ đường dẫn HDFS
df = spark.read.option("header", "true").option("inferSchema", "true") \
   .csv("hdfs://localhost:9000/storesalesdata/storesales.csv")


# Tạo một bảng tạm (temporary view) có tên "sales" từ DataFrame df để chạy Spark SQL
df.createOrReplaceTempView("sales")






query1 = spark.sql("""
    SELECT
      Year,
      `Category of Goods`,
      Segment,
      ROUND(AVG(Discount) * 100, 2) AS AvgDiscountPercent,
      ROUND(AVG(Profit), 2) AS AvgProfit,
      ROUND(SUM(Profit), 2) AS TotalProfit
  FROM Sales
  GROUP BY Year, `Category of Goods`, Segment
  ORDER BY `Category of Goods`, Segment, Year
""")
query1.show()


query2 = spark.sql("""SELECT
      `Product Name`,
      SUM(Quantity) AS TotalQuantity,
      ROUND(SUM(Sales), 2) AS TotalSales,
      ROUND(SUM(Profit), 2) AS TotalProfit
  FROM Sales
  GROUP BY `Product Name`
  ORDER BY TotalProfit DESC
  LIMIT 20""")
query2.show()




query3 = spark.sql("""
   SELECT
       `City Type`,
       `Product Name`,
       ROUND(Total_Profit, 2) AS Total_Profit
   FROM (
       SELECT
           `City Type`,
           `Product Name`,
           SUM(Profit) AS Total_Profit,
           RANK() OVER (
               PARTITION BY `City Type`
               ORDER BY SUM(Profit) DESC
           ) AS rank
       FROM sales
       GROUP BY `City Type`, `Product Name`
   ) ranked_sales
   WHERE rank <= 3
   ORDER BY `City Type`, Total_Profit DESC
""")
query3.show()






query4 = spark.sql("""
   SELECT
       `Ship Mode`,
       ROUND(AVG(DATEDIFF(`Ship Date`, `Order Date`)), 1) AS Avg_Delivery_Days,
       ROUND(AVG(Profit), 2) AS Avg_Profit,
       COUNT(*) AS Total_Orders
   FROM sales
   WHERE `Ship Date` IS NOT NULL AND `Order Date` IS NOT NULL
   GROUP BY `Ship Mode`
   ORDER BY Avg_Delivery_Days ASC
""")
query4.show()






query5 = spark.sql("""
   SELECT
       Year,
       Month,
       ROUND(Monthly_Sales, 2) AS Monthly_Sales
   FROM (
       SELECT
           YEAR(`Order Date`) AS Year,
           MONTH(`Order Date`) AS Month,
           SUM(Sales) AS Monthly_Sales,
           RANK() OVER (PARTITION BY YEAR(`Order Date`) ORDER BY SUM(Sales) DESC) AS rank
       FROM sales
       WHERE `Order Date` IS NOT NULL
       GROUP BY YEAR(`Order Date`), MONTH(`Order Date`)
   ) ranked_sales
   WHERE rank = 1
   ORDER BY Year
""")
query5.show()




query6 = spark.sql("""
   SELECT
       Year,
       State,
       ROUND(Sales, 2) AS Total_Sales,
       ROUND(Profit, 2) AS Total_Profit,
       ROUND(Profit_Ratio, 4) AS Profit_Ratio,
       ROUND(AVG(Profit_Ratio) OVER (PARTITION BY Year), 4) AS Year_Avg_Ratio,
       ROUND(STDDEV(Profit_Ratio) OVER (PARTITION BY Year), 4) AS Year_Std_Dev,
       CASE
           WHEN ABS(Profit_Ratio - AVG(Profit_Ratio) OVER (PARTITION BY Year)) >
                1 * STDDEV(Profit_Ratio) OVER (PARTITION BY Year)
           THEN 'Anomaly'
           ELSE 'Normal'
       END AS Status
   FROM (
       SELECT
           YEAR(`Order Date`) AS Year,
           State,
           SUM(Sales) AS Sales,
           SUM(Profit) AS Profit,
           SUM(Profit) / SUM(Sales) AS Profit_Ratio
       FROM sales
       WHERE `Order Date` IS NOT NULL AND Sales > 0
       GROUP BY YEAR(`Order Date`), State
   ) AS grouped_data
""")
query6.show()




query7 = spark.sql("""
   SELECT
       Year,
       Segment,
       ROUND(SUM(Profit), 2) AS Total_Profit,
       ROUND(SUM(Sales), 2) AS Total_Sales,
       ROUND(SUM(Profit) / SUM(Sales), 4) AS Profit_Ratio,
       NTILE(4) OVER (PARTITION BY Year ORDER BY SUM(Profit) DESC) AS Profit_Quartile,
       CASE
           WHEN NTILE(4) OVER (PARTITION BY Year ORDER BY SUM(Profit) DESC) = 1 THEN 'Top 25%'
           WHEN NTILE(4) OVER (PARTITION BY Year ORDER BY SUM(Profit) DESC) = 2 THEN 'Upper Middle 25%'
           WHEN NTILE(4) OVER (PARTITION BY Year ORDER BY SUM(Profit) DESC) = 3 THEN 'Lower Middle 25%'
           ELSE 'Bottom 25%'
       END AS Performance_Group
   FROM sales
   WHERE `Order Date` IS NOT NULL
   GROUP BY Year, Segment
   ORDER BY Year, Profit_Quartile
""")
query7.show()


query8 = spark.sql("""
                  SELECT
                      Year, `Ship Mode`, ROUND(AVG (Discount), 4) AS Avg_Discount, ROUND(AVG (DATEDIFF(`Ship Date`, `Order Date`)), 2) AS Avg_Shipping_Days, ROUND(AVG (Profit), 2) AS Avg_Profit, ROUND(CORR(Discount, Profit), 4) AS Discount_Profit_Corr, ROUND(CORR(DATEDIFF(`Ship Date`, `Order Date`), Profit), 4) AS ShippingTime_Profit_Corr, CASE
                      WHEN ABS(CORR(Discount, Profit)) > 0.3 THEN 'Affects Profit'
                      ELSE 'No Significant Effect'
                  END
                  AS Discount_Influence,


       CASE
           WHEN ABS(CORR(DATEDIFF(`Ship Date`, `Order Date`), Profit)) > 0.3 THEN 'Affects Profit'
           ELSE 'No Significant Effect'
                  END
                  AS ShippingTime_Influence


   FROM (
       SELECT
           YEAR(`Order Date`) AS Year,
           `Ship Mode`,
           Discount,
           Profit,
           `Order Date`,
           `Ship Date`
       FROM sales
       WHERE `Order Date` IS NOT NULL AND `Ship Date` IS NOT NULL
   ) sub
   GROUP BY Year, `Ship Mode`
   ORDER BY Year, Avg_Profit DESC
                  """)
query8.show()






query9 = """
SELECT
   `City Type`,
   `Outlet Type`,
   COUNT(DISTINCT `Order ID`) AS Total_Orders,
   ROUND(AVG(Sales), 2) AS Average_Sales,
   ROUND(AVG(Profit), 2) AS Average_Profit
FROM sales
WHERE Year >= 2022
GROUP BY `City Type`, `Outlet Type`
HAVING AVG(Sales) > 10000
ORDER BY `City Type`, Average_Sales DESC
"""
spark.sql(query9).show()


query10 = """
WITH RankedProducts AS (
 SELECT
   `Region`,
   `Category of Goods`,
   `Sub-Category`,
   ROUND(SUM(Sales), 2) AS Total_Sales,
   ROUND(SUM(Profit), 2) AS Total_Profit,
   ROW_NUMBER() OVER (
     PARTITION BY `Region`, `Category of Goods`
     ORDER BY SUM(Sales) DESC, SUM(Profit) DESC
   ) AS RankInGroup
 FROM sales
 WHERE Year >= 2022
 GROUP BY `Region`, `Category of Goods`, `Sub-Category`
)


SELECT
 `Region`,
 `Category of Goods`,
 `Sub-Category`,
 Total_Sales,
 Total_Profit
FROM RankedProducts
WHERE RankInGroup = 1
ORDER BY Region, `Category of Goods`
"""
spark.sql(query10).show()






query11_extended = """
WITH Top5Customers AS (
 SELECT
   `Customer ID`,
   `Customer Name`,
   `Last Name`,
   ROUND(SUM(Sales), 2) AS Total_Sales
 FROM sales
 GROUP BY `Customer ID`, `Customer Name`, `Last Name`
 ORDER BY Total_Sales DESC
 LIMIT 5
),


CustomerTopCategory AS (
 SELECT
   `Customer ID`,
   `Category of Goods`,
   SUM(Quantity) AS Total_Quantity,
   ROW_NUMBER() OVER (
     PARTITION BY `Customer ID`
     ORDER BY SUM(Quantity) DESC
   ) AS RankInCategory
 FROM sales
 GROUP BY `Customer ID`, `Category of Goods`
)


SELECT
 t5.`Customer ID`,
 CONCAT(t5.`Customer Name`, ' ', t5.`Last Name`) AS Full_Name,
 t5.Total_Sales,
 ctc.`Category of Goods` AS Most_Purchased_Category
FROM Top5Customers t5
LEFT JOIN CustomerTopCategory ctc
 ON t5.`Customer ID` = ctc.`Customer ID` AND ctc.RankInCategory = 1
ORDER BY t5.Total_Sales DESC
"""
spark.sql(query11_extended).show()




query12 = """
SELECT
Year,
`Ship Mode` AS Ship_Mode,
Segment,
COUNT(DISTINCT `Order ID`) AS Total_Orders,
SUM(Quantity) AS Total_Quantity,
ROUND(SUM(Sales), 2) AS Total_Sales,
ROUND(SUM(Profit), 2) AS Total_Profit,
ROUND(AVG(Sales), 2) AS Avg_Sales_Per_Order,
ROUND(AVG(Profit), 2) AS Avg_Profit_Per_Order,
ROUND(SUM(Profit) / NULLIF(SUM(Sales),0) * 100, 2) AS Profit_Margin_Percent
FROM sales
GROUP BY Year, `Ship Mode`, Segment
HAVING SUM(Sales) > 10000
ORDER BY Year ASC, Total_Sales DESC, Total_Profit DESC
"""
spark.sql(query12).show(n=30, truncate=False)






# câu sử dụng Mlib
# 2.1. Xử lý missing value
df = df.dropna(subset=["Profit", "Sales", "Discount", "Quantity"])




# 3. Tạo thêm feature bổ sung nếu có
if "Order Date" in df.columns:
 df = df.withColumn("OrderMonth", month(col("Order Date")))
 df = df.withColumn("OrderYear", year(col("Order Date")))








# 4. Mã hóa các cột phân loại
categorical_cols = ["Category of Goods", "Region", "City Type", "Outlet Type", "Ship Mode"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index") for col in categorical_cols if col in df.columns]
for indexer in indexers:
 df = indexer.fit(df).transform(df)








# 5. Vector hóa các đặc trưng đầu vào
generated_features = ["Sales", "Discount", "Quantity"]
if "OrderMonth" in df.columns: generated_features.append("OrderMonth")
if "OrderYear" in df.columns: generated_features.append("OrderYear")
feature_cols = generated_features + [col + "_Index" for col in categorical_cols if col in df.columns]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", "Profit")








# 6. Chia dữ liệu
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)








# 7. Huấn luyện mô hình RandomForest để lấy feature importance
rf = RandomForestRegressor(featuresCol="features", labelCol="Profit", numTrees=100, maxDepth=10, seed=42)
model = rf.fit(train_data)
predictions = model.transform(test_data)








# 8. Đánh giá độ chính xác
evaluator = RegressionEvaluator(labelCol="Profit", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(f"\n R-squared on Profit prediction: {r2:.3f}")








# 9. Phân tích độ quan trọng của đặc trưng
feature_names = assembler.getInputCols()
importances = model.featureImportances.toArray()
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)








print("\n Feature Importance:")
print(importance_df)








plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", palette="Blues_d", legend=False)
plt.title("Feature Importance for Predicting Profit")
plt.tight_layout()
plt.show()
# 10. Phân tích mối quan hệ giữa Discount và Profit theo từng Category
discount_profit_df = df.groupBy("Category of Goods", "Discount") \
 .agg(avg("Profit").alias("AvgProfit"))








pdf = discount_profit_df.toPandas()








plt.figure(figsize=(10, 6))
sns.lineplot(data=pdf, x="Discount", y="AvgProfit", hue="Category of Goods", marker="o")
plt.axhline(0, color="black", linestyle="--")
plt.title("Ảnh hưởng của mức giảm giá đến lợi nhuận trung bình theo loại hàng")
plt.xlabel("Tỷ lệ giảm giá")
plt.ylabel("Lợi nhuận trung bình")
plt.grid(True)
plt.tight_layout()
plt.show()





