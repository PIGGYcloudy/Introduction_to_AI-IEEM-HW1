# 利用KNN的方法填補缺失值
- 二元變數變成0, 1
- 日期以天為單位
- 本題目沒有多類別變數

最後利用四捨五入轉換回來

## 以下是Random Forest Regressor在缺失值填補方法的Public Score (n estimators = 500)
| KNN | train dropna, test 用眾數/平均數 |
| --- | --- |
| 0.05441 | 0.05038 |