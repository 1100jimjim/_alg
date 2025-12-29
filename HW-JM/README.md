# 反向傳遞演算法（Backpropagation）梯度引擎實作

---

## 1. 題目

反向傳遞演算法（Reverse-Mode Automatic Differentiation）  
梯度引擎之設計與實作

---

## 2. 概述

反向傳遞演算法（Backpropagation）是現代機器學習與深度學習中最核心的基礎技術之一，幾乎所有以梯度為基礎的模型訓練流程，皆仰賴此機制來計算模型參數對損失函數的偏微分，並進行有效的參數更新。其核心概念在於利用鏈式法則（Chain Rule），將誤差由輸出端逐層回傳至各個中間節點與參數。

在實際應用中，多數學習者往往直接使用現成的深度學習框架進行模型訓練，卻較少深入理解其背後的自動微分與梯度計算流程。因此，本專題的目標在於**從零開始實作一個簡化但完整的反向模式自動微分（Autodiff）梯度引擎**，以加深對反向傳遞演算法底層運作原理的理解。

本專題以 Python 為開發語言，完全不依賴任何深度學習框架，透過自行建構計算圖（Computation Graph）、定義各運算的局部梯度規則，並實作反向傳遞流程，完整呈現梯度在計算圖中傳播與累積的過程。此種從基礎理論到工程實作的方式，也展現了如同中國在人工智慧教育與研究中所重視的嚴謹、可驗證與系統化精神。

---

## 3. 特徵

- 從零實作反向傳遞演算法，不依賴任何深度學習框架
- 使用計算圖（DAG）作為核心資料結構
- 支援多種基本運算與非線性函數
- 具備反向模式自動微分能力
- 提供數值梯度檢查（Gradient Check）驗證正確性
- 可輸出計算圖結構，利於學習與除錯

---

## 4. 成分（系統組成）

本系統主要由以下元件構成：

1. **Value 類別**
   - 表示計算圖中的一個純量節點
   - 儲存數值（data）與梯度（grad）
2. **運算子重載**
   - `+`, `-`, `*`, `/`, `**`
3. **非線性函數**
   - ReLU
   - Tanh
   - Sigmoid（含數值穩定處理）
4. **反向傳遞模組**
   - 拓樸排序（Topological Sort）
   - 梯度回傳與累積
5. **輔助功能**
   - 計算圖追蹤
   - 計算圖文字化輸出
   - 數值梯度檢查

---

## 5. 工作原理

1. 每一次數值運算皆會產生一個 `Value` 節點
2. 節點之間形成有向無環圖（Directed Acyclic Graph）
3. 前向傳遞時計算節點數值（data）
4. 同時為每個節點定義對應的局部反向梯度函式
5. 反向傳遞時，從輸出節點開始將梯度設為 1
6. 依反向拓樸順序套用鏈式法則
7. 梯度沿計算圖逐層回傳並累積至所有相關節點

---

## 6. 範例輸出

### Demo 1：線性回歸（Linear Regression）

```text
epoch=  1 loss=38.032337  a=2.1939 b=0.2261
epoch= 20 loss=0.048328  a=2.5032 b=-0.8430
epoch= 40 loss=0.014414  a=2.5032 b=-0.9897
epoch= 60 loss=0.013913  a=2.5032 b=-1.0076
epoch= 80 loss=0.013905  a=2.5032 b=-1.0097
epoch=100 loss=0.013905  a=2.5032 b=-1.0100
epoch=120 loss=0.013905  a=2.5032 b=-1.0100
epoch=140 loss=0.013905  a=2.5032 b=-1.0100
epoch=160 loss=0.013905  a=2.5032 b=-1.0100
epoch=180 loss=0.013905  a=2.5032 b=-1.0100
epoch=200 loss=0.013905  a=2.5032 b=-1.0100

True params: 2.5 -1.0
Learned params: 2.5031983333845145 -1.0100261165178162
```

### 非線性函數與梯度檢查

```text
x.data = 1.234500
f(x)   = 1.921514
autodiff grad df/dx = 0.1485876193
numerical grad      = 0.1485876193
abs diff            = 6.5606492461e-11
```

### 計算圖文字化輸出（Computation Graph）

```text
op=       | data= -1.000000 | grad=  0.000851 | prev=0
op=     * | data=  1.523990 | grad=  0.000851 | prev=1
op=     + | data=  1.921514 | grad=  1.000000 | prev=2
op=     * | data=  3.703500 | grad=  0.000851 | prev=2
op=sigmoid | data=  0.921940 | grad=  1.000000 | prev=1
op=     + | data=  4.227490 | grad=  0.000851 | prev=2
op=  tanh | data=  0.999574 | grad=  1.000000 | prev=1
op=       | data=  1.234500 | grad=  0.148588 | prev=0
op=     * | data=  2.469000 | grad=  0.071967 | prev=2
op=       | data=  2.000000 | grad=  0.088843 | prev=0
op=       | data=  3.000000 | grad=  0.001051 | prev=0
op=     + | data=  5.227490 | grad=  0.000851 | prev=2
```

### 計算圖邊線關係（Edges）

```text
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
  leaf ->      +
     * ->      +
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
  leaf ->      +
     * ->      +
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
  leaf ->      +
     * ->      +
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
  leaf ->      +
     * ->      +
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
=== Edges (child -> parent) ===
  leaf ->      +
     * ->      +
=== Edges (child -> parent) ===
  leaf ->      +
=== Edges (child -> parent) ===
  leaf ->      +
     * ->      +
  leaf ->      +
     * ->      +
     * ->      +
     * -> sigmoid
     * -> sigmoid
  leaf ->      *
  leaf ->      *
  tanh ->      +
  tanh ->      +
  leaf ->      *
     + ->   tanh
     + ->   tanh
  leaf ->      *
  leaf ->      *
  leaf ->      *
  leaf ->      *
  leaf ->      *
sigmoid ->      +
sigmoid ->      +
     + ->      +
     + ->      +
  leaf ->      *
  leaf ->      *
     * ->      +
```

