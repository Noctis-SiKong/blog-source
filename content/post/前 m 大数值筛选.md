---
title: '前 m 大数值筛选'
date: '2025-12-25T19:30:10+08:00'
draft: false
series: '数据结构与算法优化'
---

# 前 m 大数值筛选（二分答案 + 局部排序）

## 一、题目核心分析与算法思想

### 1. 题目特点

给定 n 个整数（0 <n,m < 1e6），数值范围 [-500000,500000]，要求输出前 m 大的数（降序）。

- 关键限制：n/m 规模接近 1e6，**不能用整体排序（如 Arrays.sort 整体 O (n log n)）或冒泡排序（O (m²)）**，否则超时；

- 核心思路：利用「二分答案法」找到第 m 大的数，仅收集前 m 个数做局部排序，将时间复杂度从 O (n log n) 优化为 O (n log (数值范围)) + O (m log m)，适配大规模数据。

  <!--More-->

### 2. 核心算法思想

| 算法思想             | 应用场景                             | 核心逻辑                                                     |
| -------------------- | ------------------------------------ | ------------------------------------------------------------ |
| 二分答案（二分查找） | 找第 k 大 / 小值、数值范围有限的场景 | 对数值范围二分，通过「统计大于当前中间值的数的个数」判断第 m 大值的位置 |
| 计数统计             | 快速判断数值分布                     | 统计大于 / 等于某个值的数的个数，辅助二分判断                |
| 局部排序             | 大规模数据筛选后排序                 | 仅对前 m 个数排序，而非整体排序，减少排序规模                |
| 快速 IO              | 百万级数据输入输出                   | 用 BufferedReader/PrintWriter 替代 Scanner/System.out，避免 IO 超时 |

## 二、算法步骤拆解（配优化代码）

### 1. 完整优化代码（思想标注 + 命名规范 + 性能优化）

```java
import java.io.*;
import java.util.Arrays;

class Main {
    public static void main(String[] s) throws Exception {
        // 快速IO：百万级数据必须用BufferedReader/PrintWriter，避免Scanner超时
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(System.out));
        
        String line;
        while ((line = br.readLine()) != null) {
            // 步骤1：读取n和m
            String[] nm = line.trim().split(" ");
            int n = Integer.parseInt(nm[0]);
            int m = Integer.parseInt(nm[1]);
            
            // 步骤2：读取数组，同时确定数值范围（二分的左右边界）
            String[] valStr = br.readLine().trim().split(" ");
            int[] arr = new int[n];
            int minVal = 500000;  // 数值下界
            int maxVal = -500000; // 数值上界
            for (int i = 0; i < n; i++) {
                arr[i] = Integer.parseInt(valStr[i]);
                minVal = Math.min(minVal, arr[i]);
                maxVal = Math.max(maxVal, arr[i]);
            }
            
            // 步骤3：二分答案找「第m大的数target」
            int left = minVal, right = maxVal;
            int target = 0;
            while (left <= right) {
                int mid = (left + right) / 2;
                // 统计大于mid的数的个数（核心：判断mid是否小于第m大的数）
                int countGreater = countGreaterThan(arr, mid);
                
                if (countGreater >= m) {
                    // 大于mid的数≥m个 → 第m大的数在mid右侧
                    left = mid + 1;
                } else {
                    // 统计等于mid的数的个数
                    int countEqual = countEqual(arr, mid);
                    if (countGreater + countEqual >= m) {
                        // 大于+等于mid的数≥m个 → mid就是第m大的数
                        target = mid;
                        break;
                    } else {
                        // 第m大的数在mid左侧
                        right = mid - 1;
                    }
                }
            }
            
            // 步骤4：收集前m大的数（先收集大于target的，再补等于target的）
            int[] result = new int[m];
            int idx = 0;
            // 先加大于target的数
            for (int num : arr) {
                if (num > target && idx < m) {
                    result[idx++] = num;
                }
            }
            // 补等于target的数（凑够m个）
            if (idx < m) {
                for (int num : arr) {
                    if (num == target && idx < m) {
                        result[idx++] = num;
                    }
                }
            }
            
            // 步骤5：局部排序（降序）→ 替换原冒泡排序，用Arrays.sort+反转，O(m log m)更高效
            Arrays.sort(result);
            reverseArray(result); // 升序转降序
            
            // 步骤6：输出结果（格式控制：首尾无空格）
            for (int i = 0; i < m; i++) {
                if (i > 0) pw.print(" ");
                pw.print(result[i]);
            }
            pw.println();
        }
        
        // 释放资源
        pw.flush();
        pw.close();
        br.close();
    }
    
    // 辅助：统计数组中大于num的数的个数（O(n)）
    private static int countGreaterThan(int[] arr, int num) {
        int count = 0;
        for (int val : arr) {
            if (val > num) count++;
        }
        return count;
    }
    
    // 辅助：统计数组中等于num的数的个数（O(n)）
    private static int countEqual(int[] arr, int num) {
        int count = 0;
        for (int val : arr) {
            if (val == num) count++;
        }
        return count;
    }
    
    // 辅助：数组反转（升序转降序）
    private static void reverseArray(int[] arr) {
        int left = 0, right = arr.length - 1;
        while (left < right) {
            int temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            left++;
            right--;
        }
    }
}
```

### 2. 关键步骤解析（算法思想落地）

#### （1）快速 IO 处理（百万级数据必备）

- 思想：Scanner/System.out.println 在百万级数据下会超时，BufferedReader 按行读取、PrintWriter 批量输出，IO 效率提升 10 倍以上；
- 代码要点：`br.readLine()`读取整行，`trim()`去除首尾空格，`split(" ")`分割字符串。

#### （2）二分答案找第 m 大的数（核心）

- 思想：「二分答案」是将 “找第 m 大值” 转化为 “判断某个值是否是第 m 大值”，通过数值范围（[-5e5,5e5]）二分，每次判断仅需 O (n)，总时间 O (n log 1e6) ≈ O (n*20)，远快于整体排序；
- 逻辑拆解：
  1. 二分边界：left = 数组最小值，right = 数组最大值；
  2. 中间值 mid：判断 “大于 mid 的数的个数 countGreater”；
  3. 分支 1（countGreater ≥ m）：说明第 m 大的数比 mid 大，左边界右移（left=mid+1）；
  4. 分支 2（countGreater < m）：统计等于 mid 的数 countEqual，若 countGreater+countEqual ≥ m → mid 就是第 m 大的数；否则右边界左移（right=mid-1）。

#### （3）收集前 m 大的数

- 思想：先收集所有大于 target 的数（这些数一定在前 m 大里），再补等于 target 的数（凑够 m 个），避免收集无关数据；
- 注意：无需收集小于 target 的数，因为 target 是第 m 大的数，小于它的数不可能进入前 m。

#### （4）局部排序优化

- 原问题：原代码用冒泡排序（O (m²)），m=1e6 时完全超时；
- 优化方案：用 Arrays.sort（O (m log m)）先升序排序，再反转数组得到降序，时间复杂度从 O (m²) 降到 O (m log m)，适配大规模 m。

## 三、期末复习核心考点

### 1. 算法思想考点

| 考点               | 核心提问方式                               | 解题关键                                                     |
| ------------------ | ------------------------------------------ | ------------------------------------------------------------ |
| 二分答案法         | 如何找第 k 大 / 小值？为什么不用整体排序？ | 数值范围有限时，二分答案 + 计数统计，时间复杂度更低（O (n log 数值范围)） |
| 大规模数据 IO 优化 | 百万级数据输入输出为什么超时？             | 必须用 BufferedReader/PrintWriter，避免 Scanner/System.out 的逐字符处理 |
| 局部排序           | 为什么只对前 m 个数排序？                  | 整体排序 O (n log n)，局部排序 O (m log m)，m<n 时大幅减少计算量 |

### 2. 易错点总结（期末避坑）

1. 二分边界错误：
   - 忘记统计 “等于 mid 的数”，直接用 countGreater 判断，导致漏判 target；
   - 二分终止条件写成`left < right`，可能导致 target 未找到；
2. IO 超时：
   - 用 Scanner 读取百万级数据，或用 System.out.println 逐行输出；
3. 排序效率：
   - 对 m=1e6 的数组用冒泡排序，时间复杂度爆炸；
4. 数值范围：
   - 初始化 minVal/maxVal 时写反（比如 minVal 初始 - 5e5，maxVal 初始 5e5），导致二分边界错误。

### 3. 同类题型扩展（举一反三）

| 题型                | 核心思想复用              | 差异点                                       |
| ------------------- | ------------------------- | -------------------------------------------- |
| 找第 k 小的数       | 二分答案 + 计数统计       | 统计 “小于 mid 的数的个数”，而非 “大于”      |
| 数值范围 [-1e9,1e9] | 二分答案依然可用          | 需先遍历数组确定 min/max，而非硬编码数值范围 |
| 有重复数的前 m 大   | 收集时需包含重复的 target | 本题已兼容，无需额外修改                     |

## 四、复习建议

1. 背记二分答案模板：

   ```java
   int left = 最小值, right = 最大值;
   int target = 0;
   while (left <= right) {
       int mid = (left + right) / 2;
       int count = 统计符合条件的数;
       if (count >= m) {
           left = mid + 1;
       } else {
           // 补充等于mid的统计，判断是否找到target
       }
   }
   ```

   

2. 手动模拟样例：

   - 样例输入 1：5 3，数组 [3,-35,92,213,-644]
     - 数值范围：-644~213
     - 二分找到 target=3（第 3 大的数）
     - 收集大于 3 的数：213、92，再补等于 3 的数，凑够 3 个；
     - 排序后输出：213 92 3；

3. 重点记忆 IO 优化和局部排序：

   - 快速 IO 的固定写法（BufferedReader+PrintWriter）；
   - Arrays.sort + 反转实现降序的固定逻辑。

## 五、核心总结

这道题的核心是「用二分答案替代整体排序」，适配大规模数据场景：

1. 二分答案找第 m 大值，将时间复杂度从 O (n log n) 降到 O (n log 数值范围)；
2. 局部排序仅处理前 m 个数，进一步降低排序成本；
3. 快速 IO 是百万级数据的必备优化。