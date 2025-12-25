---
title: '拓扑排序（Topological Sort）'
date: '2025-12-25T19:30:10+08:00'
draft: false
series: '数据结构与算法优化'
---

## 拓扑排序（Topological Sort）

### 核心场景（详细版）

拓扑排序仅适用于**有向无环图（DAG）**，核心目标是对顶点进行线性排序，满足「若存在从顶点 i 到顶点 j 的有向路径，则 i 在排序结果中一定出现在 j 之前」。

- 典型应用：任务调度（依赖任务必须先执行）、课程安排（先修课需先学）；

- 额外需求：若图中存在环（非 DAG），则无法完成拓扑排序，需输出 “not acyclic” 或返回空。

  <!--More-->

### 算法思想（详细拆解，Kahn 算法 —— 迭代版最优）

Kahn 算法是拓扑排序的经典迭代实现（比递归版更稳定，避免栈溢出），核心围绕「入度」展开，步骤拆解如下：

1. **核心概念定义**：
   - 入度（In-Degree）：指向某顶点的边的数量（如顶点 v 有 3 条入边，则入度为 3）；
   - 入度为 0 的顶点：无前置依赖，可作为排序的起点。
2. **算法步骤**：
   - 步骤 1：统计所有顶点的入度（构建入度数组）。遍历邻接表，对每个顶点 u 的邻接顶点 v，v 的入度 + 1（因为 u→v 是一条入边）。
   - 步骤 2：初始化队列，将所有「入度为 0」的顶点入队（这些顶点无前置依赖，可优先处理）。
   - 步骤 3：循环处理队列中的顶点：
     - 取出队首顶点 u，加入排序结果；
     - 遍历 u 的所有邻接顶点 v：将 v 的入度 - 1（因为 u 已处理，v 的一个前置依赖已完成）；
     - 若 v 的入度减为 0，将 v 入队（v 的所有前置依赖已完成，可处理）。
   - 步骤 4：判环逻辑：若最终排序结果的顶点数 ≠ 总顶点数，说明图中存在环（环内顶点的入度永远无法减为 0，不会被加入结果）。

### 通用精简代码（注释清晰，脱离题目限制）

```java
import java.util.*;

public class TopologicalSort {
    // 通用拓扑排序：返回排序结果（空列表表示有环）
    // 参数：n=顶点数（0~n-1），adj=邻接表（adj[u]存储u的所有出边顶点v）
    public static List<Integer> topologicalSort(int n, List<List<Integer>> adj) {
        int[] inDegree = new int[n]; // 入度数组
        // 步骤1：统计入度
        for (int u = 0; u < n; u++) {
            for (int v : adj.get(u)) {
                inDegree[v]++;
            }
        }

        // 步骤2：入度为0的顶点入队
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        // 步骤3：核心排序逻辑
        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            result.add(u);
            // 处理u的邻接顶点，入度减1
            for (int v : adj.get(u)) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    queue.offer(v);
                }
            }
        }

        // 步骤4：判环（结果长度≠顶点数 → 有环）
        return result.size() == n ? result : Collections.emptyList();
    }

    // 复用入口（适配各类拓扑排序题）
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(); // 顶点数
        int m = sc.nextInt(); // 边数

        // 初始化邻接表（通用0基，题目若1基则转换）
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int i = 0; i < m; i++) {
            int u = sc.nextInt() - 1; // 1基转0基（按需调整）
            int v = sc.nextInt() - 1;
            adj.get(u).add(v);
        }

        List<Integer> res = topologicalSort(n, adj);
        if (res.isEmpty()) {
            System.out.println("not acyclic"); // 判环输出
        } else {
            // 输出：0基转回1基（按需调整）
            for (int i = 0; i < res.size(); i++) {
                if (i > 0) System.out.print(" ");
                System.out.print(res.get(i) + 1);
            }
        }
        sc.close();
    }
}
```

### 核心考点（详细解释）

1. **算法效率**：时间复杂度 O (n+m)（n = 顶点数，m = 边数），每个顶点和边仅处理一次，是拓扑排序的最优复杂度；
2. **判环逻辑本质**：环内的所有顶点入度永远无法减为 0（环内顶点互相依赖），因此不会被加入结果，结果长度小于总顶点数；
3. **迭代版 vs 递归版**：递归版（基于 DFS 后序遍历）易因顶点数过多（如 n>1000）导致栈溢出，优先用 Kahn 迭代版；
4. **顶点编号适配**：题目中顶点可能从 1 开始（如样例输入），代码中统一转 0 基处理（避免数组越界），输出时再转回 1 基。

### 易错点（详细说明错误原因 + 正确做法）

| 易错点         | 错误原因                           | 正确做法                                   |
| -------------- | ---------------------------------- | ------------------------------------------ |
| 入度统计遗漏   | 仅遍历边的起点，未更新终点入度     | 遍历 adj [u] 的每个 v，执行 inDegree [v]++ |
| 顶点编号未转换 | 题目 1 基，代码用 0 基导致数组越界 | 输入时 u/v-1 转 0 基，输出时 + 1 转回 1 基 |
| 判环逻辑错误   | 仅判断队列是否为空，未校验结果长度 | 必须用 result.size () == n 判断是否有环    |
| 递归版栈溢出   | 顶点数多（如 n=10000）             | 改用 Kahn 迭代版（队列实现）               |

### 总结（辅助记忆）

拓扑排序核心口诀：「入度为 0 入队 → 处理顶点减邻接入度 → 入度为 0 再入队 → 结果长度判环」；

核心数据结构：入度数组 + 队列；

核心适用场景：DAG 排序、依赖调度、判环。